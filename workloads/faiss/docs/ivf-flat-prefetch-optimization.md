# FAISS IVFFlat GPU Kernel Prefetch 优化分析

## 概述

本文档分析在 FAISS GPU IVFFlat 搜索中使用 PTX `prefetch.global.L2` 指令优化 UVM 性能的可行性。

## 当前代码结构分析

### 关键文件

```
faiss/gpu/impl/IVFFlatScan.cu      - IVFFlat 搜索的核心 kernel
faiss/gpu/impl/IVFFlat.cu          - IVFFlat 索引实现
faiss/gpu/impl/IVFBase.cu          - IVF 基类，包含粗量化逻辑
faiss/gpu/utils/PtxUtils.cuh       - PTX 内联汇编工具函数
```

### 搜索流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IVFFlat GPU 搜索流程                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. IVFFlat::search() [IVFFlat.cu:182-212]                                  │
│     ├── searchCoarseQuantizer_()  → 粗量化，得到 cluster IDs                │
│     └── searchImpl_()             → 调用精细搜索                             │
│                                                                              │
│  2. runIVFFlatScan() [IVFFlatScan.cu:329-491]                               │
│     ├── 分配临时内存                                                         │
│     ├── 分 tile 处理查询                                                     │
│     └── runIVFFlatScanTile()      → 处理每个 tile                           │
│                                                                              │
│  3. runIVFFlatScanTile() [IVFFlatScan.cu:185-327]                           │
│     ├── runCalcListOffsets()      → 计算偏移                                │
│     ├── ivfFlatScan<<<>>>()       → ⭐ 核心 kernel                          │
│     ├── runPass1SelectLists()     → 第一阶段 k-select                       │
│     └── runPass2SelectLists()     → 第二阶段 k-select                       │
│                                                                              │
│  4. ivfFlatScan kernel [IVFFlatScan.cu:136-183]                             │
│     ├── 每个 block 处理一个 (query, probe) 对                                │
│     ├── 获取倒排列表数据指针: vecs = allListData[listId]                    │
│     └── IVFFlatScan::scan()       → 扫描向量计算距离                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ivfFlatScan Kernel 详细分析

```cpp
// IVFFlatScan.cu:136-183
template <typename Codec, typename Metric>
__global__ void ivfFlatScan(
        Tensor<float, 2, true> queries,
        bool useResidual,
        Tensor<float, 3, true> residualBase,
        Tensor<idx_t, 2, true> listIds,         // 输入: 每个查询要搜索的 cluster IDs
        void** allListData,                      // 输入: 所有倒排列表的数据指针
        idx_t* listLengths,                      // 输入: 每个倒排列表的长度
        Codec codec,
        Metric metric,
        Tensor<idx_t, 2, true> prefixSumOffsets,
        Tensor<float, 1, true> distance) {

    auto queryId = blockIdx.y;
    auto probeId = blockIdx.x;

    idx_t listId = listIds[queryId][probeId];    // 获取 cluster ID
    if (listId == -1) return;

    auto vecs = allListData[listId];             // ⭐ 获取倒排列表数据指针
    auto numVecs = listLengths[listId];          // 倒排列表中的向量数量

    // ... 调用 scan 函数遍历向量 ...
    IVFFlatScan<Codec, Metric>::scan(..., vecs, ..., numVecs, ...);
}
```

### 数据访问模式

```
Grid 配置:
  grid = dim3(nprobe, n_queries)    // nprobe × n_queries 个 block
  block = dim3(128)                  // 4 个 warp

每个 Block 的工作:
  1. 读取 listId = listIds[queryId][probeId]
  2. 获取 vecs = allListData[listId]           ← UVM 数据可能在 CPU
  3. 获取 numVecs = listLengths[listId]
  4. 扫描 vecs[0..numVecs-1] 计算距离          ← 顺序访问

UVM 场景下的问题:
  - vecs 指向的数据可能在 CPU 内存
  - 第一次访问时触发 page fault
  - Page fault 是阻塞的，所有线程等待
```

## Prefetch 优化方案

### 方案 1: Kernel 内部预取 (当前 Block 的数据)

在 `ivfFlatScan` kernel 开始实际计算之前，先发起 prefetch：

```cpp
template <typename Codec, typename Metric>
__global__ void ivfFlatScan(...) {
    auto queryId = blockIdx.y;
    auto probeId = blockIdx.x;

    idx_t listId = listIds[queryId][probeId];
    if (listId == -1) return;

    auto vecs = allListData[listId];
    auto numVecs = listLengths[listId];
    auto dim = queries.getSize(1);

    // ===== 新增: 预取当前 block 要访问的数据 =====
    // 每个线程预取不同的 cache line
    {
        constexpr size_t CACHE_LINE = 128;  // 128 bytes
        size_t dataSize = numVecs * dim * sizeof(float);
        const char* basePtr = static_cast<const char*>(vecs);

        // 每个线程负责预取一部分
        for (size_t offset = threadIdx.x * CACHE_LINE;
             offset < dataSize;
             offset += blockDim.x * CACHE_LINE) {
            asm volatile("prefetch.global.L2 [%0];" : : "l"(basePtr + offset));
        }
    }
    __syncthreads();
    // ===== 预取结束 =====

    // 原有的扫描逻辑
    IVFFlatScan<Codec, Metric>::scan(...);
}
```

**预期效果**：
- 每个 block 在开始计算前预取自己要访问的数据
- 预取和计算有一定重叠（取决于 prefetch 延迟）

**限制**：
- 预取的是当前 block 的数据，没有跨 block 预取
- 第一批预取可能来不及完成就开始计算

### 方案 2: 预取下一个 Probe 的数据

在处理当前 probe 时，预取下一个 probe 的数据：

```cpp
template <typename Codec, typename Metric>
__global__ void ivfFlatScan(
        ...,
        int nprobe,  // 新增参数
        ...) {

    auto queryId = blockIdx.y;
    auto probeId = blockIdx.x;

    // 预取下一个 probe 的数据 (如果存在)
    if (probeId + 1 < nprobe) {
        idx_t nextListId = listIds[queryId][probeId + 1];
        if (nextListId != -1) {
            auto nextVecs = allListData[nextListId];
            auto nextNumVecs = listLengths[nextListId];
            size_t nextDataSize = nextNumVecs * dim * sizeof(float);
            const char* nextBasePtr = static_cast<const char*>(nextVecs);

            // 预取下一个 probe 的数据
            for (size_t offset = threadIdx.x * CACHE_LINE;
                 offset < nextDataSize;
                 offset += blockDim.x * CACHE_LINE) {
                asm volatile("prefetch.global.L2 [%0];" : : "l"(nextBasePtr + offset));
            }
        }
    }

    // 处理当前 probe
    idx_t listId = listIds[queryId][probeId];
    if (listId == -1) return;

    // ... 原有计算逻辑 ...
}
```

**问题**：这种方式可能不太有效，因为所有 probe 的 block 是并行执行的。

### 方案 3: 分离预取 Kernel

添加一个专门的预取 kernel，在主计算 kernel 之前执行：

```cpp
// 新增预取 kernel
__global__ void ivfFlatPrefetch(
        Tensor<idx_t, 2, true> listIds,
        void** allListData,
        idx_t* listLengths,
        int dim) {

    auto queryId = blockIdx.y;
    auto probeId = blockIdx.x;

    idx_t listId = listIds[queryId][probeId];
    if (listId == -1) return;

    auto vecs = allListData[listId];
    auto numVecs = listLengths[listId];

    constexpr size_t CACHE_LINE = 128;
    size_t dataSize = numVecs * dim * sizeof(float);
    const char* basePtr = static_cast<const char*>(vecs);

    // 预取整个倒排列表
    for (size_t offset = threadIdx.x * CACHE_LINE;
         offset < dataSize;
         offset += blockDim.x * CACHE_LINE) {
        asm volatile("prefetch.global.L2 [%0];" : : "l"(basePtr + offset));
    }
}

// 修改 runIVFFlatScanTile
void runIVFFlatScanTile(...) {
    // ... 计算 offset ...

    // ===== 新增: 启动预取 kernel =====
    auto prefetchGrid = dim3(listIds.getSize(1), listIds.getSize(0));
    auto prefetchBlock = dim3(128);
    ivfFlatPrefetch<<<prefetchGrid, prefetchBlock, 0, stream>>>(
            listIds, listData.data(), listLengths.data(), dim);
    // 不需要同步，让预取和后续操作重叠
    // ===== 预取 kernel 结束 =====

    // 原有的扫描 kernel
    ivfFlatScan<<<grid, block, codec.getSmemSize(dim), stream>>>(...);

    // ... k-select ...
}
```

**优点**：
- 预取 kernel 和扫描 kernel 可以部分重叠
- 更清晰的代码结构

**缺点**：
- 额外的 kernel launch 开销
- 预取 kernel 可能阻塞（如果 prefetch 触发 page fault）

### 方案 4: 使用 CUDA Stream 实现流水线预取

```cpp
void runIVFFlatScan(...) {
    cudaStream_t prefetchStream, computeStream;
    cudaStreamCreate(&prefetchStream);
    // computeStream = 原有的 stream

    for (idx_t query = 0; query < queries.getSize(0); query += queryTileSize) {
        // 在 prefetchStream 上预取下一个 tile 的数据
        if (query + queryTileSize < queries.getSize(0)) {
            auto nextListIdsView = listIds.narrowOutermost(
                query + queryTileSize, queryTileSize);
            ivfFlatPrefetch<<<prefetchGrid, prefetchBlock, 0, prefetchStream>>>(
                nextListIdsView, listData.data(), listLengths.data(), dim);
        }

        // 在 computeStream 上执行当前 tile 的计算
        runIVFFlatScanTile(..., computeStream);
    }
}
```

## 修改位置总结

### 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `faiss/gpu/utils/PtxUtils.cuh` | 添加 prefetch 内联函数 |
| `faiss/gpu/impl/IVFFlatScan.cu` | 在 kernel 中添加预取逻辑 |
| `faiss/gpu/impl/IVFFlatScan.cuh` | 如果需要新的函数声明 |

### PtxUtils.cuh 需要添加的函数

```cpp
// 添加在 #endif // USE_AMD_ROCM 之前

// Prefetch to L1 cache
__device__ __forceinline__ void prefetchGlobalL1(const void* ptr) {
    asm volatile("prefetch.global.L1 [%0];" : : "l"(ptr));
}

// Prefetch to L2 cache
__device__ __forceinline__ void prefetchGlobalL2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
}
```

### IVFFlatScan.cu 的修改位置

```cpp
// 在 ivfFlatScan kernel 中，line 162-166 附近

    auto vecs = allListData[listId];
    auto numVecs = listLengths[listId];
    auto dim = queries.getSize(1);

    // ===== 在这里插入预取代码 =====
    // 预取 vecs 指向的数据
    // =====

    auto distanceOut = distance[outBase].data();
```

## 测试方法

### 使用 bench_gpu_1bn.py 测试

```bash
cd /home/yunwei37/workspace/gpu/schedcp/workloads/faiss

# 1. 重新编译 FAISS (修改代码后)
cd faiss
mkdir -p build && cd build
cmake .. -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON \
         -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2
make -j8 swigfaiss

# 2. Baseline 测试 (UVM 模式)
cd /home/yunwei37/workspace/gpu/schedcp/workloads/faiss
uv run python bench_gpu_1bn.py SIFT100M IVF4096,Flat -nprobe 1,4,16 -uvm

# 3. 对比测试
# - 修改前: 记录 Add time 和 Search time
# - 修改后: 同样的测试，对比性能变化
```

### 预期的测试输出

```
# Baseline (当前结果)
Add time: 68.407 s
probe=1:  5.135 s
probe=4:  14.393 s
probe=16: 56.511 s

# 优化后 (预期)
Add time: ~68 s  (add 阶段影响不大)
probe=1:  < 5 s  (可能有 5-15% 提升)
probe=4:  < 14 s
probe=16: < 55 s
```

### 性能分析工具

```bash
# 使用 Nsight Compute 分析 kernel
ncu --target-processes all \
    --set full \
    -o profile_ivfflat \
    python bench_gpu_1bn.py SIFT10M IVF4096,Flat -nprobe 16 -uvm

# 关注的指标:
# - Memory throughput
# - Page fault count (如果可用)
# - L2 cache hit rate
# - Stall reasons (memory dependency)
```

## PTX prefetch.global.L2 的关键特性

### 异步预取 + UVM 页面迁移

`prefetch.global.L2` 指令的重要特性：

1. **异步执行**：prefetch 指令是异步的，发出后 warp 立即继续执行，不会阻塞
2. **可触发 UVM 页面迁移**：当数据在 CPU 内存时，prefetch 会触发页面迁移到 GPU
3. **只有实际访问才会阻塞**：只有当数据被真正读取时，如果迁移未完成，warp 才会暂停

```
时间线对比:

无 prefetch:
[开始扫描] → [访问数据] → [page fault, warp 阻塞等待迁移] → [继续扫描]
                              ↑
                         整个迁移时间都在等待

有 prefetch:
[prefetch 发出] → [继续执行其他代码] → [访问数据] → [数据可能已就绪]
       ↓                                              ↓
   异步触发迁移 ─────────────────────────────────→ 迁移可能已完成
       ↓
   warp 不阻塞
```

### 这意味着什么

这个特性使得 kernel 内 prefetch 变得非常有价值：

| 特性 | 影响 |
|------|------|
| 异步执行 | 可以在计算的同时进行预取 |
| 触发页面迁移 | 不需要 CPU 参与调用 cudaMemPrefetchAsync |
| 非阻塞 | 预取指令本身不会降低性能 |

## 优化策略 (基于异步特性)

### 策略 1: 提前预取下一批数据

由于 prefetch 是异步的，可以在处理当前数据时预取后续数据：

```cpp
// 伪代码
for (vec = vecStart; vec < vecEnd; vec++) {
    // 预取后续向量 (异步，不阻塞)
    if (vec + PREFETCH_DISTANCE < vecEnd) {
        prefetchGlobalL2(vecData + (vec + PREFETCH_DISTANCE) * stride);
    }

    // 处理当前向量 (此时之前预取的数据可能已就绪)
    process(vecData + vec * stride);
}
```

### 策略 2: Block 开始时批量预取

在 block 开始计算前，先发出所有预取请求：

```cpp
__global__ void ivfFlatScan(...) {
    // 获取数据指针
    auto vecs = allListData[listId];
    auto numVecs = listLengths[listId];
    size_t dataSize = numVecs * dim * sizeof(float);

    // 阶段 1: 批量发出预取 (异步，快速完成)
    const char* basePtr = static_cast<const char*>(vecs);
    for (size_t offset = threadIdx.x * 128;
         offset < dataSize;
         offset += blockDim.x * 128) {
        asm volatile("prefetch.global.L2 [%0];" : : "l"(basePtr + offset));
    }
    // 不需要 __syncthreads()，prefetch 本身不阻塞

    // 阶段 2: 开始实际计算 (此时页面迁移已经在进行)
    IVFFlatScan::scan(...);
}
```

### 策略 3: 跨 Block 预取

由于多个 block 并行执行，可以让先执行的 block 预取后续 block 的数据：

```cpp
// 每个 block 预取下几个 probe 的数据
if (probeId + PREFETCH_PROBES < nprobe) {
    for (int p = 1; p <= PREFETCH_PROBES; p++) {
        idx_t nextListId = listIds[queryId][probeId + p];
        if (nextListId != -1) {
            prefetchList(allListData[nextListId], listLengths[nextListId], dim);
        }
    }
}
```

## 风险和注意事项

### 1. 预取数量的权衡

```
prefetch 指令本身有开销:
- 每条 prefetch 指令占用指令带宽
- 过多 prefetch 可能导致 L2 cache 污染
- 需要在预取覆盖率和开销之间平衡

建议:
- 每个线程发出 4-16 条 prefetch 指令
- 预取距离根据数据大小调整
```

### 2. 编译器优化

```cpp
// 确保 prefetch 不被优化掉
asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
//    ^^^^^^^^ volatile 防止优化
```

### 3. 预取粒度

```
Cache line size: 128 bytes (L2)
Page size: 4KB - 2MB (UVM)

如果预取触发 page fault，会迁移整个页面 (4KB+)
如果只是 cache prefetch，只影响 128 bytes
```

### 4. 预取时机

```
太早: 数据可能被 evict
太晚: 预取来不及完成
最佳: 在实际访问前 100-1000 个时钟周期
```

## 下一步建议

1. **先小规模测试**：用 SIFT10M 测试，验证修改是否正确
2. **添加编译开关**：用宏控制是否启用 prefetch，方便对比
3. **测试不同 nprobe**：不同 nprobe 值可能有不同的收益
4. **分析 page fault**：使用 `nvidia-smi` 或 Nsight 监控 page fault

```cpp
// 建议的编译开关
#ifndef FAISS_GPU_ENABLE_PREFETCH
#define FAISS_GPU_ENABLE_PREFETCH 1  // 默认开启
#endif

#if FAISS_GPU_ENABLE_PREFETCH
    // prefetch code
#endif
```

---

*文档更新时间: 2025-12-03*
*基于 FAISS GPU IVFFlatScan.cu 分析*
