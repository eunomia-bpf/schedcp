# GPU Kernel 内部发起预取的可行性分析

## 问题背景

在 FAISS GPU + UVM 场景中，我们希望实现：
1. GPU 粗量化完成后，知道需要访问哪些倒排列表
2. **在 GPU kernel 内部** 发起预取请求
3. 异步将数据从 CPU 内存迁移到 GPU 显存
4. 精细搜索时数据已就绪，减少 page fault

**核心问题：GPU kernel 能否直接触发 UVM 页面迁移？**

## 结论：可以！

**`prefetch.global.L2` 指令可以触发 UVM 页面迁移，并且是异步的、非阻塞的。**

## CUDA 预取机制概述

### 两种预取层次

| 类型 | API/指令 | 作用范围 | 能触发 UVM 页面迁移？ | 阻塞？ |
|------|---------|---------|---------------------|--------|
| **Host API** | `cudaMemPrefetchAsync` | 页面级 (4KB-2MB) | ✅ 可以 | 异步 |
| **Kernel 内 PTX** | `prefetch.global.L2` | 触发页面级迁移 | ✅ 可以 | **异步，不阻塞 warp** |

### cudaMemPrefetchAsync (Host API)

```cpp
// 在 host 端调用，触发页面迁移
cudaError_t cudaMemPrefetchAsync(
    const void* devPtr,   // UVM 指针
    size_t count,         // 字节数
    int dstDevice,        // 目标设备 (GPU ID 或 cudaCpuDeviceId)
    cudaStream_t stream   // CUDA stream
);
```

特点：
- 页面级迁移，粒度较大
- 可以提前将数据从 CPU 迁移到 GPU
- 必须从 **host 端** 调用，不能在 kernel 内调用

### PTX Prefetch 指令 (Kernel 内) - 关键特性

```cuda
// 在 kernel 内使用 inline PTX
__device__ void prefetch_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
}

__device__ void prefetch_l1(const void* ptr) {
    asm volatile("prefetch.global.L1 [%0];" : : "l"(ptr));
}
```

**重要特性**：
1. **异步执行**：prefetch 指令发出后，warp 立即继续执行，不会阻塞
2. **可触发 UVM 页面迁移**：当数据在 CPU 内存时，会触发页面迁移到 GPU
3. **只有实际访问才阻塞**：只有当数据被真正读取且迁移未完成时，warp 才会暂停

## PTX Prefetch 在 UVM 下的工作原理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              prefetch.global.L2 在 UVM 场景下的行为                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  场景 1: 数据已在 GPU 显存                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  prefetch.global.L2 [ptr]                                           │    │
│  │  → 将数据从 GPU Global Memory 预取到 L2 Cache                        │    │
│  │  → 异步执行，warp 不阻塞                                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  场景 2: 数据在 CPU 内存 (UVM)                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  prefetch.global.L2 [ptr]                                           │    │
│  │  → 触发页面迁移请求 (CPU → GPU)                                      │    │
│  │  → 异步执行，warp 不阻塞                                             │    │
│  │  → 迁移在后台进行                                                    │    │
│  │                                                                      │    │
│  │  后续访问 ptr:                                                       │    │
│  │  → 如果迁移完成: 直接访问，无阻塞                                     │    │
│  │  → 如果迁移未完成: warp 暂停等待                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 时间线对比

```
无 prefetch:
[开始扫描] → [访问数据] → [page fault] → [warp 阻塞等待迁移完成] → [继续]
                              ↑
                         整个迁移延迟都在阻塞

有 prefetch:
[prefetch] → [继续执行代码] → [访问数据] → [数据可能已就绪，无阻塞或短暂等待]
     ↓              ↓
  异步触发     迁移在后台进行
  页面迁移
     ↓
 warp 不阻塞
```

## GPU Kernel 内部触发预取的可能方案

### 方案 1: 显式读取触发 Page Fault

```cuda
__global__ void prefetch_by_touch(
    char* uvm_ptr,
    int64_t* offsets,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // 触摸数据，强制触发 page fault
        // 驱动会将对应页面迁移到 GPU
        volatile char c = uvm_ptr[offsets[tid]];
        (void)c;  // 防止编译器优化掉
    }
}
```

**问题**：
- Page fault 会 **阻塞** 触发它的线程
- 不是真正的异步预取
- 多个线程同时 page fault 会导致严重性能下降

**适用场景**：
- 预取线程和计算线程分离
- 可以容忍一定的阻塞

### 方案 2: 分离 Prefetch Kernel

```cpp
// 思路: 用一个轻量级 kernel 故意触发 page fault
//       在另一个 stream 上运行，与计算 kernel 重叠

void search_with_prefetch(
    float* queries,
    float* vectors,      // UVM 内存
    int64_t* cluster_ids,
    cudaStream_t compute_stream,
    cudaStream_t prefetch_stream
) {
    // Step 1: 粗量化 (得到 cluster_ids)
    coarse_quantize_kernel<<<grid, block, 0, compute_stream>>>(
        queries, centroids, cluster_ids);

    // Step 2: 启动预取 kernel (在不同 stream)
    // 这个 kernel 会触发 page fault，但在不同 stream 上
    prefetch_kernel<<<prefetch_grid, prefetch_block, 0, prefetch_stream>>>(
        vectors, cluster_ids, invlist_offsets);

    // Step 3: 精细搜索 (希望部分数据已被预取)
    fine_search_kernel<<<grid, block, 0, compute_stream>>>(
        queries, vectors, cluster_ids, results);
}

// 预取 kernel: 每个线程触摸一个 cache line
__global__ void prefetch_kernel(
    float* vectors,
    int64_t* cluster_ids,
    int64_t* invlist_offsets
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 获取需要预取的地址
    int64_t cluster = cluster_ids[tid / THREADS_PER_CLUSTER];
    int64_t offset = invlist_offsets[cluster];
    int64_t local_offset = (tid % THREADS_PER_CLUSTER) * CACHE_LINE_SIZE;

    // 触摸数据，触发页面迁移
    volatile float f = vectors[offset + local_offset];
    (void)f;
}
```

**优点**：
- 预取和计算可以部分重叠
- 利用多 stream 并行

**缺点**：
- 预取 kernel 仍然会阻塞
- 需要额外的 kernel launch 开销
- 调度复杂

### 方案 3: GPU 通知 CPU 发起预取 (推荐)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  GPU 主动、CPU 辅助的预取架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  GPU Kernel (粗量化)                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  1. 计算 cluster_ids                                             │  │  │
│  │  │  2. 写入预取请求队列 (pinned memory)                              │  │  │
│  │  │  3. 设置 ready_flag = 1 (atomic)                                 │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Pinned Memory (CPU 可见, GPU 可写)                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  prefetch_queue: [cluster_0, cluster_1, ..., cluster_n]         │  │  │
│  │  │  ready_flag: 1                                                   │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  CPU Prefetch Thread (独立线程，持续运行)                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  while (running) {                                               │  │  │
│  │  │      if (atomicLoad(ready_flag) == 1) {                         │  │  │
│  │  │          atomicStore(ready_flag, 0);                            │  │  │
│  │  │          for (cluster : prefetch_queue) {                       │  │  │
│  │  │              void* ptr = invlist_ptrs[cluster];                 │  │  │
│  │  │              size_t size = invlist_sizes[cluster];              │  │  │
│  │  │              cudaMemPrefetchAsync(ptr, size, gpu_id, stream);   │  │  │
│  │  │          }                                                       │  │  │
│  │  │      }                                                           │  │  │
│  │  │      // 短暂 sleep 或 spin                                       │  │  │
│  │  │  }                                                               │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  GPU Kernel (精细搜索)                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  访问倒排列表数据                                                 │  │  │
│  │  │  - 如果预取成功: 数据已在 GPU，无 page fault                      │  │  │
│  │  │  - 如果预取未完成: 触发 page fault，等待迁移                      │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**实现代码**:

```cpp
// === 数据结构 ===
struct PrefetchRequest {
    int64_t* cluster_ids;      // GPU 写入的 cluster IDs
    int num_clusters;          // cluster 数量
    volatile int ready_flag;   // GPU 设置，CPU 检查
};

// 在 pinned memory 中分配
PrefetchRequest* request;
cudaMallocHost(&request, sizeof(PrefetchRequest));
cudaMalloc(&request->cluster_ids, max_clusters * sizeof(int64_t));

// === GPU Kernel: 粗量化 + 写预取请求 ===
__global__ void coarse_quantize_with_prefetch_hint(
    float* queries,
    float* centroids,
    int64_t* output_cluster_ids,
    PrefetchRequest* prefetch_request,
    int n_queries,
    int nprobe
) {
    int query_id = blockIdx.x;
    if (query_id >= n_queries) return;

    // 计算最近的 nprobe 个 cluster
    compute_nearest_clusters(
        queries + query_id * dim,
        centroids,
        output_cluster_ids + query_id * nprobe,
        nprobe
    );

    // 最后一个 block 负责设置预取请求
    __syncthreads();
    if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {
        prefetch_request->num_clusters = n_queries * nprobe;
        __threadfence_system();  // 确保写入对 CPU 可见
        prefetch_request->ready_flag = 1;
    }
}

// === CPU Prefetch Thread ===
void prefetch_thread_func(
    PrefetchRequest* request,
    void** invlist_ptrs,
    size_t* invlist_sizes,
    int gpu_id,
    cudaStream_t prefetch_stream,
    std::atomic<bool>& running
) {
    while (running) {
        // 检查 GPU 是否发出了预取请求
        if (request->ready_flag == 1) {
            request->ready_flag = 0;

            // 将 cluster_ids 复制到 host (或使用 pinned memory 直接访问)
            std::vector<int64_t> clusters(request->num_clusters);
            cudaMemcpy(clusters.data(), request->cluster_ids,
                      request->num_clusters * sizeof(int64_t),
                      cudaMemcpyDeviceToHost);

            // 去重 (可选)
            std::sort(clusters.begin(), clusters.end());
            clusters.erase(std::unique(clusters.begin(), clusters.end()),
                          clusters.end());

            // 发起预取
            for (int64_t cluster : clusters) {
                cudaMemPrefetchAsync(
                    invlist_ptrs[cluster],
                    invlist_sizes[cluster],
                    gpu_id,
                    prefetch_stream
                );
            }
        }

        // 短暂等待，避免 busy loop
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

// === 主搜索流程 ===
void search(float* queries, int n_queries, ...) {
    // 启动粗量化 kernel (会设置预取请求)
    coarse_quantize_with_prefetch_hint<<<n_queries, 256, 0, compute_stream>>>(
        queries, centroids, cluster_ids, prefetch_request, n_queries, nprobe);

    // CPU prefetch thread 会检测到 ready_flag 并发起 cudaMemPrefetchAsync
    // 这与 GPU kernel 执行是并行的

    // 启动精细搜索 kernel
    fine_search_kernel<<<grid, block, 0, compute_stream>>>(
        queries, vectors, cluster_ids, results);
}
```

**优点**：
- GPU 主动发起预取请求
- `cudaMemPrefetchAsync` 是真正的异步，不阻塞
- 预取和计算可以完全重叠
- 灵活，可以实现复杂的预取策略

**缺点**：
- 需要 CPU 线程配合
- 有一定的通信延迟 (GPU → CPU)
- 实现较复杂

### 方案 4: CUDA Graph + cudaMemPrefetchAsync

```cpp
// 使用 CUDA Graph 将预取操作和计算融合

cudaGraph_t graph;
cudaGraphExec_t graphExec;

// 创建 graph
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 1. 粗量化 kernel
coarse_quantize_kernel<<<grid, block, 0, stream>>>(...);

// 2. 预取操作 (需要提前知道要预取什么，或使用 graph update)
for (int i = 0; i < expected_clusters; i++) {
    cudaMemPrefetchAsync(invlist_ptrs[i], invlist_sizes[i], gpu_id, stream);
}

// 3. 精细搜索 kernel
fine_search_kernel<<<grid, block, 0, stream>>>(...);

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// 执行
cudaGraphLaunch(graphExec, stream);
```

**限制**：
- 预取的目标在 graph 创建时必须已知
- 不适合动态确定预取目标的场景

## 性能对比

### 预期时间线

```
方案 5 (PTX Prefetch - 推荐):
[处理 list 0 + prefetch list 1]──[处理 list 1 + prefetch list 2]──...
         ↓                                ↓
    异步预取，不阻塞                  数据可能已就绪

方案 3 (GPU→CPU 通知):
GPU:    [粗量化+写请求]────────────────[精细搜索]
CPU:              [检测]──[cudaMemPrefetchAsync]
                              ↑
                        完全异步，无阻塞

方案 2 (分离 kernel):
Stream 1: [粗量化]────────────────────[精细搜索]
Stream 2:          [预取kernel(阻塞)]
                         ↑
                   部分重叠，但仍有阻塞

方案 1 (显式读取触发):
[粗量化]──[预取kernel阻塞等待]──[精细搜索]
          ↑
        page fault 阻塞

无预取 (baseline):
[粗量化]──[精细搜索 + 随机 page fault]
                    ↑
              性能不可预测
```

### 预期收益

| 方案 | 预取延迟 | 与计算重叠 | 实现复杂度 | 适用场景 |
|------|---------|-----------|-----------|---------|
| **PTX Prefetch** | 最低 | 完全 | 低 | **推荐** |
| GPU→CPU 通知 | 低 | 完全 | 高 | 需要精细控制 |
| 分离 kernel | 中 | 部分 | 中 | 中等复杂度 |
| 显式读取 | 高 | 无 | 低 | 简单场景 |
| 无预取 | N/A | N/A | N/A | baseline |

## 推荐方案：直接使用 PTX Prefetch

既然 `prefetch.global.L2` 可以异步触发 UVM 页面迁移，**最简单直接的方案就是在 kernel 内使用 PTX prefetch 指令**。

### 方案 5: 直接 PTX Prefetch (推荐)

```cuda
__device__ __forceinline__ void prefetchGlobalL2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
}

// 在 IVFFlatScan kernel 中
__global__ void ivfFlatScan(...) {
    // 获取当前要处理的 list
    auto listId = cycleListIds[cycle];
    auto vecs = allListData[listId];
    auto numVecs = listLengths[listId];

    // === 预取下一个 list 的数据 ===
    if (cycle + 1 < numCycles) {
        auto nextListId = cycleListIds[cycle + 1];
        auto nextVecs = allListData[nextListId];
        auto nextNumVecs = listLengths[nextListId];

        // 每个线程预取不同的 cache line
        int tid = threadIdx.x;
        int stride = blockDim.x * 128;  // 128 bytes per cache line
        for (int offset = tid * 128; offset < nextNumVecs * dim * sizeof(float); offset += stride) {
            prefetchGlobalL2((char*)nextVecs + offset);
        }
    }

    // 处理当前 list
    IVFFlatScan::scan<Metric>(..., vecs, numVecs, ...);
}
```

**优点**：
- ✅ 最简单直接，无需 CPU 参与
- ✅ 异步执行，不阻塞 warp
- ✅ 可触发 UVM 页面迁移
- ✅ 与计算完美重叠

**适用场景**：
- FAISS IVF 搜索中，处理当前倒排列表时预取下一个列表
- 任何可以预知下一步访问模式的场景

## 方案对比总结

| 方案 | 复杂度 | 预取效果 | 是否阻塞 | 推荐程度 |
|------|--------|---------|---------|---------|
| **方案 5: PTX Prefetch** | 低 | 高 | 否 | ⭐⭐⭐⭐⭐ |
| 方案 3: GPU→CPU 通知 | 高 | 高 | 否 | ⭐⭐⭐ |
| 方案 2: 分离 kernel | 中 | 中 | 部分 | ⭐⭐ |
| 方案 1: 显式读取 | 低 | 低 | 是 | ⭐ |

## 硬件支持与未来发展

### 当前支持 (CUDA 12.x)

1. **PTX `prefetch.global.L2` 支持 UVM 页面迁移**
   - 异步执行，不阻塞发起预取的 warp
   - 触发后台页面迁移请求
   - 只有实际访问数据时，若迁移未完成才阻塞

2. **Page fault 仍然是阻塞的**
   - 未使用 prefetch 时，首次访问 UVM 数据会触发 page fault
   - Prefetch 的价值在于提前触发迁移，避免 page fault

### 未来发展 (Hopper/Blackwell 架构)

1. **cp.async.bulk.prefetch.L2**
   - Hopper 引入的批量预取指令
   - 更高效的批量预取

2. **Grace Hopper 的统一内存架构**
   - NVLink-C2C 连接 CPU 和 GPU
   - 更低的页面迁移延迟

## 结论

### 回答核心问题

**Q: GPU kernel 内部能否直接发起异步预取（触发 UVM 页面迁移）？**

**A: 可以！** `prefetch.global.L2` PTX 指令可以：
- ✅ 异步执行，不阻塞 warp
- ✅ 触发 UVM 页面迁移
- ✅ 在数据被实际访问前完成迁移（如果时间足够）

### 推荐方案

对于 FAISS GPU + UVM 场景，**推荐直接使用 PTX Prefetch（方案 5）**：

1. 在 `IVFFlatScan.cu` 的 `ivfFlatScan` kernel 中添加 prefetch 指令
2. 处理当前倒排列表时，预取下一个列表的数据
3. 利用 prefetch 的异步特性与计算重叠

这种方案：
- 实现最简单，只需添加几行 PTX 代码
- 无需 CPU 参与，全部在 GPU kernel 内完成
- 异步执行，不影响当前计算
- 可以显著减少 page fault 导致的阻塞

### 参考资料

- [NVIDIA Developer Blog - Boosting Application Performance with GPU Memory Prefetching](https://developer.nvidia.com/blog/boosting-application-performance-with-gpu-memory-prefetching/)
- [NVIDIA Developer Blog - Maximizing Unified Memory Performance in CUDA](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/)
- [Stack Overflow - Prefetch in CUDA](https://stackoverflow.com/questions/4755275/can-i-prefetch-specific-data-to-a-specific-cache-level-in-a-cuda-kernel)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

---

*文档更新时间: 2025-12-03*
*基于 CUDA 12.x 和当前 GPU 架构分析*
