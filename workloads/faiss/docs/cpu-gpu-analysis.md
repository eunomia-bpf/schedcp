# FAISS GPU + UVM 架构分析：GPU 发起的预取场景

## 概述

本文档分析 FAISS 在 GPU + UVM (Unified Virtual Memory) 模式下的数据流，探讨 **GPU 发起预取** 的可行性和优化机会。

## FAISS GPU + UVM 的工作模式

### 三种内存模式

```cpp
// faiss/gpu/GpuResources.h
enum class MemorySpace {
    Temporary = 0,  // 临时分配
    Device = 1,     // cudaMalloc - 纯 GPU 显存
    Unified = 2,    // cudaMallocManaged - UVM 统一内存
};
```

### UVM 模式的特点

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FAISS GPU + UVM 架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  数据分配 (cudaMallocManaged):                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  向量数据: 70M vectors × 128 dim × 4 bytes = ~34 GB                 │    │
│  │  分配方式: cudaMallocManaged() - 统一虚拟地址                        │    │
│  │  物理位置: 由 CUDA 驱动动态管理，可在 GPU/CPU 之间迁移               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  搜索时的数据访问:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. GPU kernel 访问向量数据                                          │    │
│  │  2. 如果数据在 CPU 内存 → 触发页面错误 → 自动迁移到 GPU              │    │
│  │  3. 如果数据已在 GPU → 直接访问                                      │    │
│  │  4. GPU 显存不足时 → 驱动自动将冷数据迁移回 CPU                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## FAISS GPU 搜索的完整流程

### IVF 索引搜索流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FAISS GPU IVF 搜索流程 (完全 GPU 配置)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: 粗量化 (Coarse Quantization) - 完全在 GPU                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  GpuIndexFlat::search(queries, nprobe, distances, indices)          │    │
│  │                                                                      │    │
│  │  输入: n 个查询向量 (在 GPU 显存)                                    │    │
│  │  计算: 与 nlist 个聚类中心计算距离                                   │    │
│  │  输出: 每个查询最近的 nprobe 个聚类 ID (cluster_ids)                │    │
│  │                                                                      │    │
│  │  数据位置: 聚类中心 (ivfCentroids_) 在 GPU 显存                      │    │
│  │  ✅ 完全在 GPU 上执行                                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                               │
│                     cluster_ids[n_queries × nprobe]                         │
│                              ↓                                               │
│  Step 2: 精细搜索 (Fine Search) - 在 GPU 上，但需要访问倒排列表              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  对于每个选中的 cluster:                                             │    │
│  │    - 获取该 cluster 的倒排列表                                       │    │
│  │    - 遍历列表中的所有向量，计算距离                                   │    │
│  │    - 维护每个查询的 Top-K 结果                                       │    │
│  │                                                                      │    │
│  │  ⚠️ 关键点: 倒排列表数据可能在 UVM 管理的内存中                      │    │
│  │  - 如果在 GPU 显存: 直接访问                                         │    │
│  │  - 如果在 CPU 内存: 触发页面迁移，GPU 等待                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                               │
│                     top_k_ids[n_queries × k]                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 代码证据

**粗量化器配置** (`GpuIndexIVF.cu:92-109`):
```cpp
// 默认创建 GPU 粗量化器
if (metric_type == faiss::METRIC_L2) {
    quantizer = new GpuIndexFlatL2(resources_, d, config);
} else if (metric_type == faiss::METRIC_INNER_PRODUCT) {
    quantizer = new GpuIndexFlatIP(resources_, d, config);
}
```

**搜索流程** (`IVFBase.cu:508-592`):
```cpp
void IVFBase::searchCoarseQuantizer_(...) {
    auto gpuQuantizer = tryCastGpuIndex(coarseQuantizer);
    if (gpuQuantizer) {
        // ✅ GPU 粗量化器: 完全在 GPU 上
        gpuQuantizer->search(vecs.getSize(0), vecs.data(), nprobe,
                            distances.data(), indices.data());
    } else {
        // ❌ CPU 粗量化器: 需要 GPU↔CPU 传输
        auto cpuVecs = toHost<float, 2>(vecs.data(), stream, ...);
        coarseQuantizer->search(...);
        ...
    }
}
```

## GPU 发起预取的机会分析

### 当前的数据访问模式

```
时间线 (FAISS GPU + UVM 搜索):

[粗量化 GPU kernel] → 知道 cluster_ids → [精细搜索 GPU kernel]
                              ↓                    ↓
                        需要访问这些 cluster    访问时才发现
                        的倒排列表              数据可能不在 GPU
                                                     ↓
                                               页面错误 → 迁移 → 继续

问题: 页面迁移是被动触发的，GPU 必须等待
```

### GPU 发起预取的场景

**核心观察**: 粗量化完成后，GPU 就知道需要访问哪些倒排列表，但实际访问前有一个时间窗口。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  GPU 发起预取的优化机会                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  优化前 (当前实现):                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  [粗量化] → [精细搜索开始] → [页面错误] → [等待迁移] → [继续搜索]     │    │
│  │                                   ↑                                   │    │
│  │                              GPU 被阻塞                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  优化后 (GPU 发起预取):                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  [粗量化] → [发起预取 hint] → [精细搜索开始] → [数据已就绪] → [继续]  │    │
│  │                  ↓                                                    │    │
│  │         异步预取 cluster 的倒排列表                                    │    │
│  │                  ↓                                                    │    │
│  │         与精细搜索的其他部分重叠                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 为什么 FAISS GPU + UVM 是好的预取场景？

| 条件 | FAISS GPU + UVM | 说明 |
|------|-----------------|------|
| **GPU 完全知道需要什么** | ✅ | 粗量化后知道要访问的 cluster IDs |
| **有足够的时间窗口** | ✅ | 粗量化和精细搜索之间可以插入预取 |
| **数据在慢速存储** | ✅ | UVM 数据可能在 CPU 内存，需要 PCIe 传输 |
| **批量处理** | ✅ | n_queries × nprobe 个列表可以并行预取 |
| **搜索完全在 GPU** | ✅ | 使用 GpuIndexFlat 作为粗量化器时 |

### 与 CPU prefetch_lists 的对比

FAISS CPU 版本有 `invlists->prefetch_lists()` 机制：

```cpp
// IndexIVF.cpp:337
quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());
invlists->prefetch_lists(idx.get(), n * nprobe);  // ← CPU 预取
search_preassigned(...);
```

**GPU 版本目前缺少这个预取步骤！**

## 实现 GPU 发起预取的方案

### 方案 1: 使用 cudaMemPrefetchAsync

```cpp
// 伪代码 - 在精细搜索前插入预取
void IVFBase::search(...) {
    // Step 1: 粗量化 (GPU)
    searchCoarseQuantizer_(coarseQuantizer, nprobe, queries,
                          coarseDistances, coarseIndices, ...);

    // Step 2: 预取倒排列表到 GPU (新增)
    prefetchInvertedLists_(coarseIndices, stream);

    // Step 3: 精细搜索 (GPU)
    searchImpl_(...);
}

void IVFBase::prefetchInvertedLists_(
    Tensor<idx_t, 2, true>& clusterIds,
    cudaStream_t stream) {

    // 获取需要预取的 cluster IDs (可能需要 D2H 传输)
    auto hostIds = toHost<idx_t, 2>(clusterIds.data(), stream, ...);

    // 对每个 cluster 的倒排列表数据发起预取
    for (idx_t clusterId : hostIds) {
        void* listPtr = getListDataPointer(clusterId);
        size_t listSize = getListSize(clusterId);

        // 预取到当前 GPU
        cudaMemPrefetchAsync(listPtr, listSize, device_, stream);
    }
}
```

### 方案 2: GPU Kernel 内部预取 (更高效)

```cuda
// GPU kernel 在粗量化后直接发起预取
__global__ void coarse_quantize_with_prefetch_hint(
    float* queries,
    float* centroids,
    idx_t* output_cluster_ids,
    void** invlist_pointers,  // 每个 cluster 的数据指针
    size_t* invlist_sizes
) {
    int query_id = blockIdx.x;

    // 执行粗量化，找到最近的 nprobe 个 cluster
    compute_nearest_clusters(queries + query_id * dim, centroids,
                            output_cluster_ids + query_id * nprobe);

    // 发起预取 hint (使用 __prefetch_global_l2)
    if (threadIdx.x < nprobe) {
        idx_t cluster_id = output_cluster_ids[query_id * nprobe + threadIdx.x];
        void* list_ptr = invlist_pointers[cluster_id];
        size_t list_size = invlist_sizes[cluster_id];

        // 触发预取 (CUDA 没有直接的 prefetch 指令，但可以用其他方式)
        // 方式 1: 使用 __ldg 加载少量数据到 L2 cache
        // 方式 2: 启动单独的 prefetch kernel
    }
}
```

### 方案 3: 流水线化多个查询批次

```cpp
// 将查询分成多个批次，流水线执行
void pipelined_search(
    Tensor<float, 2>& queries,
    int batch_size,
    ...
) {
    cudaStream_t coarse_stream, fine_stream, prefetch_stream;

    for (int i = 0; i < num_batches; i++) {
        // 当前批次: 精细搜索
        if (i > 0) {
            searchImpl_(batch[i-1], fine_stream);
        }

        // 当前批次: 粗量化
        searchCoarseQuantizer_(batch[i], coarse_stream);

        // 当前批次: 预取 (与下一批次的精细搜索重叠)
        prefetchInvertedLists_(batch[i], prefetch_stream);

        // 同步
        cudaStreamSynchronize(coarse_stream);
    }
}
```

## 性能分析

### 基准测试结果 (SIFT100M, IVF4096,Flat)

从 README.md 中的测试数据：

**无预取优化 (baseline)**:
```
Add time: 68.407s
probe=1:  5.135s  (1-R@1: 0.4486)
probe=4:  14.393s (1-R@1: 0.7655)
probe=16: 56.511s (1-R@1: 0.9476)
```

**有预取策略 (prefetch_adaptive_tree_iter)**:
```
Add time: 49.309s  (↓28%)
probe=1:  4.532s   (↓12%)
probe=4:  13.106s  (↓9%)
probe=16: 51.440s  (↓9%)
```

### 预取收益分析

```
时间分解 (probe=16, 100M 向量):

无预取:
[粗量化 ~1s] → [精细搜索 + 页面迁移 ~55s]
                        ↑
                 GPU 频繁等待页面迁移

有预取:
[粗量化 ~1s] → [预取 ~Xs] → [精细搜索 ~50s]
                   ↘         ↗
                    部分重叠

预期收益:
- 页面迁移与计算重叠
- 减少 GPU 空闲时间
- 收益取决于预取准确率和时间窗口
```

## 进一步优化方向

### 1. 智能预取策略

```cpp
// 根据 cluster 访问频率优先预取热门 cluster
void smart_prefetch(idx_t* cluster_ids, int n) {
    // 统计访问频率
    std::unordered_map<idx_t, int> freq;
    for (int i = 0; i < n; i++) {
        freq[cluster_ids[i]]++;
    }

    // 按频率排序，优先预取高频 cluster
    std::vector<idx_t> sorted_clusters = sort_by_frequency(freq);

    for (idx_t c : sorted_clusters) {
        cudaMemPrefetchAsync(getListPtr(c), getListSize(c), device_, stream);
    }
}
```

### 2. 预取 hint 传递给 CUDA 驱动

```cpp
// 使用 CUDA Memory Advise API
void advise_prefetch(void* ptr, size_t size, int device) {
    // 告诉驱动这块内存即将被访问
    cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device);

    // 主动触发预取
    cudaMemPrefetchAsync(ptr, size, device, stream);
}
```

### 3. 与 CXL 结合

如果底层使用 CXL 扩展内存：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GPU + UVM + CXL 架构                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GPU ←─ PCIe ─→ CPU ←─ CXL ─→ CXL Memory                                    │
│                                                                              │
│  数据位置层级:                                                                │
│  1. GPU 显存 (最快)                                                          │
│  2. CPU 内存 (中等)                                                          │
│  3. CXL 内存 (较慢，但容量大)                                                 │
│                                                                              │
│  预取策略:                                                                    │
│  - GPU 计算后发出预取 hint                                                   │
│  - 预取路径: CXL → CPU → GPU 或 CXL → GPU (如果支持 P2P)                    │
│  - 多级预取: 热数据到 GPU，温数据到 CPU                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 总结

### FAISS GPU + UVM 作为预取场景的优势

1. **完全 GPU 配置可行**: 使用 GpuIndexFlat 作为粗量化器，整个搜索在 GPU 上
2. **预取时机明确**: 粗量化后知道要访问的 cluster，可以提前预取
3. **有足够的时间窗口**: 粗量化和精细搜索之间可以插入预取操作
4. **数据量大，收益显著**: 100M+ 向量的倒排列表，预取可以显著减少等待时间

### 关键实现点

1. **在精细搜索前插入预取**: 利用粗量化的结果提前加载数据
2. **使用 cudaMemPrefetchAsync**: 异步预取，与计算重叠
3. **考虑流水线**: 多批次查询时，预取和计算可以进一步重叠
4. **智能预取策略**: 根据访问频率优先预取热门数据

### 与 llama.cpp MoE 的对比

| 方面 | FAISS GPU + UVM | llama.cpp MoE |
|------|-----------------|---------------|
| GPU 计算完整性 | ✅ 完全在 GPU | ⚠️ CPU 参与路由索引 |
| 预取时间窗口 | ✅ 粗量化后有窗口 | ❌ Router 后立即需要权重 |
| 数据量 | 大 (GB 级倒排列表) | 大 (Expert 权重) |
| 预取收益 | ✅ 显著 (已验证 ~10% 提升) | ⚠️ 有限 (时间窗口小) |

---

*文档更新时间: 2025-12-03*
*基于 FAISS GPU 实现和测试结果分析*
