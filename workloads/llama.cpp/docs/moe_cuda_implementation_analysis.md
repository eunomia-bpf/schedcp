# llama.cpp MoE CUDA 实现详细分析

本文档详细分析 llama.cpp 中 Mixture of Experts (MoE) 的 CUDA 实现机制。

## 目录

1. [概述](#概述)
2. [核心文件结构](#核心文件结构)
3. [Expert 路由机制](#expert-路由机制)
4. [MoE 矩阵乘法主流程](#moe-矩阵乘法主流程)
5. [量化 MoE 支持](#量化-moe-支持)
6. [内存管理与数据布局](#内存管理与数据布局)
7. [性能优化技巧](#性能优化技巧)
8. [MoE 卸载分析](#moe-卸载分析)
9. [支持的模型架构](#支持的模型架构)
10. [关键代码位置速查](#关键代码位置速查)

---

## 概述

llama.cpp 的 MoE CUDA 实现采用**两阶段架构**：

1. **GPU 端路由计算** - 高效的 warp 级 top-k 选择和 softmax
2. **混合 CPU-GPU 执行** - 逐 expert 执行 GEMM，支持量化权重

### 核心设计理念

- **Warp 级并行** - 充分利用 CUDA warp 的 SIMT 特性
- **编译时特化** - 通过模板参数针对不同 expert 数量优化
- **紧凑索引编码** - 减少内存带宽需求
- **Kernel 融合** - 减少 kernel launch 开销

---

## 核心文件结构

### CUDA 实现文件

| 文件路径 | 功能描述 |
|---------|---------|
| `ggml/src/ggml-cuda/topk-moe.cu` | Top-K 路由 kernel，选择激活哪些专家 |
| `ggml/src/ggml-cuda/topk-moe.cuh` | Top-K MoE kernel 头文件声明 |
| `ggml/src/ggml-cuda/mmid.cu` | MoE ID 转换 kernel，token 到 expert 的映射 |
| `ggml/src/ggml-cuda/mmid.cuh` | MoE ID helper 头文件声明 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | MoE 主流程，包含 `ggml_cuda_mul_mat_id()` |
| `ggml/src/ggml-cuda/mmq.cuh` | 量化矩阵乘法的 MoE 支持 |

### 模型实现文件

| 模型架构 | 文件路径 | 说明 |
|---------|---------|------|
| Qwen2-MoE | `src/models/qwen2moe.cpp` | 标准 top-k MoE |
| Qwen3-MoE | `src/models/qwen3moe.cpp` | 增强版本 |
| GLM-4-MoE | `src/models/glm4-moe.cpp` | 混合密集和 MoE 层 |
| DeepSeek-MoE | `src/models/deepseek2.cpp` | 专家共享机制 |
| Arctic | `src/models/arctic.cpp` | 密集 + MoE 混合 |
| Mixtral/DBRX | `src/models/dbrx.cpp` | 8 专家配置 |

---

## Expert 路由机制

### Top-K 路由 Kernel

**文件位置**: `ggml/src/ggml-cuda/topk-moe.cu:63-167`

```cpp
template <int n_experts, bool with_norm, bool delayed_softmax = false>
__launch_bounds__(4 * WARP_SIZE, 1)
__global__ void topk_moe_cuda(
    const float * logits,      // 路由 logits [n_tokens, n_experts]
    float *       weights,     // 输出权重 [n_tokens, n_expert_used]
    int32_t *     ids,         // 选中的 expert IDs [n_tokens, n_experts]
    const int     n_rows,      // token 数量
    const int     n_expert_used, // top-k 值
    const float   clamp_val)   // 权重归一化的夹持值
```

### 算法流程

#### 步骤 1: Warp 级 Softmax 预处理 (`topk-moe.cu:10-51`)

每个 warp (32 个线程) 并行处理一个 token 的所有 expert logits：

```cpp
__device__ void softmax_warp_inplace(float * vals, const int ncols, const float max_val) {
    // 1. 减去最大值进行数值稳定化
    for (int col = threadIdx.x; col < ncols; col += WARP_SIZE) {
        vals[col] -= max_val;
    }

    // 2. 计算 exp 并求和
    float sum = 0.0f;
    for (int col = threadIdx.x; col < ncols; col += WARP_SIZE) {
        vals[col] = expf(vals[col]);
        sum += vals[col];
    }

    // 3. Warp 规约求总和
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask, WARP_SIZE);
    }

    // 4. 归一化
    for (int col = threadIdx.x; col < ncols; col += WARP_SIZE) {
        vals[col] /= sum;
    }
}
```

#### 步骤 2: 贪心 Top-K 选择 (`topk-moe.cu:105-140`)

```cpp
// 迭代 n_expert_used 次，每次选择最大值
for (int k = 0; k < n_expert_used; ++k) {
    float max_val = -INFINITY;
    int   max_idx = -1;

    // 每个线程找自己负责的 experts 中的最大值
    for (int col = threadIdx.x; col < n_experts; col += WARP_SIZE) {
        if (vals[col] > max_val) {
            max_val = vals[col];
            max_idx = col;
        }
    }

    // Warp shuffle 规约找全局最大
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        float other_val = __shfl_xor_sync(0xFFFFFFFF, max_val, mask);
        int   other_idx = __shfl_xor_sync(0xFFFFFFFF, max_idx, mask);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    // 记录选中的 expert 并标记为 -inf
    if (threadIdx.x == 0) {
        weights[row * n_expert_used + k] = max_val;
        ids[row * n_experts + k] = max_idx;
    }
    vals[max_idx] = -INFINITY;  // 排除下一轮
}
```

#### 步骤 3: 权重归一化 (`topk-moe.cu:142-150`)

```cpp
if constexpr (with_norm) {
    // 对选中的 k 个 expert 权重求和并归一化
    float sum = 0.0f;
    for (int k = 0; k < n_expert_used; ++k) {
        sum += weights[row * n_expert_used + k];
    }
    for (int k = 0; k < n_expert_used; ++k) {
        weights[row * n_expert_used + k] /= sum;
    }
}
```

### 编译时模板特化 (`topk-moe.cu:184-228`)

为不同 expert 数量生成优化的 kernel：

```cpp
switch (n_expert) {
    case 1:   topk_moe_cuda<1,   with_norm><<<grid, block, 0, stream>>>(...); break;
    case 2:   topk_moe_cuda<2,   with_norm><<<grid, block, 0, stream>>>(...); break;
    case 4:   topk_moe_cuda<4,   with_norm><<<grid, block, 0, stream>>>(...); break;
    case 8:   topk_moe_cuda<8,   with_norm><<<grid, block, 0, stream>>>(...); break;
    case 16:  topk_moe_cuda<16,  with_norm><<<grid, block, 0, stream>>>(...); break;
    case 32:  topk_moe_cuda<32,  with_norm><<<grid, block, 0, stream>>>(...); break;
    case 64:  topk_moe_cuda<64,  with_norm><<<grid, block, 0, stream>>>(...); break;
    case 128: topk_moe_cuda<128, with_norm><<<grid, block, 0, stream>>>(...); break;
    case 256: topk_moe_cuda<256, with_norm><<<grid, block, 0, stream>>>(...); break;
    case 512: topk_moe_cuda<512, with_norm><<<grid, block, 0, stream>>>(...); break;
}
```

### MoE ID 转换 Kernel

**文件位置**: `ggml/src/ggml-cuda/mmid.cu:28-116`

将 expert routing IDs 转换为适合高效 GEMM 的索引结构。

#### 紧凑存储格式

```cpp
// 22 位 token 索引 + 10 位 expert 索引
struct mm_ids_helper_store {
    uint32_t data;  // [token_idx (22 bits) | expert_idx (10 bits)]
};
```

#### 输出数据结构

- `ids_src1[]`: 激活张量 src1 的读取索引 (用于 get_rows)
- `ids_dst[]`: 输出张量 dst 的写入索引 (用于 scatter)
- `expert_bounds[]`: 每个 expert 的 token 范围 `[low, high)`

#### 两种实现方式

**通用实现** (n_expert_used=0，动态专家数量):

```cpp
// mmid.cu:43-60
for (int ie = 0; ie < n_expert_used; ++ie) {
    const int32_t ids_data = ids[ie * n_tokens + row_idx];
    // 通过循环查找对应的 expert
    for (int expert = 0; expert < n_expert; ++expert) {
        if (ids_data == expert) {
            // 记录映射
        }
    }
}
```

**特化实现** (n_expert_used=2,4,6,8,16,32):

```cpp
// mmid.cu:65-93
// 使用并行扫描，编译时展开循环
#pragma unroll
for (int ie = 0; ie < N_EXPERT_USED; ++ie) {
    // 直接索引访问，避免循环
}
```

---

## MoE 矩阵乘法主流程

**文件位置**: `ggml/src/ggml-cuda/ggml-cuda.cu:2288-2434`

### 完整执行流程

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 复制 IDs 到主机 (同步点 #1)                          │
│   cudaMemcpy(ids_host, ids_device, ...)                     │
│   cudaStreamSynchronize(stream)  // 2349 行                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: CPU 端路由计算 (2351-2364 行)                        │
│   for 每个 expert:                                          │
│       for 每个 token:                                       │
│           for 每个 expert_used:                             │
│               if 这个位置选中该 expert:                      │
│                   tokens_per_expert[expert]++               │
│                   记录映射索引                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 复制索引回 GPU (同步点 #2)                           │
│   cudaMemcpy(indices_device, indices_host, ...)             │
│   cudaStreamSynchronize(stream)  // 2370 行                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 收集激活 (2375-2378 行)                              │
│   get_rows_cuda(src1, ids_to_sorted) → src1_sorted          │
│   // 按 expert 重排 token 激活                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: 逐 expert 执行 GEMM (2383-2428 行)                   │
│   for (int expert = 0; expert < n_experts; ++expert) {      │
│       if (tokens_per_expert[expert] > 0) {                  │
│           // 创建 expert 权重的视图                          │
│           src0_slice.data = src0->data + expert * nb02      │
│           // 执行矩阵乘法                                    │
│           ggml_cuda_mul_mat(src0_slice, src1_slice, dst)    │
│       }                                                     │
│   }                                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: 分散结果 (2430-2433 行)                              │
│   get_rows_cuda(dst_sorted, ids_from_sorted) → dst          │
│   // 将结果写回原始位置                                       │
└─────────────────────────────────────────────────────────────┘
```

### 关键代码片段

#### Expert 切片创建 (零拷贝)

```cpp
// ggml-cuda.cu:2388-2421
// 不复制数据，使用指针偏移创建 VIEW 张量
ggml_tensor src0_slice = *src0;
src0_slice.data = (char *)src0->data + i02 * nb02;  // expert 权重偏移
src0_slice.ne[2] = 1;  // 单个 expert

ggml_tensor src1_slice = *src1_sorted;
src1_slice.data = (char *)src1_sorted->data + expert_offset * ts1;
src1_slice.ne[1] = tokens_per_expert[i02];  // 该 expert 的 token 数
```

---

## 量化 MoE 支持

**文件位置**: `ggml/src/ggml-cuda/mmq.cuh:3136-3620`

### Expert-aware Kernel 设计

```cpp
template <ggml_type type, int mmq_x, bool need_check, bool fixup>
__global__ void mul_mat_q(
    const char * __restrict__ x,    // 量化权重 (可能在 CPU 内存)
    const float * __restrict__ y,   // 激活 (GPU)
    const int * __restrict__ ids_dst,       // MoE: expert ID 映射
    const int * __restrict__ expert_bounds, // MoE: [expert_low, expert_high)
    float * __restrict__ dst,       // 输出
    ...)
```

### 条件处理逻辑

```cpp
// mmq.cuh:3190-3214
int col_low, col_high;
if (ids_dst != nullptr) {
    // MoE 模式: 只处理当前 expert 对应的列范围
    col_low  = expert_bounds[blockIdx.z];      // expert 开始列
    col_high = expert_bounds[blockIdx.z + 1];  // expert 结束列

    // 无效块提前返回
    if (blockIdx.y * mmq_y >= col_high - col_low) {
        return;
    }

    // 从 ids_dst 加载正确的输出索引到共享内存
    for (int i = threadIdx.x; i < mmq_y; i += WARP_SIZE) {
        ids_dst_shared[i] = ids_dst[col_low + blockIdx.y * mmq_y + i];
    }
} else {
    // 非 MoE 模式: 处理所有列
    col_low = 0;
    col_high = ncols;
}
```

---

## 内存管理与数据布局

### 张量布局

```
输入张量:
┌──────────────────────────────────────────────────────────────┐
│ logits (routing):  [n_experts, n_tokens]     // 路由 logits  │
│ src0 (weights):    [d_model, d_ff, n_experts] // Expert 权重 │
│ src1 (activation): [d_model, n_tokens]       // Token 激活   │
│ ids (selection):   [n_expert_used, n_tokens] // 选中的 IDs   │
└──────────────────────────────────────────────────────────────┘

输出张量:
┌──────────────────────────────────────────────────────────────┐
│ weights:           [n_expert_used, n_tokens] // 路由权重     │
│ dst:               [d_ff, n_tokens]          // 输出激活     │
└──────────────────────────────────────────────────────────────┘
```

### GPU 内存优化

#### 池分配 (减少碎片化)

```cpp
// ggml-cuda.cu:2340, 2344-2345
// 使用 ggml_cuda_pool_alloc<> 而非 cudaMalloc()
ggml_cuda_pool_alloc<char> src1_sorted_pool(ctx.pool(), src1_sorted_size);
ggml_cuda_pool_alloc<char> dst_sorted_pool(ctx.pool(), dst_sorted_size);
ggml_cuda_pool_alloc<int>  indices_pool(ctx.pool(), 2 * ne_get_rows * sizeof(int));
```

#### 类型转换策略

```cpp
// ggml-cuda.cu:2327-2331
// 根据硬件能力选择中间精度
ggml_type type_src1_sorted;
if ((src0_type == GGML_TYPE_F16 && !fp16_mma_available) || is_quantized(src0_type)) {
    type_src1_sorted = GGML_TYPE_F32;  // 需要 FP32 中间结果
} else {
    type_src1_sorted = src0_type;      // 使用原始精度
}
```

### 内存大小估算

| 操作 | 内存大小 | 位置 | 说明 |
|-----|---------|------|------|
| IDs 副本 | `ggml_nbytes(ids)` | CPU | 同步点 |
| 索引缓冲 | `2 * ne_get_rows * 4 bytes` | GPU | 池分配 |
| src1_sorted | `n_tokens * n_expert_used * d_model * sizeof(type)` | GPU | 收集后的激活 |
| dst_sorted | `n_tokens * n_expert_used * d_ff * sizeof(type)` | GPU | 分散前的输出 |

---

## 性能优化技巧

### Kernel 融合

**文件位置**: `ggml/src/ggml-cuda/ggml-cuda.cu:3076-3107`

将多个 MoE 操作融合为单一 kernel：

```cpp
// 原始操作序列
std::initializer_list<enum ggml_op> topk_moe_ops_with_norm = {
    GGML_OP_SOFT_MAX,    // 融合 ┐
    GGML_OP_RESHAPE,     //      │
    GGML_OP_ARGSORT,     //      │
    GGML_OP_VIEW,        //      ├→ topk_moe_cuda kernel
    GGML_OP_GET_ROWS,    //      │
    GGML_OP_RESHAPE,     //      │
    GGML_OP_SUM_ROWS,    //      │
    GGML_OP_CLAMP,       //      │
    GGML_OP_DIV,         //      │
    GGML_OP_RESHAPE      // 融合 ┘
};
```

检测条件:

```cpp
// ggml-cuda.cu:3253
if (ggml_cuda_can_fuse(cgraph, i, topk_moe_ops_with_norm, ...)) {
    // 使用融合 kernel
    ggml_cuda_op_topk_moe(...);
}
```

### Warp 级别高效规约

**文件位置**: `topk-moe.cu:10-51`

使用 CUDA shuffle 实现无共享内存的规约：

```cpp
// 最大值规约
for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
    float val = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
    int   idx = __shfl_xor_sync(0xFFFFFFFF, max_idx, mask, WARP_SIZE);
    if (val > max_val) {
        max_val = val;
        max_idx = idx;
    }
}
```

**优势**:
- 避免共享内存读写延迟
- 减少同步开销
- 更好的寄存器利用

### 延迟 Softmax

**文件位置**: `topk-moe.cu:152-154`

可选优化：只对最终选中的 k 个 logits 应用 softmax，而非预先计算所有 expert 的 softmax。

```cpp
if constexpr (delayed_softmax) {
    // 只对选中的 k 个 expert 计算 softmax
    float sum = 0.0f;
    for (int k = 0; k < n_expert_used; ++k) {
        weights[k] = expf(weights[k] - max_logit);
        sum += weights[k];
    }
    for (int k = 0; k < n_expert_used; ++k) {
        weights[k] /= sum;
    }
}
```

---

## MoE 卸载分析

**参考文档**: `docs/moe_offload_analysis.md`

### 问题: 令牌生成阶段 GPU 利用率低

| 阶段 | 批大小 | GPU 卸载? | 利用率 |
|-----|-------|----------|-------|
| 提示处理 (Prefill) | 100-1000+ | ✅ 是 | 60-90% |
| 令牌生成 (Decode) | 1-8 | ❌ 否 | 5-15% |

### 根本原因

**文件位置**: `ggml-cuda.cu:3694-3700`

```cpp
static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;  // 硬编码阈值!
    return get_op_batch_size(op) >= min_batch_size;  // 1-8 < 32 → 在 CPU 执行
}
```

MoE 批大小计算:

```cpp
// ggml-cuda.cu:3679-3692
case GGML_OP_MUL_MAT_ID:  // MoE op
    return op->ne[2];  // 令牌生成时 = 1-8，小于阈值
```

### 解决方案

将 `min_batch_size` 从 32 改为 1：

```cpp
const int min_batch_size = 1;  // 允许小批量卸载
```

---

## 支持的模型架构

### 标准 Top-K MoE

| 模型 | Expert 数量 | Top-K | 文件 |
|-----|------------|-------|------|
| Mixtral 8x7B | 8 | 2 | `src/models/llama.cpp` |
| Mixtral 8x22B | 8 | 2 | `src/models/llama.cpp` |
| Qwen2-MoE | 可变 | 2-4 | `src/models/qwen2moe.cpp` |
| Qwen3-MoE | 可变 | 2-4 | `src/models/qwen3moe.cpp` |
| DBRX | 16 | 4 | `src/models/dbrx.cpp` |

### 变体架构

| 模型 | 特点 | 文件 |
|-----|------|------|
| GLM-4-MoE | 混合密集-MoE 层 (第一层密集) | `src/models/glm4-moe.cpp` |
| DeepSeek-MoE | routed + shared expert | `src/models/deepseek2.cpp` |
| Arctic | 密集 + MoE 混合 | `src/models/arctic.cpp` |
| AFMoE | 自适应路由 | `src/models/afmoe.cpp` |

### Gating 函数支持

```cpp
// llama-arch.cpp
enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,  // 标准 softmax
    LLAMA_EXPERT_GATING_FUNC_TYPE_TOPK,     // 带阈值的 top-k
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,  // Sigmoid gating
};
```

---

## 关键代码位置速查

| 功能 | 文件 | 行号 | 代码片段 |
|-----|------|------|---------|
| Top-K Kernel 主体 | topk-moe.cu | 63-167 | `__global__ void topk_moe_cuda()` |
| Warp Softmax | topk-moe.cu | 10-51 | `__device__ void softmax_warp_inplace()` |
| Warp 最大值规约 | topk-moe.cu | 105-140 | `__shfl_xor_sync()` 循环 |
| 模板特化 switch | topk-moe.cu | 184-228 | `switch (n_expert)` |
| ID 转换 Kernel | mmid.cu | 28-116 | `__global__ void mm_ids_helper()` |
| 紧凑 ID 编码 | mmid.cu | 15-25 | `struct mm_ids_helper_store` |
| MoE GEMM 主流程 | ggml-cuda.cu | 2288-2434 | `static void ggml_cuda_mul_mat_id()` |
| IDs 复制同步点 | ggml-cuda.cu | 2349 | `cudaStreamSynchronize()` |
| CPU 端路由计算 | ggml-cuda.cu | 2351-2364 | Expert 绑定循环 |
| Expert 切片创建 | ggml-cuda.cu | 2388-2421 | VIEW 张量创建 |
| 量化 MoE Kernel | mmq.cuh | 3136-3620 | `mul_mat_q()` 的 MoE 分支 |
| Expert bounds 处理 | mmq.cuh | 3190-3214 | `if (ids_dst != nullptr)` |
| Fusion 检测 | ggml-cuda.cu | 3076-3107 | `topk_moe_ops_with_norm` |
| 卸载阈值 | ggml-cuda.cu | 3694-3700 | `min_batch_size = 32` |

---

## 性能瓶颈总结

### 主要瓶颈

1. **CPU-GPU 同步点** - 2 次 `cudaStreamSynchronize()` 调用
2. **PCIe 传输** - Expert 权重从 CPU 到 GPU 的隐式传输 (moe_offload 场景)
3. **顺序 Expert 处理** - 无法并行化不同 expert 的 GEMM
4. **小批量阈值限制** - 令牌生成时批大小太小，无法卸载到 GPU

### 优化建议

1. **异步传输** - 使用 CUDA streams 实现 D2H/H2D 重叠
2. **Expert 级并行** - 在不同 stream 上并行执行 expert GEMM
3. **可配置阈值** - 允许用户根据硬件调整卸载阈值
4. **GPU 侧专家缓存** - 持久化常用 expert 权重在 GPU 内存
5. **异步专家预取** - 预测下一个 token 可能使用的 expert 并提前加载

---

## 附录: MoE 数据流图

```
                    ┌─────────────┐
                    │   Logits    │
                    │ [E, T]      │
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   topk_moe_cuda()      │
              │   - Softmax            │
              │   - Top-K Selection    │
              │   - Weight Norm        │
              └────────────┬───────────┘
                           │
              ┌────────────┴───────────┐
              │                        │
              ▼                        ▼
       ┌─────────────┐         ┌─────────────┐
       │   Weights   │         │    IDs      │
       │ [K, T]      │         │ [K, T]      │
       └─────────────┘         └──────┬──────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │   mm_ids_helper()     │
                          │   - Compact encoding  │
                          │   - Expert bounds     │
                          └───────────┬───────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
                     ▼                                 ▼
              ┌─────────────┐                 ┌─────────────────┐
              │  ids_src1   │                 │  expert_bounds  │
              │  ids_dst    │                 │  [E+1]          │
              └──────┬──────┘                 └────────┬────────┘
                     │                                 │
                     ▼                                 ▼
              ┌─────────────┐                 ┌─────────────────┐
              │  get_rows() │                 │  For each E:    │
              │  Gather     │                 │  mul_mat()      │
              └──────┬──────┘                 └────────┬────────┘
                     │                                 │
                     ▼                                 ▼
              ┌─────────────┐                 ┌─────────────────┐
              │ src1_sorted │ ───────────────▶│   dst_sorted    │
              │ [D, K*T]    │                 │   [F, K*T]      │
              └─────────────┘                 └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   get_rows()    │
                                              │   Scatter       │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │      dst        │
                                              │   [F, T]        │
                                              └─────────────────┘

Legend:
  E = n_experts
  T = n_tokens
  K = n_expert_used (top-k)
  D = d_model
  F = d_ff
```

---

## Expert 计算结果回传与预取机制分析

### 核心问题

**Q: Expert 计算完成后，是否将结果传回 CPU 进行预取 (prefetch)？**

**A: 否。llama.cpp 当前实现中没有将 expert 计算结果传回 CPU，也没有实现 expert 预取机制。**

### 详细分析

#### 1. 当前数据流向

通过分析 `ggml-cuda.cu` 中的 MoE 实现，数据流向如下：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MoE 数据流向分析                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  GPU → CPU (同步点 #1)                                                  │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ cudaMemcpyAsync(ids_host, ids->data, DeviceToHost)             │    │
│  │ cudaStreamSynchronize(stream)  // 阻塞等待                     │    │
│  │                                                                 │    │
│  │ 传输内容: expert routing IDs (选择了哪些 expert)               │    │
│  │ 大小: n_tokens * n_expert_used * sizeof(int32_t)               │    │
│  │ 目的: CPU 端计算 token → expert 映射索引                       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  CPU 端计算 (2351-2364 行)                                              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ 遍历所有 expert 和 token，计算:                                 │    │
│  │   - tokens_per_expert[]: 每个 expert 处理多少 token            │    │
│  │   - ids_to_sorted[]: gather 索引                               │    │
│  │   - ids_from_sorted[]: scatter 索引                            │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  CPU → GPU (同步点 #2)                                                  │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ cudaMemcpyAsync(ids_buf_dev, ids_to_sorted_host, HostToDevice) │    │
│  │ cudaStreamSynchronize(stream)  // 阻塞等待                     │    │
│  │                                                                 │    │
│  │ 传输内容: 排序后的索引映射                                      │    │
│  │ 大小: 2 * n_get_rows * sizeof(int32_t)                         │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  GPU 端计算 (全程在 GPU，结果不回传 CPU)                                │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ 1. get_rows_cuda(): 收集激活到 src1_sorted (GPU)               │    │
│  │ 2. for each expert:                                             │    │
│  │       ggml_cuda_mul_mat(): GEMM 计算 (GPU)                     │    │
│  │ 3. get_rows_cuda(): 分散结果到 dst (GPU)                       │    │
│  │                                                                 │    │
│  │ ⚠️ 计算结果始终保留在 GPU 内存，不传回 CPU                     │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 2. 关键代码证据

**传回 CPU 的数据 (仅 routing IDs):**

```cpp
// ggml-cuda.cu:2347-2349
std::vector<char> ids_host(ggml_nbytes(ids));
CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids),
                           cudaMemcpyDeviceToHost, stream));  // ← 只传 IDs
CUDA_CHECK(cudaStreamSynchronize(stream));
```

**计算结果留在 GPU:**

```cpp
// ggml-cuda.cu:2430-2433
// dst_sorted 在 GPU 内存中，直接 scatter 到 dst (也在 GPU)
get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, ...);
// ↑ 没有 DeviceToHost 的 memcpy，结果不回传 CPU
```

#### 3. 为什么没有实现 Prefetch？

| 原因 | 说明 |
|------|------|
| **架构限制** | MoE 层之间是顺序执行的，当前层的输出是下一层的输入 |
| **预测困难** | 无法在当前层计算完成前准确预测下一层会选择哪些 expert |
| **内存带宽** | 预取需要额外的 PCIe 带宽，可能与当前计算冲突 |
| **实现复杂度** | 需要维护 expert 缓存、LRU 策略、异步传输管理 |

#### 4. 搜索验证

通过代码搜索确认：

```bash
# 搜索 cudaMemPrefetchAsync (CUDA 统一内存预取)
grep -r "cudaMemPrefetch" ggml-cuda/ → 无结果

# 搜索 expert 预取相关逻辑
grep -ri "prefetch.*expert\|expert.*prefetch" → 无结果

# 搜索计算结果回传
grep "DeviceToHost" ggml-cuda.cu | grep -v "ids\|index" → 无与 expert 输出相关的结果
```

### 当前实现的数据传输统计

| 传输方向 | 数据内容 | 大小 | 频率 |
|---------|---------|------|------|
| GPU → CPU | Expert routing IDs | `n_tokens * n_expert_used * 4 bytes` | 每层 1 次 |
| CPU → GPU | 排序索引映射 | `2 * n_tokens * n_expert_used * 4 bytes` | 每层 1 次 |
| CPU → GPU (隐式) | Expert 权重 (offload 场景) | `expert_size * n_expert_used` | 每层 1 次 |
| GPU → CPU | **Expert 计算结果** | **0 (不传输)** | - |

### 潜在的 Prefetch 优化方案

虽然当前未实现，但以下方案在理论上可行：

#### 方案 1: 基于历史的 Expert 预取

```cpp
// 伪代码 - 未实现
class ExpertPredictor {
    // 记录每个 token 位置历史上选择的 expert
    std::vector<std::vector<int>> expert_history;

    std::vector<int> predict_next_experts(int layer, int token_pos) {
        // 基于历史统计预测下一层可能使用的 expert
        return most_frequent_experts(expert_history[layer + 1][token_pos]);
    }
};

void prefetch_experts(int layer, std::vector<int> predicted_experts) {
    cudaStream_t prefetch_stream;
    for (int expert_id : predicted_experts) {
        cudaMemcpyAsync(gpu_cache[expert_id], cpu_weights[expert_id],
                        expert_size, cudaMemcpyHostToDevice, prefetch_stream);
    }
    // 不等待，异步传输
}
```

#### 方案 2: GPU 侧 Expert 缓存 + LRU

```cpp
// 伪代码 - 未实现
class GPUExpertCache {
    std::unordered_map<int, void*> cached_experts;  // expert_id → GPU ptr
    std::list<int> lru_order;
    size_t cache_size;

    void* get_expert(int expert_id, void* cpu_ptr, size_t size) {
        if (cached_experts.count(expert_id)) {
            // 命中，更新 LRU
            update_lru(expert_id);
            return cached_experts[expert_id];
        }
        // 未命中，可能需要驱逐
        if (need_eviction()) evict_lru();
        // 同步加载
        void* gpu_ptr = allocate_and_copy(cpu_ptr, size);
        cached_experts[expert_id] = gpu_ptr;
        return gpu_ptr;
    }
};
```

#### 方案 3: 双缓冲异步传输

```cpp
// 伪代码 - 未实现
// 使用两个 stream 实现计算和传输重叠
cudaStream_t compute_stream, transfer_stream;

// Layer N 计算
for (int expert : selected_experts_layer_n) {
    ggml_cuda_mul_mat(..., compute_stream);
}

// 同时预取 Layer N+1 的 expert (如果能预测)
for (int expert : predicted_experts_layer_n1) {
    cudaMemcpyAsync(buffer, cpu_weights[expert], size,
                    cudaMemcpyHostToDevice, transfer_stream);
}
```

### 为什么不传回计算结果做预取？

**根本原因: 无法利用计算结果预测未来的 expert 选择**

```
Layer N 的数据流:
  input → Router (MLP) → Softmax → Top-K → Expert Selection
                                            ↓
                                    Selected Expert IDs
                                            ↓
                              Expert Computation (GEMM)
                                            ↓
                                        output

问题: output 是 FFN 的输出，不是 routing logits
      下一层的 expert 选择取决于 Layer N+1 的 Router 输入
      而 Router 输入 = Attention(output) + output (残差连接)
      → 需要完成整个 Attention 计算才能知道下一层选哪些 expert
      → 无法提前预测，因此无法有效预取
```

### 结论

1. **当前实现不传回 expert 计算结果到 CPU** - 结果直接留在 GPU 用于后续层
2. **没有实现 expert 预取机制** - 代码中无 `cudaMemPrefetchAsync` 或类似逻辑
3. **主要原因是预测困难** - MoE 的 expert 选择依赖于上一层的完整输出，无法提前预测
4. **优化空间存在** - 可以考虑基于历史统计的预测、GPU 侧缓存、或异步双缓冲等方案

---

## CPU 参与 Expert 路由的分析 (正常模式)

### 核心问题

**Q: 在正常模式（权重全在 GPU）下，CPU 是否参与 expert 路由？**

**A: 是的。即使权重全在 GPU 上，CPU 仍然参与 expert 路由的索引计算，存在 2 次 CPU-GPU 同步点。**

### 代码证据

**文件位置**: `ggml/src/ggml-cuda/ggml-cuda.cu:2347-2370`

```cpp
// Step 1: GPU → CPU 传输 expert routing IDs (同步点 #1)
std::vector<char> ids_host(ggml_nbytes(ids));
cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids),
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);  // ⚠️ GPU 等待传输完成

// Step 2: CPU 端遍历计算 token → expert 映射
for (int64_t i02 = 0; i02 < ne02; ++i02) {      // 每个 expert
    for (int64_t i12 = 0; i12 < ne12; ++i12) {  // 每个 token
        for (int64_t iex = 0; iex < n_expert_used; ++iex) {
            const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + ...);
            if (expert_to_use == i02) {
                tokens_per_expert[i02]++;       // 统计每个 expert 的 token 数
                ids_from_sorted_host[...] = ids_to_sorted_host.size();
                ids_to_sorted_host.push_back(...);
            }
        }
    }
}

// Step 3: CPU → GPU 传输索引映射 (同步点 #2)
cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(),
                2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream);
cudaStreamSynchronize(stream);  // ⚠️ GPU 再次等待
```

### 数据流图 (正常模式 vs Offload 模式)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    正常模式 (权重在 GPU) - CPU 仍参与路由                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GPU: Router MLP → Softmax → Top-K                                           │
│              ↓                                                               │
│       Expert IDs [n_tokens, n_expert_used]                                  │
│              ↓                                                               │
│  ════════════════════════════════════════════════════════════════            │
│       cudaMemcpy DeviceToHost  (同步点 #1) ← GPU 等待                        │
│  ════════════════════════════════════════════════════════════════            │
│              ↓                                                               │
│  CPU: 计算 tokens_per_expert[]                                               │
│        计算 ids_to_sorted[] (gather 索引)                                    │
│        计算 ids_from_sorted[] (scatter 索引)                                 │
│              ↓                                                               │
│  ════════════════════════════════════════════════════════════════            │
│       cudaMemcpy HostToDevice  (同步点 #2) ← GPU 等待                        │
│  ════════════════════════════════════════════════════════════════            │
│              ↓                                                               │
│  GPU: get_rows_cuda (gather 激活)                                            │
│        for each expert:                                                      │
│            ggml_cuda_mul_mat (GEMM) ← 权重已在 GPU，无 PCIe 传输             │
│        get_rows_cuda (scatter 结果)                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    Offload 模式 (权重在 CPU) - 额外 PCIe 传输                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  (前面步骤相同...)                                                            │
│              ↓                                                               │
│  GPU: get_rows_cuda (gather 激活)                                            │
│        for each expert:                                                      │
│            ggml_cuda_mul_mat (GEMM) ← ⚠️ 权重从 CPU 隐式传输到 GPU           │
│        get_rows_cuda (scatter 结果)                                          │
│                                                                              │
│  关键区别: Offload 模式下每个 expert 的 GEMM 都需要 PCIe 传输权重            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CPU 计算内容

| CPU 计算项 | 说明 | 复杂度 |
|-----------|------|--------|
| `tokens_per_expert[]` | 每个 expert 被分配了多少 token | O(n_experts × n_tokens × n_expert_used) |
| `ids_to_sorted[]` | gather 索引，用于收集激活到连续内存 | 同上 |
| `ids_from_sorted[]` | scatter 索引，用于分散结果到原位置 | 同上 |

### 为什么 CPU 参与？

| 原因 | 说明 |
|------|------|
| **实现简单** | CPU 上的三层循环实现简单，避免复杂的 GPU kernel |
| **动态负载** | 每个 expert 的 token 数量动态变化，CPU 处理更灵活 |
| **历史原因** | 早期实现未优化，后续可能迁移到 GPU |

### 性能影响

```
时间线分析 (单次 MoE 层):

─────────────────────────────────────────────────────────────────────────────
GPU:  [Router计算]──[等待]──────────────────[等待]──[Gather]─[GEMM×N]─[Scatter]
                     ↑                       ↑
CPU:              [D2H传输][索引计算][H2D传输]
─────────────────────────────────────────────────────────────────────────────
                     同步点#1               同步点#2

问题: GPU 在两个同步点处于空闲状态
```

**影响程度**:

| 阶段 | batch size | 同步开销占比 | 影响程度 |
|------|-----------|-------------|---------|
| Prompt 处理 | 100-1000+ | 低 (~1-5%) | 轻微 |
| Token 生成 | 1-8 | 高 (~10-30%) | 显著 |

### 与 Offload 阈值的关系

`min_batch_size = 32` 的 offload 阈值检查与 CPU 参与路由是**两个独立的问题**：

| 问题 | 触发条件 | 影响 |
|------|---------|------|
| **CPU 参与路由** | 所有 MoE 操作 | 2 次同步点开销 |
| **Offload 阈值** | 权重在 CPU + batch < 32 | 整个 MoE 在 CPU 执行 |

```
                           权重位置
                    ┌────────┴────────┐
                    GPU              CPU
                    │                 │
            CPU参与路由          CPU参与路由
            (2次同步)           (2次同步)
                    │                 │
                    │            batch >= 32?
                    │           ┌────┴────┐
                    │          Yes        No
                    │           │          │
                GPU执行GEMM   GPU执行GEMM  CPU执行全部
                (正常性能)    (PCIe瓶颈)   (最慢)
```

### 优化建议

#### 1. GPU 侧索引计算 (已有部分实现)

`mmid.cu` 中的 `mm_ids_helper` kernel 可在 GPU 上完成索引计算，但主流程未使用：

```cpp
// mmid.cu:28-116
__global__ void mm_ids_helper(
    const int32_t * ids,      // expert routing IDs
    int32_t * ids_src1,       // 输出: gather 索引
    int32_t * ids_dst,        // 输出: scatter 索引
    int * expert_bounds,      // 输出: 每个 expert 的范围
    ...)
```

**优化方向**: 将主流程迁移到使用 GPU 侧索引计算

#### 2. 异步索引计算

```cpp
// 伪代码 - 优化方案
cudaStream_t compute_stream, index_stream;

// 当前层的 Router 计算
router_forward(input, expert_ids, compute_stream);

// 异步传输 IDs 到 CPU (不等待)
cudaMemcpyAsync(ids_host, expert_ids, ..., DeviceToHost, index_stream);

// GPU 继续其他工作...
// ...

// 需要时才同步
cudaStreamSynchronize(index_stream);
compute_indices_cpu(ids_host, ...);
```

#### 3. 融合 kernel

将路由计算、索引生成、gather 融合为单一 kernel，避免 CPU 参与：

```cpp
// 理想的融合 kernel
__global__ void fused_moe_route_and_gather(
    const float * router_logits,
    const float * activations,
    float * gathered_activations,
    int * expert_assignments,
    ...)
{
    // 1. Top-K 选择 (当前已在 GPU)
    // 2. 索引计算 (当前在 CPU)
    // 3. Gather 激活 (当前已在 GPU)
    // → 全部融合，无 CPU 参与
}
```

### 总结

| 模式 | CPU 参与路由 | GEMM 执行位置 | PCIe 传输 | 性能瓶颈 |
|------|-------------|--------------|----------|---------|
| 正常模式 (权重在 GPU) | ✅ 是 | GPU | 仅索引 | 同步点开销 |
| Offload + batch≥32 | ✅ 是 | GPU | 索引 + 权重 | PCIe 带宽 |
| Offload + batch<32 | ✅ 是 | CPU | 仅索引 | CPU 计算 |

**关键发现**: 即使在"正常模式"下，MoE 实现也存在 CPU-GPU 同步瓶颈，这是独立于 offload 问题的另一个优化点。

---

*文档更新时间: 2025-12-03*
*基于 llama.cpp 仓库分析*
