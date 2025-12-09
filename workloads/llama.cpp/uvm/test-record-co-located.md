# Test 

## Single llama.cpp baseline

in llama.cpp dir

GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 /home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/llama-server --gpt-oss-20b-default -c 65536

============ Serving Benchmark Result ============
Successful requests:                     100       
Maximum request concurrency:             1         
Benchmark duration (s):                  88.87     
Total input tokens:                      23260     
Total generated tokens:                  22380     
Request throughput (req/s):              1.13      
Output token throughput (tok/s):         251.83    
Peak output token throughput (tok/s):    270.00    
Peak concurrent requests:                5.00      
Total Token throughput (tok/s):          513.57    
---------------Time to First Token----------------
Mean TTFT (ms):                          63.70     
Median TTFT (ms):                        70.27     
P99 TTFT (ms):                           98.48     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          3.67      
Median TPOT (ms):                        3.70      
P99 TPOT (ms):                           3.91      
---------------Inter-token Latency----------------
Mean ITL (ms):                           3.74      
Median ITL (ms):                         3.76      
P99 ITL (ms):                            3.93      
==================================================

## Single GCN training

time CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
        --dataset random --nodes 8000000 \
        --edges_per_node 10 --features 128 --hidden 256 \
        --epochs 50 --warmup 1 --prop chunked --use_uvm 

[UVM] Alloc #19057: 2.05 GB (total: 29.31 GB, peak: 36.09 GB)
[UVM] Alloc #19058: 2.05 GB (total: 31.35 GB, peak: 36.09 GB)
[UVM] Alloc #19059: 2.05 GB (total: 29.31 GB, peak: 36.09 GB)
======================================================================
Results Summary
======================================================================
Avg epoch time: 8.975s
Median epoch time: 9.023s
Total training time: 17.95s

Accuracy:
  train: 0.1001

UVM Statistics:
  Peak allocated: 36.09 GB
  Allocations: 3119
  Frees: 2574
======================================================================

## Co-located

### baseline UVM

~/workspace/gpu/schedcp/workloads/llama.cpp$ uv run vllm bench serve --model  Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts  100 --dataset-path /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json  --base-url http://127.0.0.1:8013  --max-concurrency=1 --request-rate 0.2

============ Serving Benchmark Result ============
Successful requests:                     100       
Maximum request concurrency:             1         
Request rate configured (RPS):           0.20      
Benchmark duration (s):                  563.29    
Total input tokens:                      23260     
Total generated tokens:                  22380     
Request throughput (req/s):              0.18      
Output token throughput (tok/s):         39.73     
Peak output token throughput (tok/s):    144.00    
Peak concurrent requests:                3.00      
Total Token throughput (tok/s):          81.02     
---------------Time to First Token----------------
Mean TTFT (ms):                          428.24    
Median TTFT (ms):                        323.23    
P99 TTFT (ms):                           1391.61   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          19.73     
Median TPOT (ms):                        16.97     
P99 TPOT (ms):                           56.06     
---------------Inter-token Latency----------------
Mean ITL (ms):                           17.79     
Median ITL (ms):                         8.42      
P99 ITL (ms):                            116.63    
==================================================


time CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
        --dataset random --nodes 8000000 \
        --edges_per_node 10 --features 128 --hidden 256 \
        --epochs 25 --warmup 1 --prop chunked --use_uvm \
        --report_json multi/100.json

[UVM] Alloc #19057: 2.05 GB (total: 29.31 GB, peak: 36.09 GB)
[UVM] Alloc #19058: 2.05 GB (total: 31.35 GB, peak: 36.09 GB)
[UVM] Alloc #19059: 2.05 GB (total: 29.31 GB, peak: 36.09 GB)
======================================================================
Results Summary
======================================================================
Avg epoch time: 23.230s
Median epoch time: 21.657s
Total training time: 580.75s

Accuracy:
  train: 0.1006

UVM Statistics:
  Peak allocated: 36.09 GB
  Allocations: 23934
  Frees: 19433
======================================================================


### UVM with BPF


~/workspace/gpu/schedcp/workloads/llama.cpp$ uv run vllm bench serve --model  Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts  100 --dataset-path /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json  --base-url http://127.0.0.1:8013  --max-concurrency=1 --request-rate 0.2

.
Traffic request rate: 0.2
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|█████████| 100/100 [08:22<00:00,  5.02s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     100       
Maximum request concurrency:             1         
Request rate configured (RPS):           0.20      
Benchmark duration (s):                  502.38    
Total input tokens:                      23260     
Total generated tokens:                  22380     
Request throughput (req/s):              0.20      
Output token throughput (tok/s):         44.55     
Peak output token throughput (tok/s):    270.00    
Peak concurrent requests:                4.00      
Total Token throughput (tok/s):          90.85     
---------------Time to First Token----------------
Mean TTFT (ms):                          341.48    
Median TTFT (ms):                        209.25    
P99 TTFT (ms):                           1202.97   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.86     
Median TPOT (ms):                        9.72      
P99 TPOT (ms):                           33.37     
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.66      
Median ITL (ms):                         4.80      
P99 ITL (ms):                            73.53     
==================================================

time CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py         --dataset random --nodes 8000000         --edges_per_node 10 --features 128 --hidden 256         --epochs 25 --warmup 1 --prop chunked --use_uvm  --wait-for-bpf 

[UVM] Alloc #19057: 2.05 GB (total: 29.31 GB, peak: 36.09 GB)
[UVM] Alloc #19058: 2.05 GB (total: 31.35 GB, peak: 36.09 GB)
[UVM] Alloc #19059: 2.05 GB (total: 29.31 GB, peak: 36.09 GB)
======================================================================
Results Summary
======================================================================
Avg epoch time: 16.718s
Median epoch time: 15.527s
Total training time: 417.95s

Accuracy:
  train: 0.1006

UVM Statistics:
  Peak allocated: 36.09 GB
  Allocations: 23934
  Frees: 19433
======================================================================

## bpf status



---

# Multi-Tenant Evaluation Plan: LLM Inference + GNN Training

## 目标

评估 llama.cpp (LC, Latency Critical) 与 PyTorch GNN training (BE, Best Effort) 在同一 GPU 上 co-located 运行时的性能，验证 gBPF per-tenant policy 的效果。

---

## 评估指标

### llama.cpp (High Priority, LC)

| 指标 | 描述 | 评估目的 |
|------|------|---------|
| **TTFT** (Time to First Token) | Mean/Median/P99 | 首 token 延迟，衡量 prefill 阶段 |
| **TPOT** (Time per Output Token) | Mean/Median/P99 | 每 token 生成延迟，衡量 decode 阶段 |
| **Token throughput** (tok/s) | Output token 吞吐量 | 整体服务能力 |

### PyTorch GNN (Low Priority, BE)

| 指标 | 描述 | 评估目的 |
|------|------|---------|
| **Epoch time** (s) | 每个 epoch 的训练时间 | 训练速度 |
| **Throughput** (epoch/s or samples/s) | 训练吞吐量 | BE 任务的资源利用 |

### 系统指标

| 指标 | 描述 |
|------|------|
| GPU utilization | SM 利用率 |
| GPU memory | 各租户内存占用 |

---

## 实验配置

### 硬件
- GPU: NVIDIA RTX 5090 (32GB)
- CPU: Intel Core Ultra 9 285K
- Memory: 128GB DDR5

### 工作负载参数

**llama.cpp**:
```bash
# Model: gpt-oss-20b (fits in memory)
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
  /path/to/llama-server --gpt-oss-20b-default -c 65536
```

**GNN Training**:
```bash
# 10M nodes, UVM enabled (oversubscribed)
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 python benchmark_gnn_uvm.py \
  --dataset random --nodes 10000000 --use_uvm --epochs 5
```

---

## 实验方案

### Phase 1: Single Baseline (已完成部分)

| Workload | Condition | 关键结果 |
|----------|-----------|---------|
| llama.cpp single | UVM, 20B model | TTFT ~70ms, TPOT ~3.7ms |
| GNN single | UVM, 10M nodes | ~69.87s/epoch (baseline) |
| GNN single | UVM + eBPF | ~27.43s/epoch (优化后) |

### Phase 2: Co-located Experiments

| 实验 | llama.cpp Policy | GNN Policy | 预期 |
|------|------------------|-----------|------|
| **Exp 1**: Global FIFO | Default UVM | Default UVM | Both degraded |
| **Exp 2**: Framework-managed | llama.cpp offload | Default UVM | LC may recover |
| **Exp 3**: gBPF per-tenant | gBPF LC priority | gBPF BE quota | LC ~solo, BE ~baseline |

### Phase 3: Stress Test

逐步增加 GNN 负载（8M → 10M → 12M nodes），观察 llama.cpp latency 变化。

---

## 测试脚本

### run_colocated.sh

```bash
#!/bin/bash
# Multi-tenant co-location test

RESULT_DIR="result_colocated"
mkdir -p $RESULT_DIR

echo "============================================"
echo "Multi-Tenant Test: llama.cpp + GNN Training"
echo "============================================"

# Step 1: Start llama.cpp server in background
echo "[1/4] Starting llama.cpp server..."
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
  /home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/llama-server \
  --gpt-oss-20b-default -c 65536 &
LLAMA_PID=$!
sleep 30  # Wait for model loading

# Step 2: Start GNN training in background
echo "[2/4] Starting GNN training..."
cd /home/yunwei37/workspace/gpu/schedcp/workloads/pytorch
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 python benchmark_gnn_uvm.py \
  --dataset random --nodes 10000000 --use_uvm --epochs 5 \
  --report_json ../$RESULT_DIR/gnn_colocated.json &
GNN_PID=$!

# Step 3: Run llama.cpp benchmark
echo "[3/4] Running llama.cpp benchmark..."
sleep 10  # Let GNN warm up
uv run vllm bench serve \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --dataset-name sharegpt --num-prompts 100 \
  --dataset-path /path/to/ShareGPT.json \
  --base-url http://127.0.0.1:8013 \
  --max-concurrency=1 2>&1 | tee $RESULT_DIR/llama_colocated.log

# Step 4: Wait for GNN to finish
echo "[4/4] Waiting for GNN training to complete..."
wait $GNN_PID

# Cleanup
kill $LLAMA_PID 2>/dev/null

echo "============================================"
echo "Done! Results in $RESULT_DIR/"
echo "============================================"
```

---

## 图表设计

### Figure 1: LC Latency Comparison (Normalized)

**目的**: 展示 co-located 对 llama.cpp latency 的影响

```
                 Normalized Latency (vs Single Baseline)

TTFT    ■ Single    ■ Global FIFO    ■ gBPF
        ──────────────────────────────────────
        1.0×        ~3-5×           ~1.1×

TPOT    ■ Single    ■ Global FIFO    ■ gBPF
        ──────────────────────────────────────
        1.0×        ~2-3×           ~1.05×

P99 TTFT ■ Single   ■ Global FIFO    ■ gBPF
         ─────────────────────────────────────
         1.0×       ~10-20×          ~1.2×
```

**图表类型**: Grouped bar chart, y-axis normalized to single baseline

### Figure 2: BE Throughput Comparison

**目的**: 展示 co-located 对 GNN training 的影响

```
                 GNN Epoch Time (s)
        │
    100 │  ████
        │  ████  ████
     50 │  ████  ████
        │  ████  ████  ████
      0 │  ░░░░  ████  ████
        └─────────────────────
          Single  FIFO  gBPF

    ████ Measured   ░░░░ Ideal (no interference)
```

**图表类型**: Bar chart with error bars

### Figure 3: Two-Tenant Tradeoff Space

**目的**: 展示 LC latency vs BE throughput 的权衡

```
        GNN Throughput (% of single baseline)
        │
   100% │    gBPF  ●
        │
    75% │  Framework ●
        │
    50% │         ● Global FIFO
        │
     0% │──────────────────────────
           1×     2×      5×     10×
               LC P99 TTFT (normalized)

    Ideal: top-left corner (high BE throughput, low LC latency)
```

**图表类型**: Scatter plot with annotations

---

## 数据收集 Checklist

| Experiment | llama.cpp TTFT | llama.cpp TPOT | llama.cpp P99 | GNN epoch/s | GPU util |
|------------|---------------|----------------|---------------|-------------|----------|
| Single llama.cpp | ✅ 63.70ms | ✅ 3.67ms | ✅ 98.48ms | - | ⬜ |
| Single GNN | - | - | - | ✅ 0.014/s | ⬜ |
| Co-loc Global FIFO | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Co-loc Framework | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Co-loc gBPF | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

---

## 预期结论

1. **Global FIFO (baseline)**: 两者都严重退化，LC P99 latency 可能 10× worse
2. **Framework-managed**: LC 部分恢复，但 BE 可能被完全饿死
3. **gBPF per-tenant**: LC 接近 solo 性能 (within 10-20%)，BE 维持 reasonable throughput (70-80% of solo)

---

## Notes

- llama.cpp 和 GNN 都使用 UVM，会竞争 GPU memory 和 PCIe bandwidth
- 确保 llama.cpp server 完全启动后再启动 GNN
- 使用 `nvidia-smi dmon` 监控实时 GPU 状态
- 建议每个配置运行 3 次取平均值

