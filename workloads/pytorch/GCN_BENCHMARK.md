# GCN UVM Oversubscription Experiment Plan

## Objective

测试 CUDA Unified Virtual Memory (UVM) 在显存超额分配 (oversubscription) 场景下的性能表现，并评估内核调度优化 (eBPF scheduler) 对 UVM 性能的影响。

---

## Test Environment

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 5090 |
| **GPU Memory** | 33.67 GB (31.36 GB usable) |
| **PyTorch** | 2.9.0+cu128 |
| **CUDA** | 12.8 |
| **Workload** | GCN (Graph Convolutional Network) |

---

## Memory Estimation

**简化估算公式** (F=128, H=256, edges_per_node=10):

```
Peak(GB) ≈ N × 0.0045 + 0.5
```

| Nodes | Edges | Estimated Peak | vs GPU (31.36 GB) |
|-------|-------|----------------|-------------------|
| 5M | 50M | ~23 GB | 73% (无需 UVM) |
| 7M | 70M | ~32 GB | **102% (临界点)** |
| 10M | 100M | ~45 GB | **143% (中度)** |
| 12M | 120M | ~54 GB | **172% (重度)** |
| 15M | 150M | ~68 GB | **217% (极限)** |

---

## Existing Results

### Summary (OSDI Eval Style)

We evaluate UVM oversubscription performance using a GCN training workload on an NVIDIA RTX 5090 GPU (31.36 GB usable memory). The workload trains a 2-layer GCN with 256 hidden dimensions on a random graph with 10M nodes and 100M edges, requiring 45.11 GB peak memory allocation—**1.44× the physical GPU capacity**.

Without UVM, training fails immediately with an out-of-memory error. With UVM enabled, training completes successfully but suffers a **4.5× slowdown** (69.87s vs 15.72s per epoch) due to page migration overhead between GPU and host memory. Notably, applying an eBPF-based kernel scheduler optimization reduces epoch time to 27.43s, achieving a **2.5× speedup** over baseline UVM while maintaining identical memory allocation patterns. This demonstrates that CPU scheduling policies significantly impact UVM page fault handling performance, even for GPU-intensive workloads.

### Results Table (10M nodes)

| Condition | Epoch Time | Peak Memory | Oversubscription | Status |
|-----------|------------|-------------|------------------|--------|
| No UVM | N/A | >31 GB | N/A | OOM |
| UVM Baseline | **69.87s** | 45.11 GB | 1.44× | Success |
| UVM + eBPF | **27.43s** | 45.11 GB | 1.44× | **2.5× faster** |

### Data Sources

| File | Config | Key Metrics |
|------|--------|-------------|
| `results_uvm_10m.json` | 10M nodes, UVM | 69.87s/epoch |
| `result/gcn_random_chunked_20251125_*.json` | 5M nodes, no UVM | 1.30s/epoch, 5.53 GB |

---

## Experiment Scripts

### 三个独立脚本，输出到不同文件夹

| Script | Condition | Output Directory |
|--------|-----------|------------------|
| `run_no_uvm.sh` | 纯 GPU，不开 UVM | `result_no_uvm/` |
| `run_uvm_baseline.sh` | 开 UVM，默认调度器 | `result_uvm_baseline/` |
| `run_uvm_ebpf.sh` | 开 UVM，需先启动 eBPF | `result_uvm_ebpf/` |

---

### Script 1: No UVM (Pure GPU)

```bash
#!/bin/bash
# run_no_uvm.sh - 纯 GPU 测试，不开 UVM

RESULT_DIR="result_no_uvm"
mkdir -p $RESULT_DIR

echo "========================================"
echo "No UVM Test - Pure GPU Memory"
echo "Output: $RESULT_DIR/"
echo "========================================"

# 小规模 (应该成功)
for NODES in 1000000 3000000 5000000; do
    echo "=== Testing ${NODES} nodes (no UVM) ==="
    uv run python benchmark_gnn_uvm.py --dataset random --nodes $NODES \
        --edges_per_node 10 --features 128 --hidden 256 \
        --epochs 2 --warmup 1 --prop chunked  --use_gpu_allocator \
        --report_json $RESULT_DIR/${NODES}.json 2>&1 | tee $RESULT_DIR/${NODES}.log
done

# 临界点 (可能 OOM)
for NODES in 7000000 8000000 10000000; do
    echo "=== Testing ${NODES} nodes (no UVM, may OOM) ==="
    timeout 300 uv run python benchmark_gnn_uvm.py --dataset random --nodes $NODES \
        --edges_per_node 10 --features 128 --hidden 256 \
        --epochs 1 --warmup 0 --prop chunked  --use_gpu_allocator \
        --report_json $RESULT_DIR/${NODES}.json 2>&1 | tee $RESULT_DIR/${NODES}.log || echo "OOM or timeout"
done

echo "========================================"
echo "Done! Results saved to $RESULT_DIR/"
echo "========================================"
```

---

### Script 2: UVM Baseline (Default Scheduler)

```bash
#!/bin/bash
# run_uvm_baseline.sh - UVM 测试，默认调度器

RESULT_DIR="result_uvm_baseline"
mkdir -p $RESULT_DIR

echo "========================================"
echo "UVM Baseline Test - Default Scheduler"
echo "Output: $RESULT_DIR/"
echo "========================================"

for NODES in 5000000 7000000 8000000 10000000 12000000 15000000; do
    echo "=== Testing ${NODES} nodes (UVM baseline) ==="
    CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
        --dataset random --nodes $NODES \
        --edges_per_node 10 --features 128 --hidden 256 \
        --epochs 2 --warmup 1 --prop chunked --use_uvm \
        --report_json $RESULT_DIR/${NODES}.json 2>&1 | tee $RESULT_DIR/${NODES}.log
done

echo "========================================"
echo "Done! Results saved to $RESULT_DIR/"
echo "========================================"
```

---

### Script 3: UVM + eBPF Scheduler

```bash
#!/bin/bash
# run_uvm_ebpf.sh - UVM 测试，需先启动 eBPF 调度器

RESULT_DIR="result_uvm_ebpf"
mkdir -p $RESULT_DIR

echo "========================================"
echo "UVM + eBPF Scheduler Test"
echo "Output: $RESULT_DIR/"
echo ""
echo "IMPORTANT: Before running this script, start eBPF scheduler:"
echo "  sudo schedcp-cli run scx_simple"
echo "========================================"
read -p "Press Enter to continue (or Ctrl+C to cancel)..."

for NODES in 5000000 7000000 8000000 10000000 12000000 15000000; do
    echo "=== Testing ${NODES} nodes (UVM + eBPF) ==="
    CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
        --dataset random --nodes $NODES \
        --edges_per_node 10 --features 128 --hidden 256 \
        --epochs 2 --warmup 1 --prop chunked --use_uvm \
        --report_json $RESULT_DIR/${NODES}.json 2>&1 | tee $RESULT_DIR/${NODES}.log
done

echo "========================================"
echo "Done! Results saved to $RESULT_DIR/"
echo ""
echo "Remember to stop eBPF scheduler when done:"
echo "  sudo schedcp-cli stop"
echo "========================================"
```

---

## How to Run

```bash
# Step 1: 纯 GPU 测试 (baseline, 确定 OOM 临界点)
bash run_no_uvm.sh

# Step 2: UVM baseline 测试 (默认调度器)
bash run_uvm_baseline.sh

# Step 3: 启动 eBPF 调度器，然后运行 UVM 测试
sudo schedcp-cli run scx_simple
bash run_uvm_ebpf.sh
sudo schedcp-cli stop
```

---

## Expected Figures

### Figure 1: Performance vs Oversubscription Ratio

**Purpose**: 展示 UVM 如何突破 GPU 显存限制

```
Epoch Time (s)
    │
100 │                              ╭── UVM
    │                         ╭────╯
 50 │                    ╭────╯
    │               ╭────╯
 10 │    ────────────╯ (OOM without UVM)
  1 │────────────
    └─────────────────────────────────────
       0.7×  1.0×  1.5×  2.0×  2.5×
              Oversubscription Ratio
```

**Data needed**: `result_no_uvm/` + `result_uvm_baseline/`

---

### Figure 2: Scheduler Impact Comparison

**Purpose**: 展示 eBPF 调度器的加速效果

```
Epoch Time (s)
    │
150 │  ████
    │  ████  ████
100 │  ████  ████  ████
    │  ████  ████  ████
 50 │  ░░░░  ████  ████  ████
    │  ░░░░  ░░░░  ████  ████
  0 │  ░░░░  ░░░░  ░░░░  ░░░░
    └─────────────────────────
       1.0×  1.44× 1.72× 2.17×
       (7M)  (10M) (12M) (15M)

    ████ UVM Baseline   ░░░░ UVM + eBPF
```

**Data needed**: `result_uvm_baseline/` + `result_uvm_ebpf/`

---

## Data Collection Checklist

| Nodes | No UVM | UVM Baseline | UVM + eBPF |
|-------|--------|--------------|------------|
| 5M | ⬜ | ⬜ | ⬜ |
| 7M | ⬜ (OOM?) | ⬜ | ⬜ |
| 8M | ⬜ (OOM?) | ⬜ | ⬜ |
| 10M | ⬜ (OOM) | ⬜ | ⬜ |
| 12M | ⬜ (OOM) | ⬜ | ⬜ |
| 15M | ⬜ (OOM) | ⬜ | ⬜ |

---

## Notes

- 所有 UVM 测试需要设置 `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1`
- cuBLAS 可能在极端 oversubscription (>2×) 时失败
- 建议每个配置运行 3 次取平均值
- 监控 `nvidia-smi` 观察实时 GPU 内存和 PCIe 带宽

---

## Allocator Overhead Analysis

### Three Allocators Comparison

| Allocator | Command | Description |
|-----------|---------|-------------|
| PyTorch Default | (none) | Built-in caching allocator |
| Custom GPU | `--use_gpu_allocator` | cudaMalloc (no caching) |
| UVM | `--use_uvm` | cudaMallocManaged |

### Test Script: Allocator Comparison

```bash
#!/bin/bash
# run_allocator_comparison.sh - 对比三种 allocator 的性能

RESULT_DIR="result_allocator_compare"
mkdir -p $RESULT_DIR

NODES=5000000

echo "=== Test 1: PyTorch Default Allocator ==="
uv run python benchmark_gnn_uvm.py --dataset random --nodes $NODES \
    --edges_per_node 10 --features 128 --hidden 256 \
    --epochs 2 --warmup 1 --prop chunked \
    --report_json $RESULT_DIR/pytorch_default.json

echo "=== Test 2: Custom GPU Allocator (cudaMalloc) ==="
uv run python benchmark_gnn_uvm.py --dataset random --nodes $NODES \
    --edges_per_node 10 --features 128 --hidden 256 \
    --epochs 2 --warmup 1 --prop chunked --use_gpu_allocator \
    --report_json $RESULT_DIR/custom_gpu.json

echo "=== Test 3: UVM Allocator (cudaMallocManaged) ==="
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
    --dataset random --nodes $NODES \
    --edges_per_node 10 --features 128 --hidden 256 \
    --epochs 2 --warmup 1 --prop chunked --use_uvm \
    --report_json $RESULT_DIR/uvm.json

echo "=== Done ==="
```

### Results (5M nodes, no oversubscription)

| Allocator | Epoch Time | Relative | Source of Overhead |
|-----------|------------|----------|-------------------|
| PyTorch Default | **1.14s** | 1× | Baseline (caching allocator) |
| Custom GPU | **1.89s** | 1.66× | No memory pooling |
| UVM (no prefetch) | **34.23s** | **30×** | Lazy page migration |
| UVM (with prefetch) | **5.57s** | **4.9×** | Eager prefetch to GPU |

### Key Findings

1. **Custom allocator overhead**: ~66% slower than PyTorch default (no caching)
2. **Prefetch is critical**: Enabling `cudaMemPrefetchAsync` reduces UVM overhead from **30×** to **5×**
3. **UVM overhead breakdown (with prefetch)**:
   - ~1.7× from custom allocator (no memory pooling)
   - ~3× from prefetch synchronization and UVM page table management
4. **UVM overhead breakdown (without prefetch)**:
   - ~1.7× from custom allocator (no pooling)
   - **~18×** from lazy page fault handling on first access
5. **Lesson**: Always use `cudaMemPrefetchAsync` when data fits in GPU memory to avoid page fault overhead
