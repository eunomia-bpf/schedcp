# UVM Benchmark Results (Without cudaMemPrefetchAsync)

This directory contains benchmark results for GCN training **without** `cudaMemPrefetchAsync` enabled in the custom UVM allocator.

## Configuration

- **GPU**: NVIDIA GeForce RTX 5090 (32GB)
- **Prefetch**: Disabled (lazy page migration)
- **Workload**: GCN training on random graphs

## Directory Structure

```
without-user-prefetch/
├── result_no_uvm1/      # PyTorch default allocator (no UVM)
├── result_uvm_baseline1/ # UVM with default Linux scheduler
├── result_uvm_ebpf1/    # UVM with eBPF scheduler optimization
├── visualize.py         # Visualization script
└── benchmark_results.png # Generated figure
```

## Results Summary

| Nodes | No UVM | UVM Baseline | UVM + eBPF | eBPF Speedup |
|-------|--------|--------------|------------|--------------|
| 1M | 0.22s | - | - | - |
| 3M | 0.68s | - | - | - |
| 5M | 1.14s | 34.23s | 12.76s | **2.68×** |
| 7M | 1.79s | 48.28s | 17.81s | **2.71×** |
| 8M | OOM | 55.36s | 20.51s | **2.70×** |
| 10M | OOM | 70.06s | 26.47s | **2.65×** |
| 12M | OOM | 93.71s | 39.74s | **2.36×** |
| 15M | OOM | 292.77s | 168.73s | **1.74×** |

## Key Findings

1. **UVM without prefetch is extremely slow**: 30× overhead vs native GPU allocation
2. **eBPF scheduler provides 2.4-2.7× speedup**: Optimizes page fault handling
3. **Benefit decreases at extreme oversubscription**: 1.74× at 15M nodes due to thrashing

## Usage

```bash
# Generate visualization
python visualize.py
```

## See Also

- `../with-user-prefetch/` - Results with cudaMemPrefetchAsync enabled
- `../UVM_EVALUATION.md` - Complete OSDI-style evaluation document
