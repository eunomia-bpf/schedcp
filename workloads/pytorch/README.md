# PyTorch UVM Benchmark

This directory contains a PyTorch benchmark with CUDA Unified Virtual Memory (UVM) support, demonstrating how to train models larger than GPU memory capacity.

## Overview

This benchmark implements a custom CUDA allocator using `cudaMallocManaged` to enable PyTorch to train models that exceed GPU physical memory by transparently using CPU memory. The implementation includes:

- **Custom UVM Allocator** (`uvm_allocator.c`): C-based CUDA allocator with memory statistics
- **GNN Benchmark** (`benchmark_gnn_uvm.py`): Graph Neural Network training benchmark
- **Memory Statistics**: Real-time tracking of allocations, deallocations, and peak usage

## Key Results

### Chunked Index_Add Optimization (CRITICAL)

**The original naive GCN implementation created O(E·d) intermediate tensors, causing massive memory waste.** After implementing chunked index_add with custom autograd:

| Configuration | Original (Naive) | Chunked + UVM | Improvement |
|--------------|-----------------|---------------|-------------|
| **2M nodes, no UVM** | ✗ OOM (19.07 GiB) | ✓ **1.38 GB** | **93% reduction** |
| **2M nodes, with UVM** | ✓ 44.47 GB peak | Not needed | UVM unnecessary |
| **5M nodes, with UVM** | Not tested | ✓ **20.46 GB peak** | New capability |
| **10M nodes, with UVM** | OOM/cuBLAS fail | cuBLAS fail (40.90 GB) | Still limited |

### Critical Discovery

**The original implementation's memory explosion (44.47GB for 2M nodes) was NOT a UVM problem, but an algorithmic inefficiency:**

1. **Naive approach**: `out[src]` creates a **[20M edges × 256 hidden] = 19.07 GiB** tensor per forward/backward
2. **Chunked approach**: Processes edges in 2M chunks → only **[2M × 256] = 1.91 GiB** per chunk
3. **Result**: Memory usage dropped from 44.47GB → 1.38GB (**97% reduction**)

**This means:**
- 2M node graphs now train **without UVM** on a 32GB GPU
- 5M node graphs train successfully **with UVM** (20.46GB peak)
- The chunking algorithm is the real solution; UVM just extends the range further

## Architecture

### 1. UVM Allocator (`uvm_allocator.c`)

The custom allocator provides:

```c
// Core allocation functions
void* uvm_malloc(ssize_t size, int device, cudaStream_t stream)
void uvm_free(void* ptr, size_t size, int device, cudaStream_t stream)

// Statistics functions
size_t uvm_get_allocated_bytes(void)
size_t uvm_get_peak_allocated_bytes(void)
size_t uvm_get_num_allocs(void)
size_t uvm_get_num_frees(void)
void uvm_reset_peak_stats(void)
```

**Key Features:**
- Uses `cudaMallocManaged` for unified memory allocation
- Thread-safe statistics using atomic operations
- Prefetches memory to GPU with `cudaMemPrefetchAsync`
- Logs large allocations (>100MB) for debugging

**Memory Tracking:**
- Maintains `total_allocated`, `peak_allocated`, `num_allocs`, `num_frees`
- Uses atomic operations for thread safety
- Provides real-time visibility into allocation patterns

### 2. GNN Benchmark (`benchmark_gnn_uvm.py`)

A Graph Neural Network (GCN) training benchmark that:

- Generates random graphs with configurable size
- Implements simplified 2-layer GCN without external dependencies
- Tests memory limits with/without UVM
- Reports detailed memory statistics

**Key Components:**
```python
class SimpleGCNLayer(nn.Module):
    """Simplified GCN layer using index_add_ for message passing"""

class GCN(nn.Module):
    """2-layer GCN model"""

def enable_uvm_allocator():
    """Load and enable UVM allocator via CUDAPluggableAllocator"""
```

## Installation

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install torch psutil

# Build UVM allocator
make
```

## Usage

### Basic Usage

```bash
# Test without UVM (expect OOM for large graphs)
python benchmark_gnn_uvm.py --nodes=2000000

# Test with UVM (expect success)
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 python benchmark_gnn_uvm.py --nodes=2000000 --use_uvm
```

### Configuration Options

```
--nodes=N              Number of nodes (default: 10M)
--edges_per_node=N     Average edges per node (default: 10)
--features=N           Feature dimension (default: 128)
--hidden=N             Hidden dimension (default: 256)
--epochs=N             Number of epochs (default: 3)
--use_uvm              Enable UVM allocator
```

### Example Tests (Chunked Implementation)

```bash
# Small graph (1M nodes - works without UVM)
python benchmark_gnn_uvm.py --nodes=1000000 --epochs=2
# Expected: ~0.7GB GPU, completes in ~0.3s

# Medium graph (2M nodes - now works WITHOUT UVM!)
python benchmark_gnn_uvm.py --nodes=2000000 --epochs=2
# Expected: ~1.4GB GPU, completes in ~0.5s
# Previously required UVM + 44.47GB peak!

# Large graph (5M nodes - works WITH UVM)
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 python benchmark_gnn_uvm.py --nodes=5000000 --epochs=2 --use_uvm
# Expected: 20.46GB peak (with CPU offload), completes in ~4s

# Very large graph (10M nodes - hits cuBLAS limit)
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 python benchmark_gnn_uvm.py --nodes=10000000 --epochs=2 --use_uvm
# Expected: CUBLAS_STATUS_ALLOC_FAILED at 40.90GB
# Limitation: cuBLAS internal allocations conflict with UVM
```

### Benchmark Results Table

| Nodes | Edges | Without UVM | With UVM | Peak Memory | Status |
|-------|-------|-------------|----------|-------------|--------|
| 1M | 10M | 0.7 GB | - | 0.7 GB | ✓ Fast |
| 2M | 20M | **1.4 GB** | Not needed | 1.4 GB | ✓ **UVM unnecessary** |
| 5M | 50M | OOM (9.54 GB) | 3.4 GB | **20.46 GB** | ✓ **UVM enables** |
| 10M | 100M | OOM | cuBLAS fail | 40.90 GB | ✗ cuBLAS limit |

## Implementation Details

### Chunked Index_Add Algorithm

The key optimization is a custom autograd function that processes graph edges in chunks:

```python
class ChunkedIndexAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, src_indices, dst_indices, chunk_size):
        out = x.new_zeros((num_nodes, num_features))

        # Process edges in chunks instead of all at once
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            src_chunk = src_indices[start:end]
            dst_chunk = dst_indices[start:end]

            # Only [chunk_size, F] instead of [E, F]
            msg_chunk = x[src_chunk]
            out.index_add_(0, dst_chunk, msg_chunk)

        return out
```

**Memory Complexity:**
- **Naive**: O(E·d) - creates [E, hidden] tensor = 19.07 GB for 20M edges
- **Chunked**: O(N·d + chunk_size·d) - creates [2M, hidden] chunks = 1.91 GB
- **Reduction**: ~93% for typical graphs

### Memory Allocation Pattern Comparison

**Original Naive Implementation (2M nodes):**
```
[UVM] Alloc #13: 20.48 GB - Temporary buffer (index_add_ creates [E, hidden])
[UVM] Alloc #70: 20.48 GB - Backward pass (same issue)
Peak allocated:  44.47 GB
```

**Chunked Implementation (2M nodes, no UVM needed):**
```
GPU Memory: 1.38 GB total
  - Graph data: 1.02 GB (edges + features)
  - Model: 0.36 GB (weights + activations)
  - Chunks: max 1.91 GB (reused across batches)
```

**Chunked Implementation (5M nodes, with UVM):**
```
[UVM] Alloc #9:  5.12 GB - Edge indices
[UVM] Alloc #13: 2.05 GB - Chunked messages (not 51.2GB!)
[UVM] Alloc #125: 5.12 GB - Backward activations
Peak allocated:  20.46 GB (vs 110+ GB without chunking)
```

### PyTorch CUDAPluggableAllocator Integration

The UVM allocator integrates with PyTorch via the `CUDAPluggableAllocator` API:

```python
# Load shared library
uvm_lib = ctypes.CDLL('uvm_allocator.so')

# Define function signatures for statistics
uvm_lib.uvm_get_allocated_bytes.restype = ctypes.c_size_t
uvm_lib.uvm_get_peak_allocated_bytes.restype = ctypes.c_size_t

# Register with PyTorch
allocator = torch.cuda.memory.CUDAPluggableAllocator(
    'uvm_allocator.so',
    'uvm_malloc',
    'uvm_free'
)
torch.cuda.memory.change_current_allocator(allocator)
```

**Important:** The allocator must be enabled **before** any CUDA operations, including `torch.cuda.get_device_name()`.

### Known Limitations

1. **CUDAPluggableAllocator doesn't support getDeviceStats**
   - `torch.cuda.memory_allocated()` will fail with UVM allocator
   - Workaround: Use custom statistics functions via ctypes
   - See: [PyTorch Issue #133281](https://github.com/pytorch/pytorch/issues/133281)

2. **CUBLAS Internal Allocations**
   - cuBLAS creates internal handles with `cudaMalloc` (not managed by PyTorch)
   - When GPU memory is full with UVM allocations, cuBLAS fails with `CUBLAS_STATUS_ALLOC_FAILED`
   - Setting `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` helps but doesn't fully solve the issue
   - This limits maximum graph size (~2-3M nodes on 32GB GPU)

3. **Scaling Limits**
   - **1M nodes**: Works without UVM
   - **2M nodes**: Requires UVM, succeeds with `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1`
   - **3M nodes**: Fails due to cuBLAS allocation conflicts
   - **10M nodes**: Requires more sophisticated memory management

## Technical Background

### CUDA Unified Virtual Memory (UVM)

UVM provides a single unified memory space accessible from both CPU and GPU. Key characteristics:

- **Automatic Migration**: CUDA driver automatically migrates pages between CPU and GPU
- **Oversubscription**: Total allocated memory can exceed GPU physical memory
- **Page Faults**: GPU page faults trigger migration from CPU to GPU
- **Performance**: Slower than explicit memory management, but enables larger models

### Environment Variables

- `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1`: Forces driver to prefer GPU for physical storage
  - Required for compatibility with some CUDA libraries (cuBLAS)
  - Limits oversubscription capability
  - See: [CUDA Runtime API Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

## Future Work

Potential improvements to explore:

1. **Hybrid Allocator**: Reserve some GPU memory for cuBLAS, use UVM for tensors
2. **Manual Prefetching**: Use `cudaMemPrefetchAsync` strategically to reduce page faults
3. **Memory-Efficient GCN**: Implement scatter/gather operations with lower memory footprint
4. **Multi-GPU UVM**: Explore UVM with multiple GPUs (requires peer-to-peer compatibility)
5. **Alternative Libraries**: Test with PyG (PyTorch Geometric) or DGL for comparison

## References

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch CUDAPluggableAllocator](https://pytorch.org/docs/stable/torch_cuda_memory.html)
- [NVIDIA Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
- [cuBLAS Library User Guide](https://docs.nvidia.com/cuda/cublas/index.html)

## Files

- `uvm_allocator.c` - Custom CUDA UVM allocator with statistics
- `uvm_allocator.so` - Compiled shared library
- `benchmark_gnn_uvm.py` - GNN training benchmark
- `Makefile` - Build configuration
- `test_pytorch.py` - PyTorch installation verification
- `.gitignore` - Git ignore patterns

## Development Notes

### Build System

The Makefile compiles the UVM allocator:

```makefile
CUDA_PATH ?= /usr/local/cuda
CC = gcc
CFLAGS = -shared -fPIC -O3
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

uvm_allocator.so: uvm_allocator.c
    $(CC) $(CFLAGS) $(INCLUDES) uvm_allocator.c $(LDFLAGS) -o uvm_allocator.so
```

### Debugging Tips

1. **Enable verbose logging**: The allocator logs allocations >100MB to stderr
2. **Check memory stats**: Use `uvm_get_allocated_bytes()` to track memory usage
3. **Monitor with nvidia-smi**: Watch GPU memory usage in real-time
4. **Profile with nsight**: Use Nsight Systems to analyze UVM page faults

### Testing Strategy

Our testing methodology:

1. **Baseline**: Test without UVM to establish OOM threshold
2. **UVM Enabled**: Test same configuration with UVM
3. **Scale Up**: Incrementally increase graph size to find new limits
4. **Analyze Logs**: Review allocation patterns to understand bottlenecks

## Related Work

- **llama.cpp**: Uses UVM for large language model inference on GH200 systems ([Issue #5026](https://github.com/ggml-org/llama.cpp/issues/5026))
- **PyTorch Request**: Discussion about native UVM support ([Issue #124296](https://github.com/pytorch/pytorch/issues/124296))
- **TorchRec**: Feature request for UVM embedding support ([Issue #125](https://github.com/meta-pytorch/torchrec/issues/125))

##

time CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
        --dataset random --nodes 8000000 \
        --edges_per_node 10 --features 128 --hidden 256 \
        --epochs 50 --warmup 1 --prop chunked --use_uvm 
        --report_json multi/100.json
