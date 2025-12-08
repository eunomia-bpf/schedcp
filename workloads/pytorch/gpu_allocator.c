/*
 * CUDA GPU Allocator for PyTorch with Memory Statistics (No UVM)
 *
 * This is a custom CUDA allocator that uses standard cudaMalloc
 * (NOT cudaMallocManaged) for comparison with UVM allocator.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdatomic.h>

// Global memory statistics
static atomic_size_t total_allocated = 0;
static atomic_size_t peak_allocated = 0;
static atomic_size_t num_allocs = 0;
static atomic_size_t num_frees = 0;

// Allocate standard CUDA device memory (NOT UVM)
void* gpu_malloc(ssize_t size, int device, cudaStream_t stream) {
    void* ptr = NULL;
    cudaError_t err;

    // Use standard cudaMalloc (NOT cudaMallocManaged)
    err = cudaMalloc(&ptr, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU] cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        return NULL;
    }

    // Update statistics
    size_t current = atomic_fetch_add(&total_allocated, size) + size;
    size_t alloc_count = atomic_fetch_add(&num_allocs, 1) + 1;

    // Update peak if needed
    size_t peak = atomic_load(&peak_allocated);
    while (current > peak) {
        if (atomic_compare_exchange_weak(&peak_allocated, &peak, current)) {
            break;
        }
    }

    // Log large allocations
    if (size > 1000 * 1024 * 1024) { // > 1GB
        fprintf(stderr, "[GPU] Alloc #%zu: %.2f GB (total: %.2f GB, peak: %.2f GB)\n",
                alloc_count, size / 1e9, current / 1e9, atomic_load(&peak_allocated) / 1e9);
    }

    return ptr;
}

// Free CUDA device memory
void gpu_free(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (ptr != NULL) {
        cudaFree(ptr);

        // Update statistics
        atomic_fetch_sub(&total_allocated, size);
        atomic_fetch_add(&num_frees, 1);
    }
}

// Get current allocated bytes
size_t gpu_get_allocated_bytes(void) {
    return atomic_load(&total_allocated);
}

// Get peak allocated bytes
size_t gpu_get_peak_allocated_bytes(void) {
    return atomic_load(&peak_allocated);
}

// Get allocation count
size_t gpu_get_num_allocs(void) {
    return atomic_load(&num_allocs);
}

// Get free count
size_t gpu_get_num_frees(void) {
    return atomic_load(&num_frees);
}

// Reset peak statistics
void gpu_reset_peak_stats(void) {
    atomic_store(&peak_allocated, atomic_load(&total_allocated));
}
