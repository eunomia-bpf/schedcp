# Makefile for PyTorch Custom Allocators

CUDA_PATH ?= /usr/local/cuda
CC = gcc
CFLAGS = -shared -fPIC -O3
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

UVM_TARGET = uvm_allocator.so
UVM_SOURCE = uvm_allocator.c

GPU_TARGET = gpu_allocator.so
GPU_SOURCE = gpu_allocator.c

.PHONY: all clean test help

all: $(UVM_TARGET) $(GPU_TARGET)

$(UVM_TARGET): $(UVM_SOURCE)
	$(CC) $(CFLAGS) $(INCLUDES) $(UVM_SOURCE) $(LDFLAGS) -o $(UVM_TARGET)
	@echo "Built $(UVM_TARGET) successfully"

$(GPU_TARGET): $(GPU_SOURCE)
	$(CC) $(CFLAGS) $(INCLUDES) $(GPU_SOURCE) $(LDFLAGS) -o $(GPU_TARGET)
	@echo "Built $(GPU_TARGET) successfully"

clean:
	rm -f $(UVM_TARGET) $(GPU_TARGET)
	@echo "Cleaned build artifacts"

test: $(UVM_TARGET)
	@echo "Running UVM allocator tests..."
	source .venv/bin/activate && python pytorch_uvm_allocator.py

help:
	@echo "PyTorch Custom Allocators Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all    - Build both UVM and GPU allocator shared libraries (default)"
	@echo "  clean  - Remove built artifacts"
	@echo "  test   - Build and run tests"
	@echo "  help   - Show this help message"
	@echo ""
	@echo "Output:"
	@echo "  uvm_allocator.so - Uses cudaMallocManaged (UVM)"
	@echo "  gpu_allocator.so - Uses cudaMalloc (standard GPU memory)"
