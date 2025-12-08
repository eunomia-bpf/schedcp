# UVM Memory Management for Large MoE Model Inference: An Experimental Analysis

## Key Findings

We evaluate five memory management strategies for running a 120B parameter Mixture-of-Experts (MoE) model on a single NVIDIA RTX 5090 GPU (32GB VRAM). The model requires 59.02 GiB of memory, significantly exceeding available GPU memory, necessitating CPU-GPU memory tiering.

### Experimental Setup

- **Hardware**: NVIDIA GeForce RTX 5090 (32GB VRAM, compute capability 12.0, VMM: yes)
- **Model**: gpt-oss-120B MXFP4 MoE (59.02 GiB, 116.83B parameters)
- **Workload**: llama-bench with pp512 (prefill 512 tokens) and tg128 (decode 128 tokens)
- **Software**: llama.cpp build 10e97801

### Results Summary

| Configuration | Prefill (pp512) | Decode (tg128) | Prefill vs Baseline | Decode vs Baseline |
|---------------|-----------------|----------------|---------------------|-------------------|
| ncmoe=32 (CPU offload) | **260.14 t/s** | 18.18 t/s | +9.1% | +136% |
| ncmoe=64 (CPU offload) | 245.63 t/s | 16.34 t/s | +3.0% | +112% |
| UVM only (baseline) | 238.48 t/s | 7.72 t/s | - | - |
| UVM + eBPF prefetch | 229.67 t/s | **86.89 t/s** | -3.7% | **+1025%** |
| UVM + user hint | 144.00 t/s | 49.31 t/s | -39.6% | +539% |

### Key Observations

1. **eBPF-based prefetching achieves 11.3x decode speedup**: The UVM + eBPF configuration delivers the best decode throughput (86.89 t/s) while maintaining competitive prefill performance (229.67 t/s). Compared to CPU offload approaches, eBPF prefetching achieves **4.8x faster decode than ncmoe=32** (86.89 vs 18.18 t/s) and **5.3x faster than ncmoe=64** (86.89 vs 16.34 t/s). This demonstrates that kernel-level prefetching can effectively hide memory transfer latency during autoregressive decoding.

2. **CPU offload (ncmoe) optimizes prefill at decode cost**: The ncmoe=32 configuration achieves the highest prefill throughput (260.14 t/s, +9.1% over baseline) by explicitly offloading MoE experts to CPU. However, decode performance is limited (18.18 t/s) due to the overhead of CPU-GPU data transfers for each token.

3. **Naive user hints degrade performance**: Using `cudaMemAdviseSetPreferredLocation` to GPU causes severe prefill degradation (-39.6%) as pages must be migrated before computation. While decode improves (49.31 t/s), it still underperforms eBPF prefetching by 43%.

4. **Phase-dependent optimization is critical**: Prefill (batch processing) and decode (sequential, memory-bound) have fundamentally different memory access patterns. eBPF prefetching excels at decode by predicting and prefetching MoE expert weights, while CPU offload excels at prefill by avoiding page faults entirely.

### Implications

For production MoE inference systems with memory oversubscription:
- **Latency-sensitive decode**: Use eBPF-based prefetching (11.3x improvement)
- **Throughput-oriented prefill**: Use explicit CPU offload with ncmoe (9.1% improvement)
- **Avoid naive UVM hints**: User-space memory hints without proper prefetching cause significant performance regression

---

# Manual Test

Test command:

yunwei37@lab:~/workspace/gpu/schedcp/workloads/llama.cpp$ uv run vllm bench serve --model  Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts  100 --dataset-path /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json  --base-url http://127.0.0.1:8013  --max-concurrency=1

## Run

CPU offload:

yunwei37@lab:~/workspace/gpu/schedcp/workloads/llama.cpp$ /home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/llama-server --gpt-oss-120b-default -ncmoe 64 -c 65536

UVM baseline:

GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 /home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/llama-server --gpt-oss-120b-default -c 65536

## Test script

```bash
python uvm/test_uvm_baselines.py --bench-args "--model Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts 1 --dataset-path /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --sharegpt-output-len 512 --seed 42 --request-rate 1"


python uvm/test_uvm_baselines.py --baselines naive_uvm --bench-args "--model Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts 1 --dataset-path /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --sharegpt-output-len 512 --seed 42 --request-rate 1"
```

## Test with llama bench under 5090 platform

/home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/llama-server --gpt-oss-120b-default -ncmoe 64 -c 65536

GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 /home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/llama-server --gpt-oss-120b-default -c 65536

$ uv run vllm bench serve --model  Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts  100 --dataset-path /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json  --base-url http://127.0.0.1:8013  --max-concurrency=1

Need to build with no VMM support

$ GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 /home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/llama-server --gpt-oss-120b-default -c 65536

In vllm dir, run

uv run /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/llamacpp_openai_client.py

with UVM memory set to CPU first and unset it:

                // SetPreferredLocation(CPU): Pages stay in system RAM, fetched on demand
                advise_err = cudaMemAdvise(*ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

```

Running llama-bench with UVM enabled...
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 build/bin/llama-bench \
        -m /home/yunwei37/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf \
        2>&1 | tee results/gpt-oss-120b-uvm-bench.log
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           pp512 |        238.48 ± 1.43 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           tg128 |          7.72 ± 0.01 |

build: 10e97801 (7099)

Benchmark complete! Results saved to: results/gpt-oss-120b-uvm-bench.log
```

UVM set to GPU and other method does not work.

With UVM set to CPU first and then set to access by:

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           pp512 |        238.45 ± 1.47 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           tg128 |          7.70 ± 0.01 |

build: 10e97801 (7099)
```


Set CPU first then set to GPU first:

```
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           pp512 |        144.00 ± 1.18 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           tg128 |         49.31 ± 3.82 |
```


with ncmoe64

```
$ build/bin/llama-bench  -ncmoe 64       -m /home/yunwei37/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           pp512 |        245.63 ± 2.05 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           tg128 |         16.34 ± 0.03 |

build: 10e97801 (7099)
yunwei37@lab:~/workspace/gpu/schedcp/workloads/llama.cpp$ 
```

with ncmoe32

```
$ build/bin/llama-bench  -ncmoe 32       -m /home/yunwei37/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_g
pt-oss-120b-mxfp4-00001-of-00003.gguf 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           pp512 |        260.14 ± 2.32 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           tg128 |         18.18 ± 0.05 |

build: 10e97801 (7099)
```

With prefetching

```
 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 build/bin/llama-bench         -m /home/yunwei37/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf         2>&1 | tee results/gpt-oss-120b-uvm-bench.log
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           pp512 |        229.67 ± 1.35 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | CUDA       |  99 |           tg128 |         86.89 ± 5.22 |

build: 10e97801 (7099)
yunwei37@lab:~/workspace/gpu/schedcp/workloads
```
