# Mannual test

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