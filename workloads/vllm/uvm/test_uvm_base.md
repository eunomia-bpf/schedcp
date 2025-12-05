vllm is also working, here we mainly test for KV cache offloading:

For a 30b FP8 MoE model (30GB model size) on 5090 (32GB memory), default setup will oom (without any offload)

Offload 8GB to CPU can work:
---------------Time to First Token----------------
Mean TTFT (ms):                          8387.80   
Median TTFT (ms):                        9066.78   
P99 TTFT (ms):                           14937.16  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          324.13    
Median TPOT (ms):                        215.65    
P99 TPOT (ms):                           1288.01 

UVM baseline:
---------------Time to First Token----------------
Mean TTFT (ms):                          9642.27   
Median TTFT (ms):                        10330.52  
P99 TTFT (ms):                           16549.02  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          374.23    
Median TPOT (ms):                        270.06    
P99 TPOT (ms):                           1288.79   

UVM with simple sequencial prefetch policy:
---------------Time to First Token----------------
Mean TTFT (ms):                          5042.22   
Median TTFT (ms):                        5419.23   
P99 TTFT (ms):                           7933.22   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          235.68    
Median TPOT (ms):                        207.69    
P99 TPOT (ms):                           583.74




## Run with offload:

uv run vllm serve Qwen/Qwen3-30B-A3B-FP8 --enforce-eager --cpu-offload-gb 8

```
$ uv run vllm bench serve --model  Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts  100 --dataset-path /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
INFO 11-26 00:26:04 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x775d79de8040>, seed=0, num_prompts=100, dataset_name='sharegpt', no_stream=False, dataset_path='/home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json', no_oversample=False, custom_output_len=256, custom_skip_chat_template=False, spec_bench_output_len=256, spec_bench_category=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, blazedit_min_distance=0.0, blazedit_max_distance=1.0, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, label=None, backend='openai', endpoint_type=None, base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', header=None, max_concurrency=None, model='Qwen/Qwen3-30B-A3B-FP8', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                   | 00:03 elapsed, 37:12:52 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
100%|█████████████████████████████████████████████████████████████████████████████████| 100/100 [01:55<00:00,  1.16s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  115.87    
Total input tokens:                      23260     
Total generated tokens:                  22061     
Request throughput (req/s):              0.86      
Output token throughput (tok/s):         190.40    
Peak output token throughput (tok/s):    504.00    
Peak concurrent requests:                100.00    
Total Token throughput (tok/s):          391.14    
---------------Time to First Token----------------
Mean TTFT (ms):                          8387.80   
Median TTFT (ms):                        9066.78   
P99 TTFT (ms):                           14937.16  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          324.13    
Median TPOT (ms):                        215.65    
P99 TPOT (ms):                           1288.01   
---------------Inter-token Latency----------------
Mean ITL (ms):                           191.40    
Median ITL (ms):                         180.39    
P99 ITL (ms):                            1299.17   
==================================================
```

## Run with baseline uvm:

~/workspace/vllm$ VLLM_USE_UVM=1  uv run vllm serve Qwen/Qwen3-30B-A3B-FP8 --enforce-eager

```
yunwei37@lab:~/workspace/gpu$ uv run vllm bench serve --model  Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts  100 --dataset-path /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
INFO 11-26 00:28:42 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7e5a51b1c180>, seed=0, num_prompts=100, dataset_name='sharegpt', no_stream=False, dataset_path='/home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json', no_oversample=False, custom_output_len=256, custom_skip_chat_template=False, spec_bench_output_len=256, spec_bench_category=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, blazedit_min_distance=0.0, blazedit_max_distance=1.0, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, label=None, backend='openai', endpoint_type=None, base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', header=None, max_concurrency=None, model='Qwen/Qwen3-30B-A3B-FP8', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                   | 00:04 elapsed, 58:05:23 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
100%|█████████████████████████████████████████████████████████████████████████████████| 100/100 [02:27<00:00,  1.48s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  147.50    
Total input tokens:                      23260     
Total generated tokens:                  22061     
Request throughput (req/s):              0.68      
Output token throughput (tok/s):         149.56    
Peak output token throughput (tok/s):    542.00    
Peak concurrent requests:                100.00    
Total Token throughput (tok/s):          307.26    
---------------Time to First Token----------------
Mean TTFT (ms):                          9642.27   
Median TTFT (ms):                        10330.52  
P99 TTFT (ms):                           16549.02  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          374.23    
Median TPOT (ms):                        270.06    
P99 TPOT (ms):                           1288.79   
---------------Inter-token Latency----------------
Mean ITL (ms):                           247.34    
Median ITL (ms):                         214.44    
P99 ITL (ms):                            1077.62   
==================================================
```

## run with improved UVM

```
$ uv run vllm bench serve --model  Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts  100 --dataset-path /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
INFO 11-26 00:41:44 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7a19b6538040>, seed=0, num_prompts=100, dataset_name='sharegpt', no_stream=False, dataset_path='/home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json', no_oversample=False, custom_output_len=256, custom_skip_chat_template=False, spec_bench_output_len=256, spec_bench_category=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, blazedit_min_distance=0.0, blazedit_max_distance=1.0, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, label=None, backend='openai', endpoint_type=None, base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', header=None, max_concurrency=None, model='Qwen/Qwen3-30B-A3B-FP8', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                   | 00:04 elapsed, 54:23:19 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
100%|█████████████████████████████████████████████████████████████████████████████████| 100/100 [02:00<00:00,  1.20s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  120.36    
Total input tokens:                      23260     
Total generated tokens:                  22061     
Request throughput (req/s):              0.83      
Output token throughput (tok/s):         183.28    
Peak output token throughput (tok/s):    623.00    
Peak concurrent requests:                100.00    
Total Token throughput (tok/s):          376.53    
---------------Time to First Token----------------
Mean TTFT (ms):                          5042.22   
Median TTFT (ms):                        5419.23   
P99 TTFT (ms):                           7933.22   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          235.68    
Median TPOT (ms):                        207.69    
P99 TPOT (ms):                           583.74    
---------------Inter-token Latency----------------
Mean ITL (ms):                           194.62    
Median ITL (ms):                         173.72    
P99 ITL (ms):                            467.74    
==================================================
```

## run with lmcache

LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=8.0 VLLM_WORKER_MULTIPROC_METHOD=spawn .venv/bin/vllm serve Qwen/Qwen3-30B-A3B-FP8   --max-model-len 2048   --enforce-eager   --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'  --cpu-offload-gb 8
```
uv run vllm bench serve --model  Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts  100 --dataset-path /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
INFO 12-03 23:28:11 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x751a9bbf2660>, seed=0, num_prompts=100, dataset_name='sharegpt', no_stream=False, dataset_path='/home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json', no_oversample=False, custom_output_len=256, custom_skip_chat_template=False, spec_bench_output_len=256, spec_bench_category=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, blazedit_min_distance=0.0, blazedit_max_distance=1.0, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, label=None, backend='openai', endpoint_type=None, base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', header=None, max_concurrency=None, model='Qwen/Qwen3-30B-A3B-FP8', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |███████████████▍                                                                    | 02:47 elapsed, 12:26 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
100%|███████████████████████████████████████████████████████████████████████████████| 100/100 [01:19<00:00,  1.26it/s]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  79.30     
Total input tokens:                      23260     
Total generated tokens:                  22061     
Request throughput (req/s):              1.26      
Output token throughput (tok/s):         278.21    
Peak output token throughput (tok/s):    744.00    
Peak concurrent requests:                100.00    
Total Token throughput (tok/s):          571.54    
---------------Time to First Token----------------
Mean TTFT (ms):                          5401.71   
Median TTFT (ms):                        4961.94   
P99 TTFT (ms):                           10072.64  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          222.24    
Median TPOT (ms):                        149.84    
P99 TPOT (ms):                           817.94    
---------------Inter-token Latency----------------
Mean ITL (ms):                           131.20    
Median ITL (ms):                         122.30    
P99 ITL (ms):                            833.54    
==================================================
```
