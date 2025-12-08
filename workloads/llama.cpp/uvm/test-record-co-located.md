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



## Co-located

