# LMCache 在 RTX 5090 (Blackwell) 上的安装与运行指南

由于 RTX 5090 使用 Blackwell 架构 (sm_120)，预编译的 lmcache 包不支持该架构，需要从源码编译。

## 环境要求

- Python 3.12
- CUDA 12.x
- RTX 5090 (Blackwell, compute capability 12.0)

## 安装步骤

### 1. 创建干净的虚拟环境

```bash
cd /home/yunwei37/workspace/gpu/LMCache
uv venv --python 3.12
```

### 2. 安装 vLLM 和 PyTorch

```bash
uv pip install vllm torch --python .venv/bin/python
```

### 3. 从源码编译 LMCache（指定 CUDA 架构）

```bash
TORCH_CUDA_ARCH_LIST="12.0" uv pip install -e . --no-build-isolation --python .venv/bin/python
```

## 运行 vLLM + LMCache

### 启动服务器（CPU KV Cache 卸载模式）

```bash
cd /home/yunwei37/workspace/gpu/LMCache

LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=5.0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/vllm serve Qwen/Qwen3-0.6B \
  --max-model-len 2048 \
  --enforce-eager \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### 环境变量说明

| 变量 | 说明 |
|------|------|
| `LMCACHE_LOCAL_CPU=True` | 启用 CPU KV Cache 存储 |
| `LMCACHE_MAX_LOCAL_CPU_SIZE=5.0` | CPU 缓存最大大小（GB） |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | 避免 CUDA fork 错误 |

### KV Transfer 配置说明

| 参数 | 说明 |
|------|------|
| `kv_connector` | 使用 `LMCacheConnectorV1` 连接器 |
| `kv_role` | `kv_both` 表示同时作为 KV cache 的生产者和消费者 |

## 测试请求

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## 预期输出

成功运行时，日志中会显示 LMCache 的 KV cache 存储信息：

```
LMCache INFO: Reqid: chatcmpl-xxx, Total tokens 10, LMCache hit tokens: 0
LMCache INFO: Storing KV cache for 10 out of 10 tokens
LMCache INFO: Stored 10 out of total 10 tokens. size: 0.0011 gb, throughput: 1.8919 GB/s
```

## 常见问题

### 1. CUDA kernel not available 错误

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

**原因**: 预编译的 lmcache 不支持 RTX 5090 (sm_120)
**解决**: 必须从源码编译，使用 `TORCH_CUDA_ARCH_LIST="12.0"`

### 2. GPU 内存不足

```
Free memory on device less than desired GPU memory utilization (0.9)
```

**解决**: 添加 `--gpu-memory-utilization 0.8` 或更低的值

### 3. CUDA fork 错误

```
Cannot re-initialize CUDA in forked subprocess
```

**解决**: 设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`
