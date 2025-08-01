python -m vllm.entrypoints.openai.api_server \
  --model /mnt/workspace/linzejin/models/Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.3 \
  --served-model-name Qwen3-14B  \
  --port 10000
