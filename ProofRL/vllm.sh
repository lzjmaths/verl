python3 -m vllm.entrypoints.openai.api_server \
  --model /data2/private/linzejin/models/Qwen/Qwen3-14B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --served-model-name Qwen3-14B \
  --port 10000