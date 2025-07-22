set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
LOCAL_DIR=/data/private/linzejin/proofrl/verl
MODEL_DIR=/data/private/linzejin/models

EPISODE=20
DATA_TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=2
MICRO_BATCH_SIZE=1
GPU_NUM=4

PROJECT_NAME='grpo_RLVR_MATH'
EXPERIMENT_NAME='q3_0.6b_default'


uv run --active python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/private/linzejin/proofrl/data/MATH/train.parquet \
    data.val_files=/data/private/linzejin/proofrl/data/MATH/test.parquet \
    data.train_batch_size=$DATA_TRAIN_BATCH_SIZE    \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="/data/private/linzejin/models/Qwen/Qwen3-0.6B-Base" \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir="$LOCAL_DIR/checkpoints/$trainer.project_name/$trainer.experiment_name" \
    trainer.test_freq=5 \
    trainer.total_epochs=$EPISODE   $@ 