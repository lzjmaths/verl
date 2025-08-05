set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_USE_MODELSCOPE=True
echo "start training"
DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

WORK_DIR='.'

LOCAL_DIR="${WORK_DIR}"
MODEL_DIR=/mnt/workspace/linzejin/models/Qwen
LOG_DIR="${WORK_DIR}/verl/logs"

K=3
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=$((1024 * $K))

if [ "$K" -gt 8 ]; then
  N=1
else
  N=4
fi
N=1

EPISODE=20
DATA_TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=2
MICRO_BATCH_SIZE=1
GPU_NUM=2

TASK="MATH500"

TRAIN_TASK="${TASK}"
if [ "$TASK" = "AMC" ]; then
  # 如果 TASK 的值是 "AMC"
  TEST_TASK="MATH500"
elif [ "$TASK" = "MATH500" ]; then
  # 如果 TASK 的值是 "MATH500"
  TEST_TASK="AMC"
else
  # 如果 TASK 是任何其他值，可以定义一个默认行为
  # 例如，直接将 TASK 的值赋给 TEST_TASK
  echo "警告: TASK 不是 AMC 或 MATH500，使用默认值。"
  TEST_TASK="$TASK"
fi
BACKBONE="Qwen2.5-Math-1.5B"
ADVANTAGE="grpo"

DATA_LOCAL_DIR="${WORK_DIR}/data"
BACKBONE_PATH="${MODEL_DIR}/${BACKBONE}"

MODEL="${TASK}-${BACKBONE}"

PROJECT_NAME="grpo_RLMR-${TASK}"
EXPERIMENT_NAME="rlmr@${K}k"
EX_NAME=""
LOG_NAME="${EXPERIMENT_NAME}-${MODEL}-${DATE}-${TIME_TAG}"
OUTPUT_DIR="checkpoints/${PROJECT_NAME}/${MODEL}/${DATE}/${EXPERIMENT_NAME}-${ADVANTAGE}-${TIME_TAG}${EX_NAME}"

REWARD_FUN="${WORK_DIR}/reward_functions/verify_reward_format.py"
TEST_FUN="${WORK_DIR}/reward_functions/math_reward.py"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=["$DATA_LOCAL_DIR/$TRAIN_TASK/train.parquet"] \
    data.val_files=["$DATA_LOCAL_DIR/$TEST_TASK/test.parquet"] \
    data.train_batch_size=$DATA_TRAIN_BATCH_SIZE    \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BACKBONE_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.optim.warmup_style='cosine' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.005 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.78  \
    actor_rollout_ref.rollout.val_kwargs.n=$N \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$LOG_NAME \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.rollout_data_dir="${LOG_DIR}/${LOG_NAME}/rollout" \
    trainer.validation_data_dir="${LOG_DIR}/${LOG_NAME}/validation" \
    trainer.save_freq=50 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.test_freq=5 \
    trainer.total_epochs=$EPISODE \
    custom_reward_function.workers=80 \
    custom_reward_function.reward_dim=2 \
    custom_reward_function.config.log_dir="${LOG_DIR}/${LOG_NAME}/reward" \
    custom_reward_function.path=${REWARD_FUN} \
    custom_test_function.path=${TEST_FUN}  $@ 


echo "Output directory: $OUTPUT_DIR"