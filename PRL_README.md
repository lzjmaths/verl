PRL是一个基于verl修改的针对证明题做训练的仓库。基本的修改为奖励函数的修改。
# 目前实现的功能有
- 补充了两个参数custom_reward_function和custom_test_function, 具体细节如下。
    - custom_reward_function:
        - path:null
        - name: for load the reward function in the file, default is 'compute_score'
        - workers: workers for computing reward, default is 1
        - reward_dim: You need to determine it in the reward function, default is 2.
        - config.model_name: name for the model, default is 'Qwen3-14B'
        - config.api_url: url link for api, default is null. We do not use it in our reward_function.
        - config.reviews: reviews number for review model. 
        - config.log_dir: location where you save the reward log, default is null.
    - custom_test_function:
        - path:null
        - name: for load the reward function in the file, default is 'compute_score'
        - reward_dim: You need to determine it in the reward function, default is 1. 目前还没有实现基于test_function的多维奖励。
    - 你可以在verl/trainer/config/ppo_trainer.yaml查看具体参数细节。
- 所有PRL用到的奖励函数放在reward_functions/里
- reward function的设计要求，参考样例reward_functions/verify_multi.
    - 首先compute_score的设计需要传入(data_source:str, solution_str:str, ground_truth, extra_info=None).
    - 目前extra_info里会包括custom_reward_function.config (if exists)里的所有键值对。以及data里的extra_info部分。如果用这里的data/preprocess.py预处理会包含"question"键。我使用extra_info.update合并两个字典，所以两个字典里的键请不要重复。
    - 返回的reward可以是一个=reward_dim的tuple，也可以是一个字典，包括"score"的键，"score"键的值必须是一个等于reward_dim的tuple
    - 对于统计量的处理，目前支持["TP","TN","FP","FN","true_score", "general_score", "accuracy", "precision", "recall", "specificity", "box_num"]的统计量计算。注意，想要调用这些统计量你需要让返回的reward是一个字典，在包括以上键的同时，你需要增加reward[\'process_keys\']一列声明你需要的统计量。
    - 你可以在reward[\'process_keys\']里添加其他统计量，这样会计算整个batch里该统计量的和并通过reward_extra_info_dict传到主函数里。同时你应该在verl/trainer/ppo/ray_trainer.py的RayPPOTrainer._process_reward_extra_info_dict()中声明这个统计量的和的计算方式。该函数输入是process_keys里的的键在一个batch的和，输出是你想要update到平台的统计量。
    - 其他实现细节，包括并行处理的细节详见verl/workers/reward_manager/naive.py和verl/trainer/ppo/ray_trainer.py
- 关于多维奖励的处理，详见verl/trainer/ppo/core_algos.py中的compute_grpo_outcome_advantage和compute_policy_loss
- 目前有些统计量例如advantage，pg_clip_ratio没有处理好多维的情况，如想修复可以查看verl/workers/actor/dp_actor.py对actor板块的上传和verl/trainer/ppo/metric_utils.py关于各个统计量上传情况的细节。

# 参考样例
- 所有PRL用到的奖励函数放在reward_functions/里
- 所有PRL用到的数据都放在data/里，你需要运行命令行
```python
python data/preprocess.py --dataset_dir=data/MATH500
```
执行数据预处理操作
- 目前可以利用的脚本只有verl/RLMR/MATH500/verify-format/run_Qwen2.5-Math-1.5B_MATH500.sh。你需要更改WORK_DIR至你的verl所在的位置，以及修改MODEL_DIR至你的模型所在位置。