# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import sys
# import os

# # 获取当前脚本所在的目录，并找到新项目verl的根目录
# # 假设你的脚本在 /data2/private/linzejin/verl/some/sub/folder/run.py
# # 我们需要把 /data2/private/linzejin/verl/ 添加到 sys.path
# # new_verl_project_root = "/data2/private/linzejin/verl" # 直接硬编码路径

# # 确保这个路径在 sys.path 的最前面，这样Python会优先在这里查找
# if new_verl_project_root not in sys.path:
#     sys.path.insert(0, new_verl_project_root)


import concurrent.futures
from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.trainer.ppo.reward import get_custom_reward_fn


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None,reward_dim = 2, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.reward_dim = reward_dim  # The dimension of the reward tensor
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto,config: dict={}, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        batch_size, seq_len = data.batch["responses"].shape
        # 假设 reward 是一个 PyTorch 张量，我们可以从中获取 reward_dim
        # reward_dim = 2 # TODO To be determined later
        if self.reward_dim >= 2:
            reward_tensor = torch.zeros(batch_size, seq_len, self.reward_dim, dtype=torch.float32)
        
        extra_info = {}
        reward_extra_info = defaultdict(list)
        reward_extra_info["_process_"] = defaultdict(int)
        

        already_print_data_sources = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=config.get("workers", 80)) as executor:
            # 提交所有任务，并创建一个future到索引的映射
            # 我们传递索引 i 和 data_item
            futures = {
                executor.submit(self._process_single_item, i, data[i], config): i
                for i in range(len(data))
            }

            # 当任务完成时，处理它们的结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    # 获取结果，如果子进程出错，这里会重新抛出异常
                    (i, score, valid_response_length, data_source,
                     prompt_str, response_str, ground_truth) = future.result()

                    # 检查子进程是否返回了一个错误
                    if isinstance(score, Exception):
                        print(f"处理数据项 {i} 时出错: {score, prompt_str, response_str}")
                        continue
                        
                    # --- 在主进程中安全地聚合结果 ---
                    if isinstance(score, dict):
                        reward = score.pop("score")
                        try:
                            process_keys = score.pop("process_keys", [])
                        except KeyError:
                            process_keys = []
                        reward_extra_info["_process_"]["process_keys"]=process_keys
                        # 存储所有额外信息
                        for key, value in score.items():
                            # 注意：由于结果是无序返回的，直接append会打乱顺序
                            # 如果需要保持顺序，需要更复杂的处理，但对于统计通常没问题
                            if key in process_keys:
                                reward_extra_info["_process_"][key] += value
                            # reward_extra_info[key].append(value)
                        extra_info[i] = score
                    else:
                        reward = score

                    # 在正确的位置更新 reward_tensor
                    if valid_response_length > 0:
                        if self.reward_dim == 1:
                            reward_tensor[i, valid_response_length - 1] = reward
                        else:
                            reward_tensor[i, valid_response_length - 1, :] = torch.tensor(reward)

                    # --- 在主进程中安全地处理打印逻辑 ---
                    if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0

                    if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        if isinstance(score, dict):
                            for key, value in score.items():
                                print(f"[{key}]", value)
                        else:
                            print("[score]", score)
                        print("-" * (25 + len(str(i)) + len(data_source)))

                except Exception as e:
                    # 捕获任务执行或结果检索中可能发生的任何其他错误
                    index = futures[future]
                    print(f"处理数据项 {index} 失败，出现严重错误: {e}")
        # 更新 reward_extra_info
        if extra_info:
            for idx in range(len(data)):
                if idx in extra_info:
                    for key, value in extra_info[idx].items():
                        reward_extra_info[key].append(value)
                # 在_dump_generation的设计中并没有考虑那些不存在的input


                # else:
                #     # 为失败的任务添加占位符，保持长度一致
                #     sample_keys = next(iter(extra_info.values())).keys()
                #     for key in sample_keys:
                #         reward_extra_info[key].append(None)  # 或其他默认值
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _process_single_item(self, i, data_item, config):
        """
        处理单个数据项的函数，这是将在子进程中运行的核心逻辑。
        注意：这个函数不应修改任何共享状态，而是返回所有必要的信息。
        """
        try:
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"] #是否必要存疑？
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            extra_info.update(config.get("config", {}))

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if i < self.num_examine:
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
            
            # 返回原始索引、计算结果和用于打印的信息
            return (i, score, valid_response_length, data_source, 
                    prompt_str, response_str, ground_truth)
        except Exception as e:
            # 在子进程中捕获错误并返回，以便主进程可以处理它
            print(f"prompt:{prompt_str}\n response:{response_str}\n")
            print(f"Exception in _process_single_item: {e}")
            return (i, e, None, None, prompt_str, response_str, None)

