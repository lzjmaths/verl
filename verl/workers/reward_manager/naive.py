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

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
            # 提交所有任务，并创建一个future到索引的映射
            # 我们传递索引 i 和 data_item
            futures = {
                executor.submit(self._process_single_item, i, data[i]): i
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

                        # 存储所有额外信息
                        for key, value in score.items():
                            # 注意：由于结果是无序返回的，直接append会打乱顺序
                            # 如果需要保持顺序，需要更复杂的处理，但对于统计通常没问题
                            if key in ["TN","TP","FN","FP","true_score"]:
                                if isinstance(reward_extra_info[key],list):
                                    reward_extra_info[key] = 0
                                reward_extra_info[key] += value
                            else:
                                reward_extra_info[key].append(value)
                    else:
                        reward = score

                    # 在正确的位置更新 reward_tensor
                    if valid_response_length > 0:
                        reward_tensor[i, valid_response_length - 1] = reward

                    # --- 在主进程中安全地处理打印逻辑 ---
                    if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0

                    if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        print("end of test for 1 time")
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

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

        # for i in range(len(data)):
        #     data_item = data[i]  # DataProtoItem

        #     prompt_ids = data_item.batch["prompts"]

        #     prompt_length = prompt_ids.shape[-1]

        #     valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        #     valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        #     response_ids = data_item.batch["responses"]
        #     valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        #     valid_response_ids = response_ids[:valid_response_length]

        #     # decode
        #     prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        #     response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        #     ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        #     data_source = data_item.non_tensor_batch[self.reward_fn_key]
        #     extra_info = data_item.non_tensor_batch.get("extra_info", {})
        #     num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        #     extra_info["num_turns"] = num_turns

        #     score = self.compute_score(
        #         data_source=data_source,
        #         solution_str=response_str,
        #         ground_truth=ground_truth,
        #         extra_info=extra_info,
        #     )

        #     if isinstance(score, dict):
        #         reward = score["score"]
        #         # Store the information including original reward
        #         for key, value in score.items():
        #             reward_extra_info[key].append(value)
        #     else:
        #         reward = score

        #     reward_tensor[i, valid_response_length - 1] = reward

        #     if data_source not in already_print_data_sources:
        #         already_print_data_sources[data_source] = 0

        #     if already_print_data_sources[data_source] < self.num_examine:
        #         already_print_data_sources[data_source] += 1
        #         print("[prompt]", prompt_str)
        #         print("[response]", response_str)
        #         print("[ground_truth]", ground_truth)
        #         if isinstance(score, dict):
        #             for key, value in score.items():
        #                 print(f"[{key}]", value)
        #         else:
        #             print("[score]", score)

        # if return_dict:
        #     return {
        #         "reward_tensor": reward_tensor,
        #         "reward_extra_info": reward_extra_info,
        #     }
        # else:
        #     return reward_tensor

    def _process_single_item(self, i, data_item):
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

# class SimplifiedRewardManager:
#     """一个简化的 RewardManager，专注于测试多进程逻辑。"""

#     def __init__(self, compute_score_fn, num_examine=2):
#         """
#         初始化。我们只需要 `compute_score` 函数和 `num_examine`。
#         """
#         self.compute_score = compute_score_fn
#         self.num_examine = num_examine
#         self.print_counter = defaultdict(int)

#     def _process_single_item(self, i, data_item):
#         """
#         这是将在子进程中运行的核心逻辑。
#         它现在处理简单的字典，而不是复杂的对象。
#         """
#         try:
#             # 直接从简化的数据项中提取信息
#             prompt_str = data_item["prompt"]
#             response_str = data_item["solution"]
#             ground_truth = data_item["ground_truth"]
#             data_source = data_item["data_source"]
#             extra_info = data_item.get("extra_info", {})
#             extra_info["prompt"]=prompt_str

#             # --- 这是你要测试的核心调用 ---
#             score_result = self.compute_score(
#                 data_source=data_source,
#                 solution_str=response_str,
#                 ground_truth=ground_truth,
#                 extra_info=extra_info,
#             )
            
#             print(f"actually score_result:type:{type(score_result)}, result:{score_result}")
#             # 返回原始索引和结果，以便主进程聚合
#             return (i, score_result)

#         except Exception as e:
#             # 在子进程中捕获错误并返回，以便主进程可以优雅地处理它
#             print(f"[Process PID: {os.getpid()}] 在处理项目 {i} 时捕获到错误: {e}")
#             return (i, e) # 返回错误对象本身

#     def __call__(self, data: list):
#         """
#         这个方法接收一个简单的列表，而不是 DataProto 对象。
#         """
#         print(f"主进程 PID: {os.getpid()}")
#         print(f"开始处理 {len(data)} 个数据项，使用多进程...\n")

#         # 我们不再需要 torch.tensor，用一个简单的字典来存储结果
#         final_rewards = {}
#         reward_extra_info = defaultdict(list)
        
#         # 核心的多进程逻辑保持不变
#         with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
#             # 提交所有任务
#             futures = {
#                 executor.submit(self._process_single_item, i, item): i
#                 for i, item in enumerate(data)
#             }

#             # 当任务完成时，处理它们的结果
#             for future in concurrent.futures.as_completed(futures):
#                 original_index = futures[future]
#                 # try:
#                     # 获取结果，如果子进程出错，这里会重新抛出异常
#                     # 但我们已经在 _process_single_item 中捕获了它，所以这里得到的是一个元组
#                 index, result = future.result()

#                 # 检查子进程是否返回了一个错误
#                 if isinstance(result, Exception):
#                     print(f"主进程：数据项 {index} 处理失败。错误: {result}\n")
#                     final_rewards[index] = {"error": str(result)}
#                     continue
                
#                 # --- 在主进程中安全地聚合结果 ---
#                 score = result
#                 final_rewards[index] = score
#                 if isinstance(result, dict):
#                     for key, value in result.items():
#                         reward_extra_info[key].append(value)

#                 # --- 在主进程中安全地处理打印逻辑 ---
#                 data_source = data[index]["data_source"]
#                 if self.print_counter[data_source] < self.num_examine:
#                     self.print_counter[data_source] += 1
#                     print(f"--- 主进程打印检查项 {index} (源: {data_source}) ---")
#                     print(f"  Prompt: {data[index]['prompt']}")
#                     print(f"  Solution: {data[index]['solution']}")
#                     print(f"  聚合结果: {result}")
#                     print("------------------------------------------\n")

#                 # except Exception as e:
#                 #     # 捕获任务执行或结果检索中可能发生的任何其他严重错误
#                 #     print(f"主进程：处理数据项 {original_index} 失败，出现严重错误: {e}\n")

#         return {
#             "final_rewards": final_rewards,
#             "reward_extra_info": reward_extra_info,
#         }

# # ==============================================================================
# # 3. 主执行部分
# # ==============================================================================
# if __name__ == "__main__":
#     # 创建模拟的数据。这是一个简单的字典列表，完全替代了 DataProto。
#     mock_data = [
#         # {"prompt": "1+1=?", "solution": "2", "ground_truth": "2", "data_source": "math"},
#         # {"prompt": "首都?", "solution": "北京", "ground_truth": "北京", "data_source": "geo"},
#         # {"prompt": "2+2=?", "solution": "4", "ground_truth": "4", "data_source": "math"},
#         # {"prompt": "代码会出错吗?", "solution": "this will fail", "ground_truth": "N/A", "data_source": "test"},
#         {"prompt": "法国的首都是?", "solution": "巴黎", "ground_truth": "巴黎", "data_source": "geo"},
#         {"prompt": "3+3=?", "solution": "6", "ground_truth": "6", "data_source": "math"},
#         {"prompt": "日本的首都是?", "solution": "东京", "ground_truth": "东京", "data_source": "geo"},
#     ]

#     # 1. 实例化简化的管理器，并传入我们想要测试的函数
#     config = {"custom_reward_function":{"path":"/data2/private/linzejin/verl/RLVR/verify_reward.py", "name":"compute_score"}}
#     extra_info = {"prompt":"1+1=2"}
#     print(f"begining extra_info:{extra_info}")
#     reward_fn = get_custom_reward_fn(config,module_type="train")
#     reward_manager = SimplifiedRewardManager(compute_score_fn=reward_fn)

#     # 2. 调用它，传入我们的模拟数据
#     final_results = reward_manager(mock_data)

#     # 3. 打印最终的聚合结果
#     print("\n================== 所有任务完成 ==================")
#     print("最终聚合的奖励分数:")
#     # 按照原始顺序打印结果
#     for i in range(len(mock_data)):
#         print(f"  - 项目 {i}: {final_results['final_rewards'].get(i, '处理失败')}")
    
#     print("\n最终聚合的额外信息:")
#     for key, values in final_results['reward_extra_info'].items():
#         print(f"  - {key}: {values}")


