# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

import multiprocessing
import os
from functools import partial
import importlib.util
import sys
import hashlib

import ray

from verl import DataProto
from verl.utils.reward_score import default_compute_score


class get_custom_reward_fn:
    def __init__(self, config, module_type="train"):
        if module_type == "train":
            reward_fn_config = config.get("custom_reward_function") or {}
        else:
            reward_fn_config = config.get("custom_test_function") or {}
        
        file_path = reward_fn_config.get("path")
        if not file_path:
            self.invokable = None
            self.reward_kwargs = {}
            return

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Custom function file '{file_path}' not found.")

        # --- 这是关键的修改 ---
        # 1. 创建一个基于文件路径的、唯一的模块名。
        #    这样可以避免命名冲突，并确保路径是唯一的。
        #    我们用 hash 来确保它是一个合法的 Python 标识符。
        abs_path = os.path.abspath(file_path)
        hasher = hashlib.md5(abs_path.encode('utf-8'))
        module_name = f"custom_reward_module_{hasher.hexdigest()}"

        # 2. 检查这个唯一的模块是否已经被加载了。
        #    这可以防止在同一个进程中重复加载同一个文件。
        if module_name in sys.modules:
            custom_module = sys.modules[module_name]
        else:
            # 动态加载模块
            spec = importlib.util.spec_from_file_location(module_name, abs_path)
            if spec is None:
                raise ImportError(f"Could not create spec from file: {abs_path}")
            
            custom_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = custom_module # 将其添加到缓存
            spec.loader.exec_module(custom_module)

        function_name = reward_fn_config.get("name")
        if not hasattr(custom_module, function_name):
            raise AttributeError(f"Function '{function_name}' not found in '{file_path}'.")

        # print(f"Using customized function '{function_name}' from '{file_path}' (module: {module_name})")
        
        self.raw_fn_or_class = getattr(custom_module, function_name)

        # print(f"--- raw_fn_or_class type:{type(self.raw_fn_or_class)}, content:{self.raw_fn_or_class} ---")
        self.reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

        # --- 让你的代码更健壮，支持工厂模式 ---
        # 检查它是否是一个类（可以被实例化）
        if isinstance(self.raw_fn_or_class, type):
            self.invokable = self.raw_fn_or_class(config) # 假设它需要config来初始化
        else:
            # 否则就认为它是一个可直接调用的函数
            self.invokable = self.raw_fn_or_class
    
    def __call__(self, *args, **kwargs):
        return self.invokable(*args, **kwargs, **self.reward_kwargs)

if __name__ == "__main__":
    config = {"custom_reward_function":{"path":"/data2/private/linzejin/verl/RLVR/verify_reward.py", "name":"compute_score"}}
    extra_info = {"prompt":"1+1=2"}
    print(f"begining extra_info:{extra_info}")
    reward_fn = get_custom_reward_fn(config,module_type="train")


    print(reward_fn("data_source", "1+1=2", "1+1=2", extra_info = {"prompt":"1+1=2"}))
# class get_custom_reward_fn:
#     def __init__(self, config, module):
#         import importlib.util
#         import sys

#         if module == "train":
#             reward_fn_config = config.get("custom_reward_function") or {}
#         else:
#             reward_fn_config = config.get("custom_test_function") or {}
#             print(f"starting use test_function{reward_fn_config}")
#         file_path = reward_fn_config.get("path")
#         if not file_path:
#             return None

#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

#         spec = importlib.util.spec_from_file_location("custom_module", file_path)
#         module = importlib.util.module_from_spec(spec)
#         try:
#             sys.modules["custom_module"] = module
#             spec.loader.exec_module(module)
#         except Exception as e:
#             raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

#         function_name = reward_fn_config.get("name")
#         if not hasattr(module, function_name):
#             raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

#         print(f"using customized reward function '{function_name}' from '{file_path}'")
#         self.raw_fn = getattr(module, function_name)

#         self.reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

#     def __call__(self, *args, **kwargs):
#         return self.raw_fn(*args, **kwargs, **self.reward_kwargs)



def load_reward_manager(config, tokenizer, num_examine, module, **reward_kwargs):
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """
    from verl.workers.reward_manager import get_reward_manager_cls

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # Try to get a custom reward function based on the configuration
    compute_score = get_custom_reward_fn(config, module)
    final_compute_score = compute_score

    if compute_score is None:
        print("compute_score is NOOOOOOOOOOOO!")
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024)
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)
