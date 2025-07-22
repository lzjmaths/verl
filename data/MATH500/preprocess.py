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
"""
Preprocess the GSM8k dataset to parquet format
"""
from datasets import Dataset
import argparse
import os
import re
import json

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--dataset_dir", default=None)

    args = parser.parse_args()

    if args.dataset_dir == None:
        print("There is no dataset found")
        exit(0)

    if args.local_dir is not None:
        save_dir = args.local_dir
    else:
        save_dir = args.dataset_dir

    train_dataset = datasets.load_dataset("json", data_files=os.path.join(args.dataset_dir, 'train.json'), split='train')
    test_dataset = datasets.load_dataset("json", data_files=os.path.join(args.dataset_dir, 'test.json'), split='train')

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")

            data_source = "lighteval/MATH"
            
            question = question_raw + " " + instruction_following

            solution = example.pop("answer")
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                    "source": data_source
                },
            }
            return data

        return process_fn
    train_dataset = Dataset.from_list(train_dataset)
    test_dataset = Dataset.from_list(test_dataset)
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
