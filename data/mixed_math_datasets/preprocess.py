dataset_sources = {
    "GSM8k": "openai/gsm8k",
    "AIME2024": "Maxwell-Jia/AIME_2024",
    "AIME2025": "opencompass/AIME2025", 
    "MATH500": "HuggingFaceH4/MATH-500"
}

def load_json_dataset(json_path):
    """Load dataset from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Direct list of examples
            return Dataset.from_list(data)
        elif isinstance(data, dict):
            # Dictionary with train/test splits or other structure
            if 'train' in data and 'test' in data:
                return {
                    'train': Dataset.from_list(data['train']),
                    'test': Dataset.from_list(data['test'])
                }
            else:
                # Single dataset structure
                return Dataset.from_dict(data)
        else:
            raise ValueError(f"Unsupported JSON structure: {type(data)}")
    except Exception as e:
        print(f"Error loading JSON dataset from {json_path}: {e}")
        return None# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess multiple math datasets (GSM8k, AIME2024, AIME2025, MATH500) to parquet format
"""
import argparse
import os
import re
import json
import datasets
from datasets import concatenate_datasets, Dataset
from verl.utils.hdfs_io import copy, makedirs

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def extract_solution_flexible(solution_str, dataset_name):
    """Extract solution with flexible patterns for different datasets"""
    if dataset_name.startswith("AIME") or dataset_name == "MATH500":
        # For AIME and MATH500, try different patterns
        patterns = [
            r"#### (\\-?[0-9\\.\\,]+)",  # Standard GSM8k pattern
            r"\\boxed{([^}]+)}",  # LaTeX boxed format
            r"Answer: ([0-9\\.\\,]+)",  # Answer: format
            r"Final answer: ([0-9\\.\\,]+)",  # Final answer: format
            r"([0-9\\.\\,]+)$",  # Number at end of string
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution_str)
            if match:
                solution = match.group(1).replace(",", "")
                return solution
        
        # If no pattern matches, return the last number found
        numbers = re.findall(r"\\-?[0-9\\.\\,]+", solution_str)
        if numbers:
            return numbers[-1].replace(",", "")
        
        # Fallback: return the original string (you might want to handle this differently)
        return solution_str.strip()
    else:
        # Use original GSM8k extraction for GSM8k dataset
        return extract_solution(solution_str)

def load_dataset_safe(data_source, config_name="default"):
    """Safely load dataset with error handling"""
    # Check if it's a local JSON file
    if data_source.endswith('.json') and os.path.exists(data_source):
        return load_json_dataset(data_source)
    
    try:
        return datasets.load_dataset(data_source, config_name)
    except Exception as e:
        print(f"Error loading {data_source} with config {config_name}: {e}")
        try:
            # Try with "main" config
            return datasets.load_dataset(data_source, "main")
        except Exception as e2:
            print(f"Error loading {data_source} with main config: {e2}")
            try:
                # Try without config name
                return datasets.load_dataset(data_source)
            except Exception as e3:
                print(f"Error loading {data_source} without config: {e3}")
                return None

def save_dataset_as_json(dataset, filepath):
    """Save dataset to JSON file"""
    try:
        # Convert dataset to list of dictionaries
        data_list = []
        for item in dataset:
            data_list.append(item)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"Saved JSON to: {filepath}")
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")
    def process_fn(example, idx):
        # Handle different dataset structures
        if dataset_name == "GSM8k":
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")
        elif dataset_name.startswith("AIME") or dataset_name == "MATH500":
            # Adjust field names based on actual dataset structure
            # You might need to modify these based on the actual field names
            question_raw = example.get("problem", example.get("question", ""))
            answer_raw = example.get("solution", example.get("answer", ""))
        else:
            question_raw = example.get("question", "")
            answer_raw = example.get("answer", "")
        
        instruction_following = 'Let\'s think step by step and output the final answer after "####".'
        question = question_raw + " " + instruction_following
        
        try:
            solution = extract_solution_flexible(answer_raw, dataset_name)
        except:
            # If extraction fails, use a placeholder or skip
            solution = "ERROR"
        
        data = {
            "data_source": dataset_name,
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
                "answer": answer_raw,
                "question": question_raw,
                "original_dataset": dataset_name,
            },
        }
        return data
    return process_fn



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--datasets", nargs='+', default=["AIME2024", "AIME2025", "MATH500"],
                       help="List of datasets to process")
    parser.add_argument("--json_files", nargs='+', default=None,
                       help="List of JSON files to include in processing")
    parser.add_argument("--save_json", action="store_true", default=True,
                       help="Save datasets in JSON format in addition to parquet")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode to inspect dataset structure")
    args = parser.parse_args()
    
    if args.local_dir is not None:
        save_dir = args.local_dir
    else:
        save_dir = "/data3/private/linzejin/verl/data/mixed_math_datasets"
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Dataset source mapping
    
    
    all_train_datasets = []
    all_test_datasets = []
    
    def make_map_fn(split, dataset_name):
        def process_fn(example, idx):
        # Handle different dataset structures
            if dataset_name == "GSM8k":
                question_raw = example.pop("question")
                answer_raw = example.pop("answer")
            elif dataset_name.startswith("AIME") or dataset_name == "MATH500":
                # Adjust field names based on actual dataset structure
                # You might need to modify these based on the actual field names
                question_raw = example.get("problem", example.get("question", ""))
                answer_raw = example.get("solution", example.get("answer", ""))
            else:
                question_raw = example.get("question", "")
                answer_raw = example.get("answer", "")
            
            instruction_following = 'Let\'s think step by step and output the final answer after "####".'
            question = question_raw + " " + instruction_following
            
            try:
                solution = extract_solution(answer_raw, dataset_name)
            except:
                # If extraction fails, use a placeholder or skip
                solution = "ERROR"
            
            data = {
                "data_source": dataset_name,
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
                    "answer": answer_raw,
                    "question": question_raw,
                    "original_dataset": dataset_name,
                },
            }
            return data
        return process_fn

    # Process Hugging Face datasets
    for dataset_name in args.datasets:
        print(f"Processing {dataset_name}...")
        
        if dataset_name not in dataset_sources:
            print(f"Unknown dataset: {dataset_name}")
            continue
        
        data_source = dataset_sources[dataset_name]
        dataset = load_dataset_safe(data_source)
        
        if dataset is None:
            print(f"Failed to load {dataset_name}, skipping...")
            continue
        
        # Debug: print dataset structure
        if args.debug:
            print(f"Dataset {dataset_name} structure:")
            print(f"  Available splits: {list(dataset.keys())}")
            first_split = list(dataset.keys())[0]
            print(f"  Columns in {first_split}: {dataset[first_split].column_names}")
            print(f"  First example: {dataset[first_split][0]}")
            print("-" * 50)
        
        # Handle different dataset structures
        if "train" in dataset:
            train_dataset = dataset["train"]
        else:
            # If no train split, use the entire dataset and we'll split later
            train_dataset = dataset[list(dataset.keys())[0]]
        
        if "test" in dataset:
            test_dataset = dataset["test"]
        elif "validation" in dataset:
            test_dataset = dataset["validation"]
        else:
            # Split the dataset if no test set exists
            split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset["train"]
            test_dataset = split_dataset["test"]
        
        # Process datasets
        try:
            processed_train = train_dataset.map(
                function=make_map_fn("train", dataset_name), 
                with_indices=True,
                remove_columns=[]  # Don't remove columns, let the function handle it
            )
            processed_test = test_dataset.map(
                function=make_map_fn("test", dataset_name), 
                with_indices=True,
                remove_columns=[]  # Don't remove columns, let the function handle it
            )
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
        
        all_train_datasets.append(processed_train)
        all_test_datasets.append(processed_test)
        
        print(f"Processed {dataset_name}: {len(processed_train)} train, {len(processed_test)} test")
    
    # Process JSON files if provided
    if args.json_files:
        for json_file in args.json_files:
            print(f"Processing JSON file: {json_file}...")
            
            if not os.path.exists(json_file):
                print(f"JSON file not found: {json_file}")
                continue
            
            dataset = load_dataset_safe(json_file)
            if dataset is None:
                print(f"Failed to load {json_file}, skipping...")
                continue
            
            # Get dataset name from filename
            dataset_name = os.path.splitext(os.path.basename(json_file))[0]
            
            # Handle different dataset structures
            if isinstance(dataset, dict) and "train" in dataset:
                train_dataset = dataset["train"]
                test_dataset = dataset.get("test", None)
            else:
                # Single dataset, need to split
                train_dataset = dataset
                test_dataset = None
            
            # Split if no test set
            if test_dataset is None:
                split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = split_dataset["train"]
                test_dataset = split_dataset["test"]
            
            # Process datasets
            try:
                processed_train = train_dataset.map(
                    function=make_map_fn("train", dataset_name), 
                    with_indices=True,
                    remove_columns=[]  # Don't remove columns, let the function handle it
                )
                processed_test = test_dataset.map(
                    function=make_map_fn("test", dataset_name), 
                    with_indices=True,
                    remove_columns=[]  # Don't remove columns, let the function handle it
                )
            except Exception as e:
                print(f"Error processing JSON file {dataset_name}: {e}")
                continue
            
            all_train_datasets.append(processed_train)
            all_test_datasets.append(processed_test)
            
            print(f"Processed {dataset_name}: {len(processed_train)} train, {len(processed_test)} test")
    
    if not all_train_datasets:
        print("No datasets were successfully processed!")
        exit(1)
    
    # Concatenate all datasets
    print("Concatenating datasets...")
    combined_train = concatenate_datasets(all_train_datasets)
    combined_test = concatenate_datasets(all_test_datasets)
    
    # Shuffle the combined datasets
    combined_train = combined_train.shuffle(seed=42)
    combined_test = combined_test.shuffle(seed=42)
    
    print(f"Combined datasets: {len(combined_train)} train, {len(combined_test)} test")
    
    # Save to parquet
    print("Saving to parquet...")
    combined_train.to_parquet(os.path.join(save_dir, "train.parquet"))
    combined_test.to_parquet(os.path.join(save_dir, "test.parquet"))
    
    # Save to JSON if requested
    if args.save_json:
        print("Saving to JSON...")
        save_dataset_as_json(combined_train, os.path.join(save_dir, "train.json"))
        save_dataset_as_json(combined_test, os.path.join(save_dir, "test.json"))
    
    # Copy to HDFS if specified
    hdfs_dir = args.hdfs_dir
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=save_dir, dst=hdfs_dir)
    
    print("Processing complete!")
    print(f"Files saved to: {save_dir}")
    if args.save_json:
        print("Both parquet and JSON formats saved.")
    else:
        print("Only parquet format saved.")
    if hdfs_dir:
        print(f"Files copied to HDFS: {hdfs_dir}")