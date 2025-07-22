import json
from datasets import load_dataset

def main():
    """简单加载AIME 2024/2025数据集并保存为JSON"""
    
    # 要加载的数据集列表
    datasets_to_load = [
        "opencompass/AIME2025",
        "Maxwell-Jia/AIME_2024",
        "HuggingFaceH4/MATH-500"
    ]
    
    all_data = []
    
    for dataset_name in datasets_to_load:
        try:
            print(f"正在加载: {dataset_name}")
            if dataset_name == "Maxwell-Jia/AIME_2024":
                dataset = load_dataset(dataset_name)
                for split_name, split_data in dataset.items():
                    for item in split_data:
                        item['source'] = "AIME2024"
                        all_data.append(item)
            elif dataset_name == "opencompass/AIME2025":
                dataset = dict(load_dataset(dataset_name, "AIME2025-I"))
                for split_name, split_data in dataset.items():
                    for item in split_data:
                        item = dict(item)
                        item['source'] = "AIME2025"
                        all_data.append(item)
                dataset = dict(load_dataset(dataset_name, "AIME2025-II"))
                for split_name, split_data in dataset.items():
                    for item in split_data:
                        item = dict(item)
                        item['source'] = "AIME2025"
                        all_data.append(item)
            elif dataset_name == "HuggingFaceH4/MATH-500":
                dataset = dict(load_dataset(dataset_name))
                for split_name, split_data in dataset.items():
                    for item in split_data:
                        item = dict(item)
                        print(item)
                        item['source'] = "MATH500"
                        all_data.append(item)
                
            
            # 将所有split的数据合并
            
            
            print(f"成功加载 {dataset_name}")
            
        except Exception as e:
            print(f"加载失败 {dataset_name}: {e}")
    
    # 保存为JSON
    with open('aime_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"总共加载了 {len(all_data)} 条数据")
    print("已保存到 aime_dataset.json")

if __name__ == "__main__":
    main()