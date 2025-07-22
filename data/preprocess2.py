import json
import os
from collections import defaultdict
import random

def process_aime_dataset():
    """
    处理AIME数据集，按source分类并按3:1比例分割到train和test
    """
    
    # 读取原始数据
    try:
        with open('aime_dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("错误：找不到aime_dataset.json文件")
        return
    except json.JSONDecodeError:
        print("错误：JSON文件格式不正确")
        return
    
    # 按source分组数据
    source_groups = defaultdict(list)
    
    for item in data:
        source = item.get('source', 'unknown')
        
        # 根据source提取对应的problem和answer字段
        if source == "MATH500":
            problem = item.get('problem', '')
            answer = item.get('answer', '')
        elif source == "AIME2025":
            # 其他源使用Problem和answer
            problem = item.get('question', '')
            answer = str(item.get('answer', ''))
        elif source == "AIME2024":
            # 其他源使用Problem和answer
            problem = item.get('Problem', '')
            answer = str (item.get('Answer', ''))
        
        # 统一格式存储
        processed_item = {
            'problem': problem,
            'answer': answer,
            "source": source
        }
        
        source_groups[source].append(processed_item)
    
    print(f"找到 {len(source_groups)} 个不同的数据源:")
    for source, items in source_groups.items():
        print(f"  {source}: {len(items)} 条数据")
    
    # 创建MATH目录
    os.makedirs('MATH', exist_ok=True)
    
    # 准备train和test数据
    train_data = []
    test_data = []
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 对每个source按3:1比例分割
    for source, items in source_groups.items():
        # 打乱数据顺序
        random.shuffle(items)
        
        # 计算分割点（3:1比例）
        total_count = len(items)
        train_count = int(total_count * 0.85)  # 85%用于训练
        
        # 分割数据
        train_items = items[:train_count]
        test_items = items[train_count:]
        
        print(f"{source}: {len(train_items)} 条训练数据, {len(test_items)} 条测试数据")
        
        # 添加到总的train和test数据中
        train_data.extend(train_items)
        test_data.extend(test_items)
    
    # 再次打乱最终数据
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # 保存train.json
    with open('MATH/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存test.json
    with open('MATH/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成！")
    print(f"训练集: {len(train_data)} 条数据 -> MATH/train.json")
    print(f"测试集: {len(test_data)} 条数据 -> MATH/test.json")
    
    # 验证数据格式
    print("\n验证数据格式:")
    if train_data:
        sample_train = train_data[0]
        print(f"训练集样本格式: {list(sample_train.keys())}")
    if test_data:
        sample_test = test_data[0]
        print(f"测试集样本格式: {list(sample_test.keys())}")

def verify_split():
    """
    验证分割结果
    """
    try:
        with open('MATH/train.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open('MATH/test.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"\n验证结果:")
        print(f"训练集大小: {len(train_data)}")
        print(f"测试集大小: {len(test_data)}")
        print(f"总数据量: {len(train_data) + len(test_data)}")
        print(f"训练/测试比例: {len(train_data)/(len(train_data) + len(test_data)):.2f}:{len(test_data)/(len(train_data) + len(test_data)):.2f}")
        
        # 检查数据格式
        if train_data:
            required_keys = {'problem', 'answer'}
            train_keys = set(train_data[0].keys())
            test_keys = set(test_data[0].keys())
            
            if train_keys == required_keys and test_keys == required_keys:
                print("✓ 数据格式正确")
            else:
                print("✗ 数据格式不正确")
                print(f"训练集键: {train_keys}")
                print(f"测试集键: {test_keys}")
                print(f"期望键: {required_keys}")
        
    except FileNotFoundError as e:
        print(f"验证失败: {e}")

if __name__ == "__main__":
    print("开始处理AIME数据集...")
    process_aime_dataset()
    verify_split()