# -*- coding: utf-8 -*-

import json
import re
import matplotlib.pyplot as plt
import time
import os

# --- 1. 在此处定义您的JSON文件名 ---
JSON_FILE_PATH = "/mnt/workspace/linzejin/verl/verl/logs/rlmr-f-l@3k-MATH500-Qwen2.5-Math-1.5B-0805-112316/reward/false_positive_instances.json"
FINAL_CHART_IMAGE_PATH = "solution_verification_results.png"

# --- 2. 模拟的 compute_score 函数 ---
# 重要提示：请将此函数替换为您自己的实际评分函数。
#
# 这个模拟函数简单地检查提取的答案是否与标准答案完全相同。
# 在您的实际应用中，它可能是一个更复杂的评估逻辑。
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

def read_jsonl_simple(file_path):
    """
    解析一个非标准的、缺少逗号且对象连续的文件。

    该函数处理如下格式的文件:
    {
        "key1": "value1"
        "key2": "value2"
    }
    {
        "key3": "value3"
    }
    """
    all_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 1. 将整个文件读入一个字符串
            content = f.read()
    except FileNotFoundError:
        print(f"错误: 文件未找到 at path: {file_path}")
        return []

    # 2. 将内容分割成独立的、类似对象的文本块。
    # 我们用一个特殊的分隔符替换对象之间的边界，然后分割。
    # '}\n{' 是对象之间的典型模式。
    object_strings = content.replace('}\n{', '}\n###SEPARATOR###\n{').split('###SEPARATOR###')

    for i, obj_str in enumerate(object_strings):
        # 跳过可能因分割产生的空字符串
        if not obj_str.strip():
            continue

        # 3. 在块内部，为缺失的逗号打补丁。
        # 我们查找一个双引号 + 换行符 + 双引号的模式 ( " \n " )
        # 并将其替换为双引号 + 逗号 + 换行符 + 双引号 ( " , \n " )
        # 这能安全地在键值对之间添加逗号。
        fixed_obj_str = obj_str.replace('"\n    "', '",\n    "') # 针对有缩进的情况
        fixed_obj_str = fixed_obj_str.replace('"\n"', '",\n"')       # 针对无缩进的情况


        # 4. 尝试用标准库解析修复后的字符串
        try:
            # 移除可能存在的前后空白
            clean_str = fixed_obj_str.strip()
            if clean_str:
                data = json.loads(clean_str)
                all_data.append(data)
        except json.JSONDecodeError as e:
            # 如果即使修复后仍然失败，就打印警告并跳过这个块
            print(f"警告：跳过第 {i+1} 个无法解析的对象块。错误: {e}")
            print(f"   > 问题块内容:\n{obj_str}\n")
            
    return all_data
def compute_score(data_source, solution_str: str, ground_truth: str,extra_info=None, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(), StringExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(), StringExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [solution_str])
    except Exception:
        print(f"answer:{solution_str}")
        print(f"ground_truth:{ground_truth}")
        pass
    except TimeoutException:
        print(f"answer:{solution_str}")
        print(f"ground_truth:{ground_truth}")
        ret_score = timeout_score
    # print(ret_score)
    return ret_score

# print(compute_score("123","our answer is \\boxed{ 30 } cffwef", "30^\\circ"))

# --- 3. 主处理与可视化函数 ---
def process_and_visualize_solutions(json_path: str):
    """
    读取JSON文件，处理每项数据，并实时可视化结果。
    """
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误：找不到文件 '{json_path}'。")
        print("已为您生成一个示例JSON文件 'solutions_data.json'，请根据您的数据修改后重新运行。")
        generate_sample_json()
        return

    # 读取JSON数据
    data = read_jsonl_simple(json_path)

    # 初始化统计计数器
    true_count = 0
    false_count = 0
    total_items = len(data)

    # 设置matplotlib进行实时绘图
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()
    
    print(f"--- 开始处理 {total_items} 项数据 ---")

    # 遍历JSON文件中的每一项
    for index, item in enumerate(data):
        print(f"\n--- 正在处理第 {index + 1}/{total_items} 项 ---")
        
        # 获取 solution_str 和 ground_truth 字段
        solution_str = item.get("solution_str", item.get("input", None))
        ground_truth = item.get("ground_truth")

        if solution_str is None or ground_truth is None:
            print(f"警告：第 {index + 1} 项缺少 'solution_str' 或 'ground_truth' 字段，已跳过。")
            # 此项不计入总数，或者可以视为false
            # 为了保持比例的准确性，我们暂时不统计此项
            total_items -=1
            continue
            
        # 使用正则表达式查找所有 \boxed{...} 的内容
        # re.DOTALL 确保可以匹配包括换行符在内的任何字符
        extracted_solutions = re.findall(r'\\boxed{(.*?)}', solution_str, re.DOTALL)
        extracted_solutions = ["\\boxed{" + extracted_solutions + "}" for extracted_solutions in extracted_solutions] # 添加 \boxed{} 前缀
        
        print(f"原始 'solution_str': {solution_str[:100]}...") # 打印部分solution_str
        print(f"提取到的 'boxed' 内容: {extracted_solutions}")
        print(f"原始 'ground_truth': {ground_truth}")

        is_item_true = False
        if not extracted_solutions:
            print("未在此项中找到 \\boxed{} 语句。此项记为 False。")
            is_item_true = False
        else:
            # 对比所有提取出的答案
            for extracted in extracted_solutions:
                score = compute_score(
                    "MATH500",  # 问题名称是固定的
                    solution_str=extracted,
                    ground_truth=str(ground_truth) # 确保 ground_truth 是字符串
                )
                if score == 1.0:
                    is_item_true = True
                    break # 只要有一个答案正确，就停止对比，并将该项标记为True

        # 根据对比结果更新计数
        if is_item_true:
            true_count += 1
            print("该项最终判定为: True")
        else:
            false_count += 1
            print("该项最终判定为: False")

        # --- 实时更新图表 ---
        ax.clear() # 清除旧的图表
        
        labels = 'True', 'False'
        sizes = [true_count, false_count]
        colors = ['#4CAF50', '#F44336'] # 绿色和红色
        print(f"ratio:{(true_count*1.0/(true_count*1.0+false_count*1.0))}")
        
        # 如果没有任何数据，则显示提示信息
        if true_count == 0 and false_count == 0:
             ax.text(0.5, 0.5, '等待数据中...', horizontalalignment='center', verticalalignment='center')
        else:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, wedgeprops={'edgecolor': 'white'})
        
        ax.set_title(f'实时验证结果\n(已处理: {index + 1}/{total_items})')
        ax.axis('equal')  # 保证饼图是正圆形
        
        plt.draw()
        plt.pause(0.2) # 暂停一小段时间以显示更新，可以调整

    print(f"\n--- 处理完成 ---")
    print(f"最终结果: True = {true_count}, False = {false_count}")
    
    # 保存最终的图表
    fig.savefig(FINAL_CHART_IMAGE_PATH)
    print(f"最终图表已保存至: {FINAL_CHART_IMAGE_PATH}")

    # 关闭交互模式，并保持最终窗口显示
    plt.ioff()
    plt.show()

def generate_sample_json():
    """生成一个示例JSON文件，以便用户测试。"""
    sample_data = [
        {
            "solution_str": "The first step is to add 2 and 2, which gives 4. The final answer is \\boxed{4}.",
            "ground_truth": "4"
        },
        {
            "solution_str": "We calculate the area of the square with side length 5. Area = 5*5 = 25. So the result is \\boxed{25}.",
            "ground_truth": "25"
        },
        {
            "solution_str": "The question is complex. Let's try to simplify. Maybe the answer is \\boxed{10}. Or is it \\boxed{12}?",
            "ground_truth": "12"
        },
        {
            "solution_str": "After solving the equation x - 3 = 7, we get x = 10. The wrong answer might be \\boxed{9}.",
            "ground_truth": "10"
        },
        {
            "solution_str": "There is no boxed answer here.",
            "ground_truth": "5"
        }
    ]
    with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=4)


# --- 4. 运行主程序 ---
if __name__ == "__main__":
    process_and_visualize_solutions(JSON_FILE_PATH)