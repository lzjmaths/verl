
import json
import re
import os

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")
def extract_boxed_content(text: str) -> list[str]:
    """
    一个健壮的函数，用于提取 \\boxed{...} 的内容，能正确处理嵌套的大括号。
    例如，可以正确处理 \\boxed{\\frac{1}{2}}。

    Args:
        text: 包含LaTeX格式的源字符串。

    Returns:
        一个包含所有提取内容的字符串列表。
    """
    extractions = []
    # 使用 finditer 来查找所有 '\boxed{' 的起始位置
    for match in re.finditer(r'\\boxed{', text):
        # 内容的起始位置是在 '\boxed{' 之后
        content_start_pos = match.end()
        
        brace_level = 1
        # 从内容开始处向后查找匹配的右括号
        for i in range(content_start_pos, len(text)):
            char = text[i]
            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
            
            # 当括号层级回到0时，说明找到了匹配的结束括号
            if brace_level == 0:
                content = text[content_start_pos:i]
                extractions.append(content)
                # 找到了就跳出内层循环，继续查找下一个 \boxed{
                break
                
    return extractions


def verify(data_source, solution_str: str, ground_truth: str,extra_info=None, timeout_score: float = 0) -> bool:
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


def compute_score(data_source, solution_str: str, ground_truth: str,extra_info=None, timeout_score: float = 0) -> bool:
    if solution_str is None or ground_truth is None:
            print(f"警告：第 {index + 1} 项缺少 'solution_str' 或 'ground_truth' 字段，已跳过。")
            # 此项不计入总数，或者可以视为false
            # 为了保持比例的准确性，我们暂时不统计此项
            total_items -=1
            return 0.0
            
        # 使用正则表达式查找所有 \boxed{...} 的内容
        # re.DOTALL 确保可以匹配包括换行符在内的任何字符
    extracted_solutions = extract_boxed_content(solution_str)
    extracted_solutions = ["\\boxed{" + extracted_solutions + "}" for extracted_solutions in extracted_solutions] # 添加 \boxed{} 前缀
    
    # print(f"原始 'solution_str': {solution_str[:100]}...") # 打印部分solution_str
    # print(f"提取到的 'boxed' 内容: {extracted_solutions}")
    # print(f"原始 'ground_truth': {ground_truth}")

    is_item_true = False
    if not extracted_solutions:
        # print("未在此项中找到 \\boxed{} 语句。此项记为 False。")
        is_item_true = False
    else:
        # 对比所有提取出的答案
        for extracted in extracted_solutions:
            score = verify(
                "MATH500",  # 问题名称是固定的
                solution_str=extracted,
                ground_truth=str(ground_truth) # 确保 ground_truth 是字符串
            )
            if score == 1.0:
                is_item_true = True
                return 1.0 # 只要有一个答案正确，就停止对比，并将该项标记为True

    # 根据对比结果更新计数
    if is_item_true:
        return 1.0
        print("该项最终判定为: True")
    else:
        return 0.0
        print("该项最终判定为: False")