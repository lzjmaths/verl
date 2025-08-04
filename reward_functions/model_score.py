import openai
import json
import logging
from typing import Union
import re
import gc
import threading


from tqdm import tqdm
from transformers import AutoTokenizer
import time
import os

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


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

# print(compute_score("123","our answer is \\boxed{ 30 } cffwef", "30^\\circ"))

WORK_DIR = "/mnt/workspace/linzejin"

with open(os.path.join(WORK_DIR,'.apiconfig.json'), 'r', encoding='utf-8') as file:
    apiconfig = json.load(file)



def extract_boxed_score(text):
    """
    从文本中提取最后一个\\boxed{X}中的分数，转换为浮点数返回
    
    Args:
        text (str): 包含\\boxed{}的文本
        
    Returns:
        float: 提取到的分数，如果没找到则返回None
    """
    # 查找所有的\boxed{...}模式
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if not matches:
        return None
    
    # 取最后一个匹配
    last_match = matches[-1].strip()
    
    # 尝试从匹配的内容中提取数字
    return extract_number_from_content(last_match)

def extract_number_from_content(content):
    """
    从内容中提取数字并转换为浮点数
    
    Args:
        content (str): \\boxed{}中的内容
        
    Returns:
        float: 提取到的数字，如果没找到则返回None
    """
    # 清理内容，移除多余的空格和特殊字符
    content = content.strip()
    
    # 方法1: 直接尝试转换整个内容
    try:
        return float(content)
    except ValueError:
        pass
    
    # 方法2: 查找分数形式 (如 "9/10", "1/2")
    fraction_pattern = r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)'
    fraction_match = re.search(fraction_pattern, content)
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        if denominator != 0:
            return numerator / denominator
    
    # 方法3: 查找小数 (如 "8.5", "9.0")
    decimal_pattern = r'(\d+\.\d+)'
    decimal_match = re.search(decimal_pattern, content)
    if decimal_match:
        return float(decimal_match.group(1))
    
    # 方法4: 查找整数 (如 "10", "score: 9", "答案是8")
    integer_pattern = r'(\d+)'
    integer_matches = re.findall(integer_pattern, content)
    if integer_matches:
        # 如果有多个数字，优先选择看起来像分数的(0-10范围内)
        for num_str in integer_matches:
            num = int(num_str)
            if 0 <= num <= 10:  # 假设分数在0-10范围内
                return float(num)
        # 如果没有0-10范围内的，返回第一个数字
        return float(integer_matches[0])
    
    # 方法5: 查找带有"points"或"分"等关键词的数字
    score_pattern = r'(\d+(?:\.\d+)?)\s*(?:points?|分|pts?)'
    score_match = re.search(score_pattern, content, re.IGNORECASE)
    if score_match:
        return float(score_match.group(1))
    
    return None

def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return None
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    a = a.lower()
    if a in ["1","2","3","4","5","6","7","8","9","0",1,2,3,4,5,6,7,8,9,0]:
        return int(a)
    else:
        return 0

def extract_boxed(text):
    matches = re.findall(r'(true|false)', find_box(text), flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    else:
        return "false"

def extract_judgement(text):
    matches = re.findall(r'\*\*(true|false)\*\*', text, re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    return "false"

def remove_think_tags(text):
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text

def extract_tag_content(text, tag):
    pattern = fr"<{tag}>.*?</{tag}>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1]
    else:
        logging.warning(f"No content extracted in given tag <{tag}>.")
        return None

def extract_all_tag_content(text, tag):
    pattern = fr"<{tag}>.*?</{tag}>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches
    else:
        logging.warning(f"No content extracted in given tag <{tag}>.")
        return None

def remove_tag_content(text, tag):
    cleaned_text = re.sub(fr'<{tag}>.*?</{tag}>', '', text, flags=re.DOTALL)
    return cleaned_text

def remove_all_tag_content(text: str) -> str:
    if isinstance(text, str):
        return re.sub(r'<.*?>', '', text) # Delete all <> tags
    return text




class AgentBase:
    """
    The base class for all agents instance
    Capatible with both remote API calls and local inference with SGL
    """
    temperature = 0.6
    seed = 1121
    max_retries = 7
    max_tokens = 16384
    debug = False

    sgl_model  = None
    tokenizer = None
    llm = None
    TP = 1
    cilent = None

    remote_models = [
        "deepseek-r1",
        "deepseek-v3",
        "gpt-4o",
        "o3-mini",
        "o4-mini",
        "o1",
        "o3",
        "deepseek-r1-0528",
        "deepseek-r1-250528",
        "deepseek-v3-250324",
        "o4-mini-2025-04-16",
        "claude-sonnet-4-20250514-thinking",
        "claude-sonnet-4-20250514",
        "gemini-2.5-flash-preview-05-20-thinking",
        "gemini-2.5-flash-preview-05-20",
        "gpt-4.1-2025-04-14"
    ]

    local_models = [
        "/data3/private/linzejin/models/Qwen3-14B",
        "/data2/private/linzejin/models/models/Qwen/Qwen3-14B",
        "/data2/private/linzejin/models/Qwen/Qwen3-14B",
        "Qwen3-14B"
    ]

    def __init__(self, model: str):
        self.model = model
        # self.remote = True if model in AgentBase.remote_models else False
         
        if model in AgentBase.local_models:
            self.remote = True
            apiconfig = {
                "OPENAI_API_KEY": "fake-key",  # 必填但 SGLang 实际不验证 key，可填任意字符串
                "OPENAI_BASE_URL": "http://localhost:10000/v1"  # 注意加上 /v1
            }
        elif model in AgentBase.remote_models:
            self.remote = True
        self.client = openai.OpenAI(
            api_key = apiconfig['OPENAI_API_KEY'],
            base_url = apiconfig['OPENAI_BASE_URL']
        )
        if not self.remote:
            raise ValueError(f"model{model} is not in the remote_models list!")

    def format_prompt(self):
        """
        This method should be overrided by subclasses for specific useage
        """
        raise NotImplementedError("format_prompt should be implemented by subclasses of agent")

    def call_from_remote_API(self, prompt: Union[str, list[dict]]):
        for attempt in range(self.max_retries):
            try:
                client_params = {
                    'model': self.model,
                    'temperature': AgentBase.temperature,
                    'timeout': 300,
                    'messages': prompt,
                    # 'messages': [
                    #     {"role": "system", "content": "You are a helpful assistant."}
                    # ],
                    # 'max_tokens': AgentBase.max_tokens,
                    'stream': True
                }
                if AgentBase.debug:
                    client_params['seed'] = AgentBase.seed
                stream = self.client.chat.completions.create(**client_params)
                response_content = ""
                for chunk in stream:
                    if len(chunk.choices) == 0:
                        continue
                    chunk_content = chunk.choices[0].delta.content
                    if chunk_content is not None:
                        if AgentBase.debug:
                            print(chunk_content, end="", flush=True)
                        response_content += chunk_content
                if response_content.strip():
                    return response_content
                else:
                    logging.warning(f"Attempt {attempt+1}: Response was empty. Retrying...")
                    
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed with exception: {e}")
                
        # If all attempts fail, log the error and raise an exception
        error_msg = f"All {AgentBase.max_retries} attempts failed. Terminating."
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    def batch_generate(self, args_list: list, workers: int = 1):
        prompts = [self.format_prompt(*args) for args in args_list]
        if self.remote:
            results = []
            for prompt in prompts:
                result = self.call_from_remote_API(prompt)
                results.append(result)
            return results
        else:
            raise ValueError("We cannot find a possible remote model!")
    def __call__(self, *args):
        prompt = self.format_prompt(*args)
        if self.remote:
            return self.call_from_remote_API(prompt)
        else:
            return self.call_from_local_SGL_engine(prompt)

class Reviewer(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, proof: str):
        # change the seed each time
        self.seed += 1
        prompt = [
            {'role': 'user', 'content':
             'You are a reviewer for this math problem. You are provided a candidate solution of this problem, and you need to analyze and point out one part of this solution that might be incomplete or contain some flaws. We will use your advice for further judgement of this solution.\n'
             '\n'
             '### Candidate Solution\n'
             '\n'
             f'{proof}\n'
             '### Problem\n'
             '\n'
             f'{problem}\n'
             '\n'
             'A valid solution should have correct answer and rigorous process, and contain no additional unrelated output, including other problems, meaningless content, or any irrelevant text or symbols. You need to explain your rationales and evaluate this candidate solution using the following scoring criteria:\n'
           '\n'
           '**Scoring Criteria (Total: 10 points)**\n'
           '- **Format Score (1 point)**: Award 1 point if the final answer is placed inside `\\boxed{}` and there is no additional unrelated output or commentary. Otherwise, award 0 points.\n'
           '- **Answer Score (9 points)**: You need to analyze the solution for the problem under the title "Problem" step by step. For each step, examine whether the reasoning is correct, complete, and clearly explained. Point out any issues in logic, clarity, or rigor.\n\nUse the following grading criteria:\n\n- 9 points (Full credit): The reasoning is fully correct, complete, and rigorous. Each step is clearly justified and mathematically or logically sound. If there is any lack of rigorousness, do not give full points.\n\n- 7 points (Partial credit): The final answer is correct, but there are issues in the solution process, such as missing justifications, unclear reasoning, skipped steps, or minor logical gaps. Code verification will not be accepted. These issues do not affect the correctness of the final answer but indicate a lack of rigor.\n\n- 0 points (Incorrect): If meets the case that the final answer is incorrect, or the reasoning contains quite serious gap, give 0 points imediately'
           '\n'
           'There are no other point values available - only 0 or 1 for format, and 0 or 7 or 9 for answer\n'
           '\n'
           'Please try to solve this problem step by step by yourself first. Compared to the candidate solution, you need to carefully evaluate both criteria and explain your justification for each individual score. State your final score inside $\\boxed{}$ as $\\boxed{X}$ where X is the total score at the end of your response.\n'
             }]
        return prompt


def peval_pipeline(
    problems: Union[list[str], list[dict]],
    proofs: list[str],
    reviewer: str,
    reviews: int = 1,
    workers: int = 1
) -> dict:
    if isinstance(problems[0], dict):
        problems = [p['problem'] for p in problems]
    evaluate_prompts = [[pr, pf] for pr, pf in zip(problems, proofs) for _ in range(reviews)]
    reviewer_agent = Reviewer(reviewer)
    raw_reviews = reviewer_agent.batch_generate(evaluate_prompts, workers=workers)
    results = []
    for i in range(len(problems)):
        solved = 0.0
        for j in range(reviews):
            score = extract_boxed_score(raw_reviews[i * reviews + j])
            if score is not None:
                results.append({
                    'problem': problems[i],
                    'proof': proofs[i],
                    'review': remove_think_tags(raw_reviews[i * reviews + j]),
                    'judgement': score
                })
                solved += score
                break
            else:
                continue
        if solved == 0.0:
            results.append({
                'problem': problems[i],
                'proof': proofs[i],
                'review': remove_think_tags(raw_reviews[i * reviews]),
                'judgement': 0.0
            })
    return results
def extract_prompt_from_extra_info(extra_info) -> str:
    """
    从一个字典（extra_info）中按优先级提取指定的字段值。

    它会按以下顺序查找字段：
    1. 'prompt'
    2. 'problem'
    3. 'question'
    4. 'Problem' (大写)
    5. 'Question' (大写)
    6. 'Prompt' (大写)

    如果找到任何一个字段，立即返回其对应的值。
    如果所有字段都未找到，则抛出 ValueError。

    Args:
        extra_info (Dict[str, Any]): 包含附加信息的字典。

    Returns:
        Union[str, Any]: 找到的字段值。

    Raises:
        ValueError: 如果字典中不包含任何一个指定的字段。
        TypeError: 如果输入的 extra_info 不是一个字典。
    """
    if not isinstance(extra_info, dict):
        raise TypeError("输入参数 'extra_info' 必须是一个字典。")
        
    # 定义要按优先级顺序检查的字段列表
    keys_to_check = ['prompt', 'problem', 'question', 'Problem', 'Question', 'Prompt']

    # 遍历列表，查找第一个存在的字段
    for key in keys_to_check:
        if key in extra_info:
            return extra_info[key]

    # 如果循环结束后仍未返回，说明所有字段都不存在
    # 构造清晰的错误信息
    error_message = f"extra_info 中不包含任何指定的字段。期望的字段为: {', '.join(keys_to_check)}"
    raise ValueError(error_message)

def compute_score(data_source:str, solution_str:str, ground_truth, extra_info=None, timeout_score=0.0) -> bool:
    rewards = []
    # print("begin to ")
    # print(f"extra_info:{extra_info}")
    prompt_str = extract_prompt_from_extra_info(extra_info)
    # print(f"prompt:{prompt_str}")
    reviewer = "Qwen3-14B"
    reviews = 1
    workers = 4

    
    # print("begin to eval")
    result = peval_pipeline([prompt_str], [solution_str], reviewer, reviews, workers)
    # print(f"prompt_str:{prompt_str}")
    # print(f"solution_str:{solution_str}")
    # print(f"ground_truth:{ground_truth}")
    # print(f"reviews:{result[0]['review']}")
    rewards = [p['judgement'] if p['judgement'] else 0.0 for p in result] 
    
    # if len(rewards) > 1:
    #     # 3. 抛出 ValueError，并提供清晰的错误信息
    #     raise ValueError(f"Reward 的长度不能大于 1。当前长度为: {len(reward)}")
    true_score = verify(data_source, solution_str, ground_truth)
    score = rewards[0]
    reward = {}
    if score is None:
        score = 0.0
    reward["true_score"]=true_score
    reward["score"]=score
    reward["TP"]=0
    reward["FP"]=0
    reward["TN"]=0
    reward["FN"]=0
    if score  >= 7.0 and true_score == 1.0:
        reward["TP"] += 1
    elif score >= 7.0 and true_score == 0.0:
        reward["FP"] += 1
        import json
        import os
        from datetime import datetime
        
        # 构建要保存的数据
        fp_data = {
            "timestamp": datetime.now().isoformat(),  # 添加时间戳
            "prompt_str": prompt_str,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "score": score,
            "reviews": result[0]['review']
        }
        
        try:
            # 确保目录存在
            fp_dir = os.path.join(WORK_DIR, "verl/ProofRL/FP")
            os.makedirs(fp_dir, exist_ok=True)
            
            # 文件路径
            fp_file = os.path.join(fp_dir, "false_positive_instances.json")
            # 保存到JSON文件
            with open(fp_file, "a", encoding="utf-8") as f:
                json.dump(fp_data, f, ensure_ascii=False, indent=2)
                f.write(",\n")  # 添加换行符分隔不同实例
            print(f"False Positive instance saved to JSON!")
        except Exception as e:
            print(f"Failed to save FP instance: {e}")
            print(f"There exists False Positive instance!")
            print(f"prompt_str:{prompt_str}")
            print(f"solution_str:{solution_str}")
            print(f"ground_truth:{ground_truth}")
            print(f"reviews:{result[0]['review']}")
    elif score <= 1.0 and true_score == 1.0:
        reward["FN"]+=1
        import json
        import os
        from datetime import datetime
        
        # 构建要保存的数据
        fp_data = {
            "timestamp": datetime.now().isoformat(),  # 添加时间戳
            "prompt_str": prompt_str,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "score": score,
            "reviews": result[0]['review']
        }
        
        try:
            # 确保目录存在
            fp_dir = os.path.join(WORK_DIR, "verl/ProofRL/FP")
            os.makedirs(fp_dir, exist_ok=True)
            # 文件路径
            fp_file = os.path.join(fp_dir, "false_negative_instances.json")
            # 保存到JSON文件
            with open(fp_file, "a", encoding="utf-8") as f:
                json.dump(fp_data, f, ensure_ascii=False, indent=2)
                f.write(",\n")  # 添加换行符分隔不同实例
            print(f"False Negative instance saved to JSON!")
        except Exception as e:
            print(f"Failed to save FP instance: {e}")
            print(f"There exists False Negative instance!")
            print(f"prompt_str:{prompt_str}")
            print(f"solution_str:{solution_str}")
            print(f"ground_truth:{ground_truth}")
            print(f"reviews:{result[0]['review']}")
    elif score <= 1.0 and true_score == 0.0:
        reward["TN"]+=1
    if score == 3.0:
        print(f"Invalid score: {score}. Expected 0.0, 1.0, 9.0, or 10.0.")
        print(f"prompt_str:{prompt_str}")
        print(f"solution_str:{solution_str}")
        print(f"ground_truth:{ground_truth}")
        print(f"reviews:{result[0]['review']}")
    return reward
if __name__ == "__main__":
    with open("/mnt/workspace/linzejin/verl/ProofRL/example/false_positive_instances.json", "r", encoding="utf-8") as f:
        datas = json.load(f)
    statics = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "true_score": 0.0, "total_num": 0, "score": 0.0}
    for data in datas:
        score=compute_score("lighteval/MATH",data["solution_str"], data["ground_truth"], {"prompt":data["prompt_str"]})
        for k in score:
            statics[k] += score[k]
        statics["total_num"] += 1
        print(f"score:{score}")
        print(f"statics:{statics}")
        print(f"FN rate:{statics['FN']/statics['total_num']}")
        print(f"FP rate:{statics['FP']/statics['total_num']}")




