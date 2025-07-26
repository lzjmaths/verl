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

with open('/data3/private/linzejin/nev/.apiconfig.json', 'r', encoding='utf-8') as file:
    apiconfig = json.load(file)





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
    if "true" in a:
        return "true"
    elif "false" in a:
        return "false"
    else:
        return "true"

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
             'You are a reviewer for this math problem. You are provided a candidate solution of this problem, and you need to analyze whether the answer is rigorous and correct. We will use your advice for further judgement of this proof.\n'
             '\n'
             '### Problem\n'
             '\n'
             f'{problem}\n'
             '\n'
             '### Candidate Proof\n'
             '\n'
             f'{proof}\n'
             'You need to explain your rationales and decide whether this candidate can be accepted as a valid solution of this problem. State your judgement inside $\\boxed{}$ as $\\boxed{true}$ or $\\boxed{false}$ at the end of your response.\n'
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
        solved = True
        for j in range(reviews):
            if extract_boxed(raw_reviews[i * reviews + j]) == "false":
                results.append({
                    'problem': problems[i],
                    'proof': proofs[i],
                    'review': remove_think_tags(raw_reviews[i * reviews + j]),
                    'judgement': False
                })
                solved = False
                break
            else:
                continue
        if solved:
            results.append({
                'problem': problems[i],
                'proof': proofs[i],
                'review': remove_think_tags(raw_reviews[i * reviews]),
                'judgement': True
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
    # reviewer = "/data2/private/linzejin/models/Qwen/Qwen3-14B"  # 默认值
    # reviewer = "deepseek-r1"
    reviews = 1
    workers = 4
    # try:
    #     if isinstance(extra_info, dict):
    #         # 如果 'model' 键不存在，.get() 会返回 None
    #         model_value = extra_info.get("model")
    #         if model_value is not None:
    #             reviewer = str(model_value) # 确保转换为字符串类型
    #         reviews_value = extra_info.get("reviews")
    #         if reviews_value is not None:
    #             reviews = int(reviews_value)
    #         workers_value = extra_info.get("worker_value")
    #         if workers_value is not None:
    #             workers = int(workers_value)
    # except Exception as e:
    #     print(e)

    
    # print("begin to eval")
    result = peval_pipeline([prompt_str], [solution_str], reviewer, reviews, workers)
    # print(f"prompt_str:{prompt_str}")
    # print(f"solution_str:{solution_str}")
    # print(f"ground_truth:{ground_truth}")
    # print(f"reviews:{result[0]['review']}")
    rewards = [1.0 if p['judgement'] else 0.0 for p in result] 
    
    # if len(rewards) > 1:
    #     # 3. 抛出 ValueError，并提供清晰的错误信息
    #     raise ValueError(f"Reward 的长度不能大于 1。当前长度为: {len(reward)}")
    true_score = verify(data_source, solution_str, ground_truth)
    score = rewards[0]
    reward = {}
    reward["score"]=score
    reward["true_score"]=true_score
    reward["TP"]=0
    reward["FP"]=0
    reward["TN"]=0
    reward["FN"]=0
    if score == 1.0 and true_score == 1.0:
        reward["TP"]+=1
    elif score == 1.0 and true_score ==0.0:
        reward["FP"]+=1
        print(f"There exists False Positive instance!")
        print(f"prompt_str:{prompt_str}")
        print(f"solution_str:{solution_str}")
        print(f"ground_truth:{ground_truth}")
        print(f"reviews:{result[0]['review']}")
    elif score == 0.0 and true_score == 1.0:
        reward["FN"]+=1
    elif score == 0.0 and true_score == 0.0:
        reward["FP"]+=1
    return reward
if __name__ == "__main__":
    print(compute_score("lighteval/MATH","\\boxed{1+1=2}", "1+1=2", {"prompt":"prove 1+1=2"}))



