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

with open('/data3/private/linzejin/nev/.apiconfig.json', 'r', encoding='utf-8') as file:
    apiconfig = json.load(file)

client = openai.OpenAI(
    api_key = apiconfig['OPENAI_API_KEY'],
    base_url = apiconfig['OPENAI_BASE_URL']
)



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

    def __init__(self, model: str):
        self.model = model
        self.remote = True if model in AgentBase.remote_models else False
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
                stream = client.chat.completions.create(**client_params)
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
        print("processing prompts")
        if self.remote:
            for prompt in prompts:
                results = self.call_from_remote_API(prompt),
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
             'You are a reviewer for this math problem. You are provided a candidate proof of this problem, and you need to analyze and point out one part of this proof that might be incomplete or contain some flaws. We will use your advice for further judgement of this proof.\n'
             '\n'
             '### Problem\n'
             '\n'
             f'{problem}\n'
             '\n'
             '### Candidate Proof\n'
             '\n'
             f'{proof}\n'
             'You need to explain your rationales and decide whether this candidate can be accepted as a valid proof of this problem. State your judgement inside $\\boxed{}$ as $\\boxed{true}$ or $\\boxed{false}$ at the end of your response.\n'
             }]
        return prompt
