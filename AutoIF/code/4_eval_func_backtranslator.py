import json

import jsonlines
import json
import random
import re
import os
import copy
import nltk
import numpy as np
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError
)
import logging
import signal
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import tenacity
from utils import chat_completion_openai
import ast

random.seed(0)


out_dir=os.environ['OUTPUT_DIR']



filter_results = []
count = 0 
filter_count =0

out=open(f'{out_dir}/4_backtranslate_prompt.jsonl', 'w')
with open(f'{out_dir}/3_cross_validation.jsonl', 'r') as file:
    for i,line in enumerate(tqdm(file)):
        line = json.loads(line)
        funcs = line["eval_func"] #[:6]
        # import pdb
        # pdb.set_trace()


        
      
        instruction = f"""You are an expert in converting the Python eval function code into the corresponding instruction text. I will provide the eval function code. Please strictly follow the code to convert it into the corresponding instruction text. Here's an example: \n\n[["def evaluate(response):\n    return 'e' not in response.lower()", 1.0], ["def evaluate(response):\n    words = response.split()\n    for word in words:\n        if 'e' in word.lower():\n            return False\n    return True", 1.0], ["def evaluate(response):\n    return all('e' not in word.lower() for word in response.split())", 1.0]] \n\n["Answer without using any words that contain the letter 'E'.","Answer with words that do not contain the letter 'E'.","Answer with words that do not contain the letter 'E'."] Please convert the following eval function into instructions stored in a list: \n\n{funcs}"""

        '''
        Plead Add code:

        Using Supervison Model for back translation
        back_instruction --> str :  the back translation instruction for each vertification function
        '''
        # back_instruction='["Answer using only words that start with the letter 'B'.", "Answer with words that begin with the letter 'B'.", "Answer with words that start with the letter 'B'."]'
        # back_instruction = chat_completion_openai(messages=[{"role": "user", "content": instruction}])
        messages=[{"role": "user", "content": instruction}]
        r={'model': os.environ['OPENAI_MODEL'], 'temperature': 0.7, 'max_tokens': 1024, 
            'messages': messages, 'metadata': {'instruction': line['instruction'], 'eval_func': line['eval_func'], 'cases': line['cases']}}
        out.write(json.dumps(r, ensure_ascii=False)+'\n')
#         try:
#             # def extract_list(s):
#             #     match = re.search('\[.*\]', s)
#             #     if match:
#             #         return ast.literal_eval(match.group())
#             #     else:
#             #         print('wrong: ', s)
#             #         return None
#             def extract_list(s):
#                 start = s.find('[')
#                 end = s.rfind(']') + 1
#                 list_str = s[start:end]
#                 return ast.literal_eval(list_str)
            
#             back_instruction = extract_list(back_instruction)
#             print(back_instruction)
#             assert len(back_instruction) == 3
#         except Exception as e:
#             print(back_instruction, e)
#             filter_count+=1
#             continue

            
#         line["back_instruction"] = back_instruction
        
#         filter_results.append(line)
            
#         count+=1



# print("filter_count",filter_count)

# with jsonlines.open("./output/back_trans.jsonl", "w") as f:
#     for each in filter_results:
#         f.write(each)

'''the example of output format is in ./sample_data/back_trans.jsonl'''

