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

# out=open(f'{out_dir}/4_backtranslate_prompt.jsonl', 'w')
with open(f'{out_dir}/4_backtranslate_prompt_results.jsonl', 'r') as file:
    for i,line in enumerate(tqdm(file)):
        line = json.loads(line)
        # import pdb
        # pdb.set_trace()


        
      
        instruction = line[0]['messages'][0]["content"]

        '''
        Plead Add code:

        Using Supervison Model for back translation
        back_instruction --> str :  the back translation instruction for each vertification function
        '''
        # back_instruction='["Answer using only words that start with the letter 'B'.", "Answer with words that begin with the letter 'B'.", "Answer with words that start with the letter 'B'."]'
        back_instruction = line[1]["choices"][0]["message"]["content"]
        try:
            def extract_list(s):
                start = s.find('[')
                end = s.rfind(']') + 1
                list_str = s[start:end]
                return ast.literal_eval(list_str)
            
            back_instruction = extract_list(back_instruction)
            # print(back_instruction)
            # print(len(back_instruction), len(line[2]['eval_func']))
            assert len(back_instruction) ==  len(line[2]['eval_func'])
        except Exception as e:
            # print(back_instruction, e)
            filter_count+=1
            continue

        res={
            'instruction': line[2]['instruction'],
            'back_instruction': [b[0] for b in back_instruction] if isinstance(back_instruction[0], list) else back_instruction,
            'eval_func': line[2]['eval_func'],
            'cases': line[2]['cases']
            }
        # line["back_instruction"] = back_instruction
        
        filter_results.append(res)
            
        count+=1



print("filter_count",filter_count)

with open(f"{out_dir}/4_back_trans.jsonl", "w") as f:
    for each in filter_results:
        f.write(json.dumps(each, ensure_ascii=False)+'\n')

'''the example of output format is in ./sample_data/back_trans.jsonl'''

