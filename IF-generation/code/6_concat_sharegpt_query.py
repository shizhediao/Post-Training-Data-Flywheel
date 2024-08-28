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


random.seed(0)

out_dir=os.environ['OUTPUT_DIR']



filter_results=[]

with jsonlines.open(f"{out_dir}/5_back_trans_fliter.jsonl", "r") as f:
    for each in f:
        filter_results.append(each)

sft_data = json.load(open('/home/panxingyuan/minitron-data/AutoIF/AutoIF-main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry_cleaned_reduce.json'))
# queries = [each['messages'][1]['content'] for each in sft_data if each['source'] == 'en:sharegpt']
queries = [r['conversations'][0]['value'] for r in sft_data if r['conversations'][0]['from'] == 'human']

queries = [each for each in queries if len(each) > 20 and len(each) < 300]

print(len(queries))

inputs = []
for instruction in tqdm(filter_results):
    ins_queries = random.sample(queries, 8) #拼16个
    for q in ins_queries:
        prompt = f"Please answer the query strictly following the instruction.\n[instruction] {instruction['instruction']}\n[Query] {q}"
        item = copy.deepcopy(instruction)
        item['prompt'] = prompt
        inputs.append(item)
        # import pdb
        # pdb.set_trace()


inp=[]
for i in inputs:
    messages=[{"role": "user", "content": i['prompt']}]
    r={'model': os.environ['OPENAI_MODEL'], 'temperature': 0.7, 'max_tokens': 1024, 'n': 2, 'messages': messages, 
        'metadata': {'instruction': i['instruction'], 'back_instruction': i['back_instruction'], 'eval_func': i['eval_func'], 'nli_scores': i['nli_scores'], 'cases': i['cases']}}
    inp.append(r)

with open(f"{out_dir}/6_instruction_filtered_query_prompt.jsonl", "w") as f:
    for each in inp:
        f.write(json.dumps(each, ensure_ascii=False)+'\n')




'''
Please TODO:

Please use supervision model perform RFT to generate k Responses for each query
'''