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

out_dir=os.environ['OUTPUT_DIR']

random.seed(0)


# test gpt4

os.environ['NLTK_DATA'] = 'your nltk_data data path'
logging.getLogger('nltk').setLevel(logging.CRITICAL)
from nltk import data
data.path.append('your nltk_data data path')

path=f"{out_dir}/2_eval_func_rft_prompts_results.jsonl"


results_ = list(jsonlines.open(path))
results = []
for r in results_:
    results.append(
        {
            'instruction': r[2]['instruction'],
            'gpt-answer': [x["message"]["content"] for x in r[1]["choices"]]
        }
    )
# print(results)

all_instruction_set=set(r['instruction'] for r in results)

print("Preprocess vertification functions")

def col_formatter(s):
    start = s.find("def evaluate(response):")
    end = s.find("return", start)
    
    if start == -1 or end == -1:
        return s  # if either "asfdsfer" or "return" is not found, return the original string
    
    # Extract the part of the string that needs modification
    part_to_modify = s[start:end]
    
    # Replace '\n' with '\\n' in the extracted part
    modified_part = part_to_modify.replace('\n', '\\n')
    
    # Reconstruct the final string
    final_string = s[:start] + modified_part + s[end:]
    
    return final_string

collect_packages = []
filter_json = 0
for result in results:
    res = result['gpt-answer']
    eval_funcs, test_cases = [], []
    for each in res:
        try:
            json_dict = re.findall(r'```json(.*?)```', each, re.DOTALL)[0].strip(' \n')# .replace('\n', '\\n')
        except IndexError:
            continue
    
    # func rejection
    try:
        json_dict=col_formatter(json_dict)
        res_dict = json.loads(json_dict, strict=False)
    except Exception as e:
        print(f'error: {e}, ', json_dict)
        filter_json += 1
        continue
        
    func = res_dict['func']
    
    if '\\n' in func:
        func = func.replace('\\n', '\n')
    try:
        exec(func)
    except Exception:
        continue
    
    for line in func.split('\n'):
        if 'import' in line or 'download' in line or 'requests' in line:
            collect_packages.append(line)
print(list(set(collect_packages)))
print(f'filtered because of wrong json format: {filter_json}, all: {len(results)}')




print("cross validation for functions and cases")

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

filter_results = []
filter_count_exec=0
filter_count_func_case=0
for result in results:
    res = result['gpt-answer']
    eval_funcs, test_cases = [], []
    for each in res:
        try:
            json_dict = re.findall(r'```json(.*?)```', each, re.DOTALL)[0].strip()
        except IndexError:
            continue

        try:
            json_dict=col_formatter(json_dict)
            res_dict = json.loads(json_dict)
        except json.JSONDecodeError:
            continue

        # func rejection
        func = res_dict['func']
        func = func.strip()
        func = '\n'.join([each for each in func.split('\n') if 'download' not in each and 'requests' not in each])
        try:
            exec(func)
        except Exception:
            continue
        eval_funcs.append(func)
        if 'cases' not in res_dict.keys():
            print(f'No cases found for {res_dict}')
            continue
        for each in res_dict['cases']:
            try:
                test_cases.append((each['input'], each['output']))
            except KeyError:
                print(each)
    eval_funcs = list(eval_funcs)
    # eval_funcs = list(set(eval_funcs))
    test_cases = list(map(json.loads, map(json.dumps, test_cases)))
    # test_cases = list(map(json.loads, set(map(json.dumps, test_cases))))
    print(len(eval_funcs), len(test_cases))

    if len(eval_funcs) < 3 or len(test_cases) < 10:
        filter_count_func_case += 1
        continue

    filtered_test_cases = []

    for each in test_cases:
  

        flag = False
        for func in eval_funcs:
            local_vars = {}
 
            try:
                exec(func, globals(), local_vars)
            except Exception:
                filter_count_exec+=1
                continue
            
            if 'evaluate' not in local_vars:
                filter_count_exec+=1
                continue
            eval_func = local_vars['evaluate']
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                res = eval_func(each[0])
            except Exception:
                res = None
            finally:
                signal.alarm(0)
            if res is not None and res == each[1]:
                flag = True
        if flag:
            filtered_test_cases.append(each)

    scored_funcs = []
    for func in eval_funcs:
        local_vars = {}
        try:
            exec(func, globals(), local_vars)
        except Exception:
            continue
        if 'evaluate' not in local_vars:
            continue

        eval_func = local_vars['evaluate']
        acc = []
        for inp, out in filtered_test_cases:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                res = eval_func(inp)
            except Exception:
                res = None
            finally:
                signal.alarm(0)
            if res is None or res != out:
                acc.append(0)
            else:
                acc.append(1)
        acc = np.mean(acc) if acc else 0
        scored_funcs.append((func, acc))

    valid_funcs = [each for each in scored_funcs if each[1] >= 0.8]
    if not valid_funcs:
        continue

    filter_results.append({
        "instruction": result['instruction'],
        "eval_func": valid_funcs,
        "cases": filtered_test_cases
    })

print(f'filter_results: {len(filter_results)}')
print(f'filter_count_func_case: {filter_count_func_case}')
filtered_instruction=all_instruction_set - set(r['instruction'] for r in filter_results)

print("finish!!!")
print(f'filtered instructions: {filtered_instruction}, {len(filtered_instruction)}')

with jsonlines.open(f"{out_dir}/3_cross_validation.jsonl", "w") as f:
    for each in filter_results:
        f.write(each)