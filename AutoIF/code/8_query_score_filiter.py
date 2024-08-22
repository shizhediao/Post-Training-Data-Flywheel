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


results_ = list(jsonlines.open(f"{out_dir}/7_query_need_quality_score_prompt_results.jsonl"))
results = []

for r in results_:
    results.append({
        'instruction': r[2]['instruction'],
        'query': r[2]['query'],
        'response': r[2]['response'],
        'prompt': r[0]['messages'][0]['content'],
        'gen': [x['message']['content'] for x in r[1]['choices']]
    })


# results = list(jsonlines.open("./sample_data/query_rft_score.jsonl"))
filter_results = []
print(len(results))
for result in tqdm(results):
    scores = []
    for each in result['gen']:
        score = re.findall(r'Score: (\d+?)$', each)
        if score:
            scores.append(int(score[0]))
    score = np.mean(scores) if scores else 0
    if score > 8: # quality score
        filter_results.append(result)
print(len(filter_results))



with jsonlines.open(f"{out_dir}/8_query_score_filter.jsonl", "w") as f:
    for each in filter_results:
        f.write(each)
