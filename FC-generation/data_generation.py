import os
os.environ['OPENAI_API_KEY'] = ''

from openai import OpenAI
from datasets import load_dataset
import json
import jsonlines
import numpy as np
from tqdm import tqdm
import time

hf_apis = []
with jsonlines.open('gorilla/data/api/huggingface_api.jsonl', 'r') as f:
    for line in f:
        hf_apis.append(line)
torch_apis = []
with jsonlines.open('gorilla/data/api/torchhub_api.jsonl', 'r') as f:
    for line in f:
        torch_apis.append(line)
tf_apis = []
with jsonlines.open('gorilla/data/api/tensorflowhub_api.jsonl', 'r') as f:
    for line in f:
        tf_apis.append(line)

examples_per_hub = 6
hf_examples = []
with jsonlines.open('gorilla/data/apibench/huggingface_eval.json', 'r') as f:
    for i, line in enumerate(f):
        hf_examples.append(line)
        if i+1 == examples_per_hub:
            break
torch_examples = []
with jsonlines.open('gorilla/data/apibench/torchhub_eval.json', 'r') as f:
    for i, line in enumerate(f):
        torch_examples.append(line)
        if i+1 == examples_per_hub:
            break
tf_examples = []
with jsonlines.open('gorilla/data/apibench/tensorflow_eval.json', 'r') as f:
    for i, line in enumerate(f):
        tf_examples.append(line)
        if i+1 == examples_per_hub:
            break

def process_example(item):
    code = item['code']
    instruction = code.split('###Output:')[0].lstrip('###Instruction:').strip()
    ex_co = code.split('<<<explanation>>>:')[1].strip()
    explanation, code = ex_co.split('<<<code>>>:')
    return {'instruction': instruction, 'output': {'explanation': explanation.strip(), 'code': code.strip()}, 'api_data': item['api_data']}
def process_example_torch(item):
    code = item['code']
    instruction = code.split('###Output:')[0].lstrip('###Instruction:').strip()
    ex_co = code.split("'explanation':")[1].strip()
    explanation, code = ex_co.split("'code':")
    return {'instruction': instruction, 'output': {'explanation': explanation.strip().rstrip(',').strip("'").strip(), 'code': code.strip().rstrip('}').strip("'").strip()}, 'api_data': item['api_data']}

hf_examples = [process_example(item) for item in hf_examples]
torch_examples = [process_example_torch(item) for item in torch_examples]
tf_examples = [process_example(item) for item in tf_examples]

with jsonlines.open('data/autoGen/examples.json', 'w') as f:
    for hf_ex in hf_examples:
        f.write(hf_ex)
with jsonlines.open('data/autoGen/examples.json', 'a') as f:
    for torch_ex in torch_examples:
        f.write(torch_ex)
with jsonlines.open('data/autoGen/examples.json', 'a') as f:
    for tf_ex in tf_examples:
        f.write(tf_ex)
all_examples = hf_examples + torch_examples + tf_examples
# all_examples = hf_examples

# instruction, output: domain, api_call, api_provider, explanation, code

format_inst = 'The output MUST strictly adhere to the following JSON format,\
and NO other text MUST be included:\n\
```\
[\n\
    {\n\
        "instruction": "The generated instruction.",\n\
        "output": {\n\
            "code": "code using the api to complete the instruction",\n\
            "explanation": "explanation of the code"\n\
        },\n\
    }\n\
]\n\
```'

def format_gen_prompt(api, num_examples=3, gen_number=2):
    examples = np.random.choice(all_examples, num_examples, replace=False)
    generate_prompt = f"You are a data labeler. The responsibility for you is to generate a set of diverse instructions and corresponding outputs for the given functions in JSON format.\nConstruct instructions and outputs that exemplifies how to use these functions in a practical scenario. Include in each instruction specific, plausible values for each parameter. For instance, if the function requires a date, use a typical and reasonable date.\n\
Ensure the instruction:\n- Is clear and concise\n\
- Contain multiple parallel instructions in natural language for\
the given functions, they could use either the same\
function with different arguments or different functions\n\
- Demonstrates typical use cases\n\
- Includes all necessary parameters in a meaningful way. For\
numerical parameters, it could be either numerals or words\n\
- Across a variety level of difficulties, ranging from\
beginner and advanced use cases\n\
- The corresponding result's parameter types and ranges match\
with the functions descriptions.\n\
Ensure the output:\
- Is a list of function calls in JSON format.\n\
- The length of the answer list should be equal to the number\
of requests in the instruction\n\
- Can solve all the requests in the instruction effectively\n\
Here are examples of instructions and corresponding outputs for\
similar functions:\n\
{examples}\n\
Note that the instruction could be interpreted as a combination of several independent requests.\n\
Based on these examples and the above instructions, generate \
{gen_number} diverse instruction and output pairs for the functions \
`{api['api_name']}`. Make sure the generated code in output includes the 'api_call' part in the api.\n\
The detailed functions description is as follows:\n\
{api}\n\n\
{format_inst}\n\n\
Now please generate {gen_number} diverse instruction and answer pairs following the above format."

    return generate_prompt

print_every = 20
gens = []
start = 0
stop = 10000
client = OpenAI()

f = jsonlines.open('data/autoGen/gen-bfcl-2.json', 'w')

for i, hf_api in enumerate(tqdm(hf_apis[start:])):
    gen_prompt = format_gen_prompt(hf_api, gen_number=10)

    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": gen_prompt
                    }
                ]
            )
            break
        except:
            print('Wait 2s for GPT')
            time.sleep(2)

    if (i+1) % print_every == 0:
        print(completion.choices[0].message.content)
    try:
        gen = json.loads(completion.choices[0].message.content.strip('```').strip('json').strip())
    except:
        gen = []
    for j in range(len(gen)):
        gen[j]['api_data'] = hf_api
    gens.append(gen)
    f.write(gen)
    if i >= stop:
        break

f.close()

# bfcl api check
filtered_gens = []
api_call_not_in = 0
total = 0
for i, gen in enumerate(gens):
    # if i >= 30:
    #     break
    tmp = []
    for item in gen:
        total += 1
        if (hf_apis[i]['api_name'] in ittem['output']['code']) and (hf_apis[i]['api_call'].split('(')[0] in item['output']['code']):
            tmp.append(item)
            continue
        if (hf_apis[i]['api_call'].replace(hf_apis[i]['api_name'], 'model_id') in item['output']['code']):
            tmp.append(item)
            continue
        if (hf_apis[i]['api_call'].replace(hf_apis[i]['api_name'], 'model_name') in item['output']['code']):
            tmp.append(item)
            continue 
        if hf_apis[i]['api_call'] in item['output']['code']:
            tmp.append(item)
            continue
        api_call_not_in += 1
        print(f"{hf_apis[i]['api_call']} not in {item['output']['code']}")
        print(hf_apis[i]['api_name'])
    filtered_gens.append(tmp)
print(f'api_call not in: {api_call_not_in/total}, {api_call_not_in}/{total}')

with jsonlines.open('data/autoGen/gen-bfcl-filtered.json', 'w') as f:
    for gen in filtered_gens:
        f.write(gen)