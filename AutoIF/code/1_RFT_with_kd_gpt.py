from utils import chat_completion_openai
import random
import sys
from api_request_parallel_processor import process_api_requests_from_list
import json
import os


if __name__ == '__main__':
    K=400
    if len(sys.argv)>2:
        inp=sys.argv[1]
        out=sys.argv[2]

    random.seed(0)

    # cd ./AutoIF/code

    seed_instructions = [each.strip() for each in open("./sample_data/seed_ifeval.txt").readlines()]
    # seed_instructions = [each.strip() for each in open("./sample_data/seed_instruction.txt").readlines()]

    augment_instruction_prompt = """You are an expert for writing instructions. Please provide 50 different instructions that meet the following requirements:
    - Instructions are about the format but not style of a response
    - Whether instructions are followed can be easily evaluate by a Python function
    Here are some examples of instructions we need:
    {seed_instructions}
    Do not generate instructions about writing style, using metaphor, or translation. Here are some examples of instructions we DO NOT need:
    - Incorporate a famous historical quote seamlessly into your answer
    - Translate your answer into Pig Latin
    - Use only words that are also a type of food
    - Respond with a metaphor in every sentence
    - Write the response as if you are a character from a Shakespearean play
    Please generate one instruction per line in your response and start each line with '- '. Be creative, DO NOT repeat the examples provided.
    """


    augment_instructions = augment_instruction_prompt.format(seed_instructions='\n'.join(seed_instructions))

    print(augment_instructions, flush=True)

    messages = [{"role": "user", "content": augment_instructions}]

    # results=seed_instructions
    results=[]
    requests_list=[{'model': os.environ['OPENAI_MODEL'], 'temperature': 0.7, 'max_tokens': 1024, 'messages': messages}]*K
    # for k in range(K):
    # response = process_api_requests_from_list(requests_list=requests_list, save_filepath='./output/1rft.jsonl', max_requests_per_minute=200, max_tokens_per_minute=1600, )
    # print(response)
    # for r in response.split('\n'):
        # r=r.strip('- \t')
        # results.append(r)

    # print(results)
    with open(out, "w") as f:
        for r in requests_list:
            json_string = json.dumps(r, ensure_ascii=False)
            f.write(json_string + "\n")