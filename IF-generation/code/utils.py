import openai
import time
from openai import OpenAI, AsyncOpenAI
import httpx
import sys
import random
import asyncio


API_RETRY_SLEEP = 3
API_MAX_RETRY = 100


def chat_completion_openai(model='gpt-4o-mini', message=None, temperature=0.7, max_tokens=1024, api_dict=None):
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
        
    KEYS=['sk-GiLhe8X3HvcVL60R23Bc2464Bd2d46A6A54898AfD10f5e2a',
          'sk-4xuCZWKqpz0ZX2Tb5f5bE78f197544CcB56f3661227418F4', 
          'sk-8jXpCLspBVxzRhvvCd612eAcCb044cC7Bd0f5597E1A54604',
          'sk-KhKTIF5KqvNcOmppC3E5D736Bf5148259094CbC07e557b1d',
          'sk-GYqGExMOTtfOdsDI6305B00dF7C44f4293E5De03Eb7c6f40',
    ]
    output = 'Error in quering openai!'
    key=KEYS[0]
    # print(message)
    for _ in range(API_MAX_RETRY):
        try:
            client = OpenAI(
                api_key='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',
            )
            # messages = conv.to_openai_api_messages()
            response = client.chat.completions.create(
                model=model,
                messages=message,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # output = response["choices"][0]["message"]["content"]
            output = response.choices[0].message.content
            break
        except Exception as e:
            print(type(e), e, flush=True)
            # if isinstance(e, openai.PermissionDeniedError):
                # print(e.body['code'])
            if isinstance(e, openai.PermissionDeniedError) and e.body['code'] == 'pre_consume_token_quota_failed':
                print(f'NO QUOTA on {key}, CHANGE TO NEXT KEY', flush=True)
                key=KEYS[(_+1)%len(KEYS)]
            time.sleep(API_RETRY_SLEEP)

    return output


def a_chat_completion_openai(model='gpt-4o-mini', message=None, temperature=0.7, max_tokens=1024, api_dict=None):
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
        
    client = AsyncOpenAI(
        api_key='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',
    )

    # print(model, message, temperature, max_tokens)
    cor = client.chat.completions.create(
        model=model,
        messages=message,
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return cor


async def chat_completion(message: dict, model, temperature, max_tokens, api_dict):
    return await a_chat_completion_openai(model=model, message=message, temperature=temperature, max_tokens=max_tokens, api_dict=api_dict)

async def rate_limited_chat_completion(message: dict, sem: asyncio.Semaphore, model, temperature, max_tokens, api_dict):
    async with sem:  
        return await chat_completion(message, model, temperature, max_tokens, api_dict)

async def rate_limited_as_completed(sem: asyncio.Semaphore, Messages, model='gpt-4o-mini', temperature=0.7, max_tokens=1024, api_dict=None):
    results = []
    tasks_get_persons = [rate_limited_chat_completion(message, sem, model, temperature, max_tokens, api_dict) for message in Messages]
    for person in asyncio.as_completed(tasks_get_persons):
        try:
            results.append(await person) 
        except Exception as e:
            print(e)

def async_chat_completion_openai(model='gpt-4o-mini', Messages=None, temperature=0.7, max_tokens=1024, api_dict=None, RateLimit=10):
    
    sem = asyncio.Semaphore(RateLimit)

    responses = asyncio.run(rate_limited_as_completed(sem, Messages, model, temperature, max_tokens, api_dict))

    # print(responses)
    return responses

if __name__ == '__main__':

    Messages=[
        [{"role": "user", "content": 'Hi!'}],
        [{"role": "user", "content": 'Hi!'}],
        [{"role": "user", "content": 'Hi!'}],
        [{"role": "user", "content": 'Hi!'}],
        [{"role": "user", "content": 'Hi!'}],
        [{"role": "user", "content": 'Hi!'}],
        [{"role": "user", "content": 'Hi!'}],
        [{"role": "user", "content": 'Hi!'}],
        [{"role": "user", "content": 'Hi!'}],
    ]

    res=async_chat_completion_openai(Messages=Messages)
    # res=chat_completion_openai(message=Messages[0])

    print(res)