This code base is built on [Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models](https://github.com/QwenLM/AutoIF)

## To run the generation:
```bash
pip install -r requirements.txt
# set the OPENAI_API_KEY, OPENAI_MODEL, OUTPUT_DIR in run.sh
./run.sh
```

## Description of each step:
1. Using [seed data](./sample_data/seed_ifeval.txt) as examples to prompt GPT generates similar instructions. These generated instructions plus seed instructions serve as python-verifiable constraints for later specific query.
2. Prompt GPT to generate K evaluation python functions and input cases for each instruction in Step 1. 
3. Cross validate the functions in Step 2. Keep the high-quality ones.
4. Prompt GPT to translate the filter instructions in Step 3. back to natural language instructions (constraints).
5. Call the [NLI model](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) to test whether the origin instruction in Step 1 contradicts with the translated-back instruction. Discard instructions that have contradiction.
6. For each instruction in Step 5, randomly pick M user queries from ShareGPT, then concat the inctruction and one user query using a prompt template to form a new user query with constraints. Then prompt GPT to generate response for each new user query.
7. For each input-response pair in Step 6 Call the previous obtained python functions to verify if it satisfies the constraint. Discard those not satisfy.
8. Prompt GPT to score the input-response pair in Step 7 in both input and response. Keep only high quality input-response pairs.
9. Format the input-response pairs in Alpace format.

## Hyperparameter in each step
1. K in ./code/1_RFT_with_kd_gpt.py determines the instructions (constraints) we will get. Roughly it will be K*50 before deduplication.
2. We generate 5 evaluation python functions for each constraints in Step 1. This value should be no less than 3.
3. No hyperparameter in this step.
4. No hyperparameter in this step.
5. No hyperparameter in this step.
6. Each instruction is used 8 times to concat ShareGPT user queries. For each catenated input, we pormpt GPT to generate 2 response. Large value will incur more duplicate and less diversity.
7. No hyperparameter in this step.
8. We set the threshold of score filtering to 8.
9. No hyperparameter in this step.

## A real case to see the size of data changes during generation steps.
We use gpt-4o-mini model.
1. We set K=400 and obtain 16928 instruction after deduplication.
3. After cross validation, we have 14579 instructions left.
5. After applying the NLI model, we have 13077 instructions left.
7. After concat filter, we have 122932 input-response pairs left.
9. Finally after score filter we get 61492 input-response pairs for fine-tuning.

Example data in each steps of generation [here](./generation_example)