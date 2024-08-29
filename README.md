# Post-Training-Data-Flywheel

<p align="center">
<img src="./assets/flywheel.jpeg"  width="256">
</p>

## Goal
We aim to provide the best references to search, select, and synthesize high-quality and large-quantity data for post-training your LLMs.

## Introduction

There are three contributions in this repository:
1. Data Generation: We provide the data generation process for two important domains: instruction following and function calling.
2. Dataset Compilation: We collected and compiled a list of high-quality datasets for post-training LLMs in the domains of instruction following, coding, and math. We provide a quality check for the datasets.
3. Dataset Curation: According to the quality check, we carefully curated a new dataset for post-training LLMs. The datasets are carefully collected and evaluated to ensure high quality and relevance for post-training.


## Download
You can download the datasets directly from the [Hugging Face Hub](https://huggingface.co/Post-training-Data-Flywheel).
There are two versions:
1. [Flywheel-v1](https://huggingface.co/datasets/Post-training-Data-Flywheel/flywheel-v1): A small and highly curated datasets.
2. [Flywheel-v2](https://huggingface.co/datasets/Post-training-Data-Flywheel/flywheel-v2): A large and diverse datasets. (recommended)

## Data Generation
We provide the data generation process for two important domains: [instruction following](./IF-generation/) and [function calling](./FC-generation/).


## Quality Check
- Domain: we are only concerned about the following tasks: instruction following, coding, and math. Datasets other than those in English are not considered.
- Data source: only keep GPT-4 generated data. Drop inferior data sources (gpt-3.5-turbo).
- popular dataset, download > 1K.
- Accuracy (%): randomly sample 20 for the instruction tuning dataset and 10 for other domains. Check the quality manually and provide quality signal = x / 20
- Relevance Score (1-5):
    - 5: Directly corresponds to one of [IFEval*, MTBench, AGIEval*, AlpacaEval, …] (Overfitting)
    - 4: Generally have instruction following format and GPT-4 / human level response.
    - 3: Most have instruction following format and correct response.
    - 2: Have major flaws (e.g. irrelevant) but may be useful
    - 1: low quality or potentially harmful impact


## Dataset

### Function Calling, Code, Math

| Name | Description | Domain | Quantity | Accuracy | Relevance | Notes for Quality | License |
|------|-------------|--------|----------|----------|-----------|-----------|-----------|
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | No duplicate in first 10. Wide variety of tasks. | Function calling | 113K | 4.5 | 4.5 |    |  apache-2.0  |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Answers are function names and parameter lists. Contains functions with ambiguous parameter types and trivial functions | Function calling | 60K | 5 | 4.5 |    |  cc-by-4.0  |
| [Gorilla OpenFunctions-v2](https://github.com/ShishirPatil/gorilla/tree/main/data) | GitHub JSON format data, no Hugging Face dataset. Uses AST to determine if API calls are correct | Function calling | 17K | 5 | 5 |    |  Apache-2.0  |
| [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | Question 1 gives task, inputs, constraints, example (Leetcode style), question 2 gives method signature, question 3 gives just problem description | Code | 75.2K | 4.5 | 3.5 |    |  mit  |
| [RLHFlow/CodeUltraFeedback-standard](https://huggingface.co/datasets/RLHFlow/CodeUltraFeedback-standard) | RLHF format, including chosen and rejected, The total chosen-rejected pairs are 50156 while the unique chosen answers are around 38.4K | Code | 38.4k/50.2k (see notes) | 4 | 4 |  Sizes are unique chosen answers and total chosen-rejected pairs, respectively  |  mit  |
| [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps) | Competitive Programming (Codeforces) style prompts with inputs, constraints, examples, descriptions. Includes separate test cases. Sometimes method signature provided. Relatively complicated. Items too long to check. | Code | 10K | N/A | N/A |    |  mit  |
| [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) | Prompts with supplied examples sometimes. Even with supplied examples, models only sometimes give the corresponding output | Code | 18.6K | 5 | 4 |    |  N/A  |
| [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | Question 4 does not provide public machine expressions. Original questions sometimes rewritten to be parameterized. | Math | 395k | 4.75 | 4.5 |    |   mit |
| [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) | Contains 13 datasets, such as camel math, etc. We examined the first 10 questions. Question 2 does have candidates and the answer is correct Question 3 does not provide a specific answer Question 5 does not provide a specific answer Most do not provide specific answers | Math | 262K | 4.5 | 3 |    |  mit  |
| [camel-ai/math](https://huggingface.co/datasets/camel-ai/math) | Dataset is composed of 50K problem-solution pairs obtained using GPT-4 | Math | 50k | 5 | 4.5 |    |  cc-by-nc-4.0  |
| [xinlai/Math-Step-DPO-10K](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K) | RLHF format, including chosen and rejected. Use step-by-step prompt. `initial_reason_steps` includes preliminary calculation and hints. | Math | 10.8k | 4.5 | 3.5 |    |  cc-by-nc-4.0  |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | Commonly used for many benchmarks, including the [LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard). `Answer` includes `<>` formated calculation. | Math | train 7.47k test 1.32K | 5 | 4.5 |    |  mit  |
| [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) | Final self-alignment training dataset for StarCoder2-Instruct. | Code | 50.7k |  |  |  | odc-by
| [theblackcat102/evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) | Similar to [ise-uiuc/Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K). | Code | 111k |  |  |  | apache-2.0



### Instruction Following

| Name | Description | Domain | Number of Samples | Accuracy | Relevance | Note for Quality | License |
|------|-------------|-----------------------------------|-------------------|----------|-----------|------------------|-----------|
| [Open-Orca/1million-gpt-4](https://huggingface.co/datasets/Open-Orca/1million-gpt-4) | FLAN collection which has been augmented by submitting the listed question to GPT-4. Many questions supply a passage as context. | instruct | 1M | 5 | 4 |    | N/A |
| [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | This release provides an efficient means of reacting our OpenOrca dataset with using larger slices of our data, while only including ~500k GPT-4 completions. Many questions supply a passage as context | Instruct (1-turn chat) | 518k | 5 | 4 |    | mit |
| [teknium/GPT4-LLM-Cleaned](https://huggingface.co/datasets/teknium/GPT4-LLM-Cleaned) | Instruction-Following Data generated by GPT-4 using Alpaca prompts. Separated into main instruction, with optional accompanying parameter. E.g. "instruction": "what does this code do?", " input":"def function()" | Instruct | 54.6k | 5 | 4 |    | apache-2.0 |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | Dolly2.0 (Pairs, English, 15K+ entries) — A dataset of human-written prompts and responses, featuring tasks like question-answering and summarization. Categorized questions, e.g. "closed_qa", "classification", "open_qa", etc. Sometimes an optional "context" parameter is supplied.| Instruct | 15k | 5 | 4 |    | cc-by-sa-3.0 |
| [allenai/WildChat-1M (GPT4-EN)](https://huggingface.co/datasets/allenai/WildChat-1M) | 1 million conversations between human users and ChatGPT. 25.53% of the conversations come from the GPT-4 chatbot, while the rest come from the GPT-3.5 chatbot. Contains accompanying scores/classifications on various categories of harmfulness, e.g. "harassment", "self-harm", etc. Many non-English entries. | chat, instruct | 168k | 4 | 5 | filter gpt-4-en. Size refers to gpt-4 entries only | odc-by |
| [sablo/oasst2_curated](https://huggingface.co/datasets/sablo/oasst2_curated) | A filtered and curated dataset taken from the top scoring OpenAssistant/oasst2 conversations. Saved in HF Chat format. The result is a high quality dataset for SFT. | chat | train 4.69k, test 24 | 5 | 4 | open-ended conversation, human annotated    | apache-2.0 |
| [CollectiveCognition/chats-data-2023-09-22](https://huggingface.co/datasets/CollectiveCognition/chats-data-2023-09-22) | Collection of chats between users and the ChatGPT model. These conversations have been shared by users on the "Collective Cognition" website.  Includes ChatGPT generated conversation titles. | chat, instruct | 156 | 4.75 | 4 | Human: after filter out GPT-4    | mit |
| [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | one million real-world conversations with 25 state-of-the-art LLMs. Includes conversation topics with model tags, language, harmfulness ratings across multiple axes, and PII redaction. Many non-English prompts. | chat, instruct | 1M | 4.5 | 4 | Human: after filter out GPT-4 | [LMSYS-Chat-1M Dataset License](https://huggingface.co/datasets/lmsys/lmsys-chat-1m#lmsys-chat-1m-dataset-license-agreement) |
| [teknium/GPTeacher-General-Instruct](https://huggingface.co/datasets/teknium/GPTeacher-General-Instruct) | GPT-4 Generated self-instruct dataset. Mix of open/closed qa, rewriting, answering questions based on supplied passage.| Instruct | 89.3k | 4.5 | 4 | gpt-4 generated    | mit |
| [stingning/ultrachat](https://hf.co/datasets/stingning/ultrachat) | Some data inside the 774K are very long, basically exceeding 10000 in length. Questions and responses combined into one field. | Chat | 774k | 4.5 | 4 | Human: The dialogue is a list of strings chatgpt generated with human refinements | mit |
| [jondurbin/airoboros-3.2](https://huggingface.co/datasets/jondurbin/airoboros-3.2?not-for-all-audiences=true) | modified self-instruct gpt4. Contains some harmful/toxic content. | instruct | 58,709 | 4.5 | 4 | Accuracy: Errors in mathematical calculations. Data was generated primarily with gpt-4 | cc-by-4.0 |
| [openbmb/UltraInteract_sft](https://huggingface.co/datasets/openbmb/UltraInteract_sft) | a large-scale, high-quality alignment dataset specifically designed for complex reasoning tasks. | Instruct, with coding, math, etc. as sub-tasks | 289K | 4 | 5 | specifically for reasoning | mit |
| [AutoIF](https://github.com/QwenLM/AutoIF) | Synthetic dataset that matches IFEval, no open source download available. Restrictions on output format, length. E.g. 50 words, 5 sentences, 4-syllable words, palindromes. Strong emphasis on conciseness. | Instruct | N/A | N/A | N/A | hack IFEval to generate data | apache-2.0 |
| [WizardLM/WizardLM_evol_instruct_V2_196k](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k) | original wizard lm data | Instruct | 143k (only mixture of evolved data from Alpaca and ShareGPT) | 4.5 | 3 | Human: Some errors; gpt-3.5-turbo generated | mit |
| [TIGER-Lab/WebInstructSub](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub) | vast amounts of high-quality instruction data exist in the web corpus, spanning various domains like math and science. Specifically contains data from mathstackexchange, stackexchange, and socratic. | instruct (math, science Q&A) | 2.34M | 5 | 3 | Human: not relevant | apache-2.0 |
| [allenai/soda](https://huggingface.co/datasets/allenai/soda) | Dialogue dataset covering a wide range of social interactions. | Chat | train 1.19M validation 146k test 149k | 5 | 3 | Accuracy: Discrepancy in the amount of dialogue and conversation data. Dialogue contains proper_name information. Human: not GPT-4 level | cc-by-4.0 |
| [nvidia/Daring-Anteater](https://huggingface.co/datasets/nvidia/Daring-Anteater) | consisting of 100k conversations, each averaging 2.88 model turns, generated using NVIDIA proprietary model and Mistral-8x7B-Instruct-v0.1, while the remaining samples are sourced from FinQA, wikitablequestions, and commercially-friendly subsets of Open-Platypus | Chat | 99.5k | 5 | 3 | Human: from NVIDIA proprietary models and Mistral-8x7B-Instruct-v0.1 not GPT-4 | cc-by-4.0 |
| [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | Some Alpaca/ LLaMA-like models (Pairs, English) — Cleaned version of Alpaca, GPT_LLM, and GPTeacher. Cleaned to correct: hallucinations, merged instructions, empty outputs, empty code examples, instructions to generate images, N/A outputs, wrong answers (?), non-sensical/unclear instructions, extra escape and control characters | Instruct | 52k | 5 | 3 |  Should review some of the choices for cleaning data.  | cc-by-4.0 |
| [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | ChatGLM-fine-tune-LoRA; Koala (Dialog, Pairs, English, 52K entries, 21.4MB) — A dataset generated by text-davinci-003 to enhance language models' ability to follow human instruction. Contains instruction field (all unique), optional input in ~40% of data, model output, and finally a formatted combination following a prompt template.| Instruct | 52k | 4.5 | 3 |    | cc-by-nc-4.0 |
| [cascip/ChatAlpaca](https://github.com/cascip/ChatAlpaca) | use ChatGPT (GPT-3.5-turbo) to generate follow-up utterances and continue the conversation with ChatGPT | chat, instruct | 20k | 4 | 3 |    | apache-2.0 |
| [philschmid/guanaco-sharegpt-style](https://huggingface.co/datasets/philschmid/guanaco-sharegpt-style) | Some code content, mostly general conversations. Mostly non-English | Chat | 9.03k rows | 3 | 3 | Accuracy: Many foreign languages. Human: After filtering, a high-quality GPT4 daily Q&A dataset, size 6K, mainly knowledge Q&A, programming questions, reasoning calculations, including Simplified Chinese, Traditional Chinese, English, Japanese, Korean, and various languages | N/A |
| [andersonbcdefg/gpt4all](https://huggingface.co/datasets/andersonbcdefg/gpt4all) | Questions from stackoverflow. Contains HTML tags. | 1-turn chat, user-assistant interaction | 438k | 3 | 2 | Human: prompt is html coding and math, not relevant to instruction following | N/A |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | This version of the dataset contains data collected on the open-assistant.io website until April 12 2023. Human-generated, human-annotated assistant-style conversation corpus consisting of 161,443 messages in 35 different languages, annotated with 461,292 quality ratings, resulting in over 10,000 fully annotated conversation trees.| chat | train:84.4k val:4.4k | N/A | 4 | human-level response; need process conversation tree to inspect data | apache-2.0 |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | This version of the dataset contains data collected on the open-assistant.io website until Nov 5 2023. Same type of data as oasst1. Data contains message trees, where initial prompt is root node with multiple child nodes as different replies, representing different conversation routes. | chat | train:129k val:6.6k | N/A | 4 | human-level response; need process conversation tree to inspect data | apache-2.0 |
| [Salesforce/dialogstudio](https://huggingface.co/datasets/Salesforce/dialogstudio) | Towards Richest and Most Diverse Unified Dataset Collection and Instruction-Aware Models for Conversational AI. Variety of dialogues including: Knowledge-Grounded-Dialogues, Natural-Language-Understanding, Open-Domain-Dialogues, Task-Oriented-Dialogues, Dialogue-Summarization, Conversational-Recommendation-Dialogs | Chat | See details at https://github.com/salesforce/DialogStudio/blob/main/Dataset_Statistics.csv train: 600 ? val: 200 ? test: 400 ? | 0 | 2 | focus on conversational AI, irrelevant Accuracy: Cannot be viewed online, need to download locally first. Organize dialogues in the form of turn list, including many other auxiliary information | apache-2.0 |
| [argilla/magpie-ultra-v0.1](https://huggingface.co/datasets/argilla/magpie-ultra-v0.1) | synthetically generated dataset for supervised fine-tuning using the new Llama 3.1 (70B-turbo) model together with other Llama models like Llama-Guard-3-5B and Meta-Llama-3.1-8B-Instruct. Includes synthetic difficulty tags, required knowledge info as well. Base instructions generated by Llama-405B, supplementing info generated by 8B Llama models. | instruction | 50k | 4.75 | 3.5 | llama-3.1-40B generated | llama3.1 |
| [bigscience/P3](https://huggingface.co/datasets/bigscience/P3) | Wide variety of NLP tasks including multiple-choice QA, sentiment analysis or natural language inference. | Instruct (QA) | 122,039,002 1220396 (?) Directly obtained from huggingface num rows | 5 | 3 | Responses are short, mostly 1-2 sentences. A LOT of duplicates. Should probably do a lot of additional filtering for this dataset. | apache-2.0 |
| [yizhongw/self_instruct](https://huggingface.co/datasets/yizhongw/self_instruct) | The huggingface dataset also includes P3 and Super Natural Instructions data. Self-Instruct is a framework that helps language models improve their ability to follow natural language instructions. It does this by using the model's own generations to create a large collection of instructional data. With Self-Instruct, it is possible to improve the instruction-following capabilities of language models without relying on extensive manual annotation. Mostly in prompt completion format given a passage.| Instruct | 82.6k |  | 3 | Human: not GPT-4 level | apache-2.0 |
| [meta-llama/Meta-Llama-3.1-8B-Instruct-evals](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-Instruct-evals) | This dataset contains the Meta evaluation result details for Meta-Llama-3.1-8B-Instruct. The dataset has been created from 30 evaluation tasks. | Instruct, function calling | 157,823 157k (?) |  | 3 | Human: not GPT-4 level, llama3 generated on benchmarks! | llama3.1 |
| [mosaicml/instruct-v3](https://huggingface.co/datasets/mosaicml/instruct-v3) | Each piece of data has a marked source. This is an aggregate dataset comprised of Dolly, HFRLHF (derived from Databricks Dolly) Self-Instruct (Yizhong Wang) and HH (Anthropic Harmless) datasets, combined with Competition Math, Duorc, CoT GSM8k, Qasper, Quality, Summ Screen FD and Spider. Brief prompt template included with every instruction. | Instruct | train 56.2k test 6.81k |  | 2 | not GPT-4 level, irrelevant task | cc-by-sa-3.0 |
| [teknium/OpenHermes-2.5](https://hf.co/datasets/teknium/OpenHermes-2.5) | Airoboros 2.2 + CamelAI Domain Expert Datasets (Physics, Math, Chemistry & Biology) + Fatidici4K-orca CoT + GPT4 Collective Cognition (09-10-2023 ~ CoT) + Alpaca GPT4 + Evol Instruct 70K && 140K + Glaive Code Assistant + GPT4-LLM + GPTeacher + Medical Tasks + MetaMath 40k + SlimOrca 550K + Platypus + ShareGPT (GPT4-Only) + Unnatural Instructions GPT4 | Chat instruct, including coding, math, etc. | 1M |  |  | naive mixture of multiple datasets Filtering included removal of OpenAI refusals, disclaimers, and "As an AI" type examples and more | N/A |
| [bilexi/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | The user provides questions, and the response is a prompt from the assistant | Instruct | 26.9k |  | 2 | irrelevant (Customer Service) | cdla-sharing-1.0 |

### Safety

| Name | Description | Domain (Math/Coding/Instruct/Chat/Safety) | Number of Samples | Accuracy | Relevance | Note for Quality | License |
|------|-------------|-----------------------------------|-------------------|----------|-----------|------------------|-----------|
| [Anthropic/hh-rlhf(harmless-base)](https://huggingface.co/datasets/Anthropic/hh-rlhf) | RLHF format, collected by Anthropic's 52B base model, but has many errors and incorrect annotations. | safety | 42.5k | 2 | 2 |  There are many errors in the annotations, many "chosen" responses are still not safe. |  mit  |
| [Anthropic_HH_Golden](https://huggingface.co/datasets/Unified-Language-Model-Alignment/Anthropic_HH_Golden) | RLHF format, Extending the harmless dataset of Anthropic/hh-rlhf, but rewrite the chosen response with gpt-4. | safety | 42.5k | 5 | 5 |    |  apache-2.0  |
| [nvidia/Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0) | The datasets contains prompt, response and safety labels. Prompts are from Antropic's HH-RLHF dataset, and reponses are generated from Mistral-7B-v0.1. The human annotation is high-quality, but the prompts and reponses are concatenated, without clear spliting symbol. | safety | 10.8k | 5 | 4 |    |  cc-by-4.0  |

## Contributors

<a href="https://github.com/eryajf/learn-github/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=shizhediao/Post-Training-Data-Flywheel" />
</a>
