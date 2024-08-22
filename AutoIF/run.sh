#!/bin/bash
set -e

export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
export OPENAI_MODEL="gpt-4o-mini"

export OUTPUT_DIR=./gene_k50_n8_inst_dedup-4o-mini/
mkdir -p ${OUTPUT_DIR}


# read ./sample_data/seed_instruction.txt save to ${OUTPUT_DIR}/1_rft_prompts.jsonl
# step 1 prompt gpt to generate constraints similar to seed data
python3 ./code/1_RFT_with_kd_gpt.py ./sample_data/seed_instruction.txt ${OUTPUT_DIR}/1_rft_prompts.jsonl
python3 ./code/api_request_parallel_processor.py --requests_filepath ${OUTPUT_DIR}/1_rft_prompts.jsonl --max_attempts 100

# step 2 prompt gpt to generate python function and test cases corresponds to the constraints
# read ./sample_data/seed_instruction.txt and ${OUTPUT_DIR}/1rft_prompts_results.jsonl save to ${OUTPUT_DIR}/2_eval_func_rft_prompts.jsonl
python3 ./code/2_verification_funcs_cases_generation_with_kd.py 
python3 ./code/api_request_parallel_processor.py --requests_filepath ${OUTPUT_DIR}/2_eval_func_rft_prompts.jsonl --max_attempts 100

# step3 call these functions to verify their correctness and filter
# read ${OUTPUT_DIR}/2_eval_func_rft_prompts_results.jsonl" save to ${OUTPUT_DIR}/3_cross_validation.jsonl
python3 ./code/3_cross_validation.py

# step 4 prompt gpt to translate functions back to instructions
# read ${OUTPUT_DIR}/cross_validation.jsonl save to ${OUTPUT_DIR}/4_backtranslate_prompt.jsonl
python3 ./code/4_eval_func_backtranslator.py
python3 ./code/api_request_parallel_processor.py --requests_filepath ${OUTPUT_DIR}/4_backtranslate_prompt.jsonl --max_attempts 100
python3 ./code/4_check_func_backtranslator.py # filter and save to ${OUTPUT_DIR}/4_back_trans.jsonl

# step 5 use NLI to judge whether the origin instruction and the back-translated instruction are correlated
# read ${OUTPUT_DIR}/4_back_trans.jsonl then filter and save to ${OUTPUT_DIR}/5_back_trans_fliter.jsonl
export CUDA_VISIBLE_DEVICES=3
python3 ./code/5_eval_func_backtranslator_filter.py

# above process generates constraints that can be verified by python code. Afterwards, Step 6 concat these constraints with sharegpt user prompts and query supervision model for response. Step 7 verifies the response using pre-acquired verification code and score each data.

# read 5_back_trans_fliter.jsonl and shaegpt and save to ${OUTPUT_DIR}/6_instruction_filtered_query_prompt.jsonl
python3 ./code/6_concat_sharegpt_query.py
python3 ./code/api_request_parallel_processor.py --requests_filepath ${OUTPUT_DIR}/6_instruction_filtered_query_prompt.jsonl --max_attempts 100

# read ${OUTPUT_DIR}/6_instruction_filtered_query_prompt.jsonl and save to {out_dir}/7_query_need_quality_score_prompt.jsonl and {out_dir}/7_query_wo_score.jsonl
python3 ./code/7_query_vertification.py
python3 ./code/api_request_parallel_processor.py --requests_filepath ${OUTPUT_DIR}/7_query_need_quality_score_prompt.jsonl --max_attempts 100

# read ${OUTPUT_DIR}/7_query_need_quality_score_prompt.jsonl and save to {out_dir}/8_query_score_filter.jsonl
python3 ./code/8_query_score_filiter.py

# construct
python3 ./code/9_sft_data_construction.py