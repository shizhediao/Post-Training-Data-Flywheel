# Data Process and Generation for Function Calling, Code and Math Datasets

## Process
We formulate the data into [LMFlow format](https://optimalscale.github.io/LMFlow/examples/DATASETS.html#data-format), which includes `system` prompt, `tools` as a list of description of functions, and messages from `user` and `assistant` one by one in turn.

## Generation
For the apis used as the target in generation, we adopt the huggingface, tensorflow and torchhub apis collected by [Gorilla](https://github.com/ShishirPatil/gorilla/tree/main/data/api).
For generation, we utilize the query template from [APIGen](https://arxiv.org/abs/2406.18518), and prompt GPT-4o-mini to generate extra examples.

After generation, we use a filter to check whether the generated instances include the target `api_call` in its generated code part. And the criterion is both the `api_name` and the method to call the api are included. After generation, we obtain 11319 items out of 15153 in total.

To run the generation script, use
```bash
python data_generation.py
```
To add your own apis, just define them in a json file with a form similar to those in [Gorilla](https://github.com/ShishirPatil/gorilla/tree/main/data/api) and load them in the script.