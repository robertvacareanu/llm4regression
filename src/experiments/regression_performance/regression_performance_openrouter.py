"""
The performance over regression problems.
Models considered:
    - anthropic/claude-3-opus
    - anthropic/claude-3-sonnet
    - anthropic/claude-3-haiku
    - anthropic/claude-2.1
    - google/gemini-pro
    - mistralai/mistral-medium
    
Running command: python -i -m src.experiments.regression_performance.regression_performance_openrouter
"""
from src.regressors.openrouter_llm_regressor import *
from src.dataset_utils import get_dataset
from src.score_utils import scores
import tqdm
import json
import os
import warnings
from pathlib import Path


if 'OPENROUTER_API_KEY' not in os.environ:
    print("No OpenRouter API key found in environment variables. Will attempt to read from `api_openrouter_personal.key`.")
    if os.path.exists('api_openrouter_personal.key'):
        with open('api_openrouter_personal.key') as fin:
            os.environ['OPENROUTER_API_KEY'] = fin.readlines()[0].strip()
    else:
        print("No `api_openrouter_personal.key` file found. Please create one with your OpenAI API key or set the `OPENROUTER_API_KEY` variable.")
        exit()



for (llm, model_name) in [
    (ChatOpenRouter(model_name='anthropic/claude-3-opus', temperature=0, max_retries=5), 'claude3opus'),
    # (ChatOpenRouter(model_name='anthropic/claude-3-sonnet', temperature=0, max_retries=5), 'claude3sonnet'),
    # (ChatOpenRouter(model_name='anthropic/claude-3-haiku', temperature=0, max_retries=5), 'claude3haiku'),
    # (ChatOpenRouter(model_name='anthropic/claude-2.1', temperature=0, max_retries=5), 'claude21'),
    # (ChatOpenRouter(model_name='anthropic/claude-2.0', temperature=0, max_retries=5), 'claude20'),
    # (ChatOpenRouter(model_name='anthropic/claude-1.2', temperature=0, max_retries=5), 'claude12'),
    # (ChatOpenRouter(model_name='google/gemini-pro', temperature=0, max_retries=5), 'geminipro'),
    # (ChatOpenRouter(model_name='mistralai/mistral-medium', temperature=0, max_retries=5), 'mistralmedium'),

    # (ChatOpenRouter(model_name='cohere/command', temperature=0, max_retries=5), 'coherecommand'),
    # (ChatOpenRouter(model_name='cohere/command-r', temperature=0, max_retries=5), 'coherecommand-r'),
    # (ChatOpenRouter(model_name='cohere/command-r-plus', temperature=0, max_retries=5), 'coherecommand-r-plus'),
    # (ChatOpenRouter(model_name='togethercomputer/stripedhyena-nous-7b', temperature=0, max_retries=5), 'stripedhyena-nous-7b'),
]:
    for dataset in [
        'regression_ni11',
        'regression_ni22',
        'regression_ni33',
        'regression_ni12',
        'regression_ni13',
        'regression_ni23',

        'original1',
        'original2',
        'original3',
        'original4',
        'original5',
        'friedman1',
        'friedman2',
        'friedman3',

        'simple_random_nn1',
        'transformer1',

        'character_regression1',

    ]:
        outputs = []
        if os.path.exists(f'results/regression_performance/{model_name}/{dataset}.jsonl'):
            with open(f'results/regression_performance/{model_name}/{dataset}.jsonl',) as fin:
                for line in fin:
                    outputs.append(json.loads(line))

        seeds_done = set([x['seed'] for x in outputs])

        # ' '.join([ for x in outputs])

        for seed in tqdm.tqdm(range(1, 101)):
            ((x_train, x_test, y_train, y_test), y_fn) = get_dataset(dataset)(max_train=50, max_test=1, noise=0, random_state=seed, round=True, round_value=2)
            def run():
                # fspt = construct_few_shot_prompt(x_train, y_train, x_test, encoding_type='vanilla')
                # prmpt = fspt.format(**x_test.to_dict('records')[0])
                # output = llm.call_as_llm('The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.\n\n\n\n\n' + prmpt, max_tokens=50).strip() # Similar to OpenAI
                # output = llm.call_as_llm(prmpt, max_tokens=50).strip() # Similar to OpenAI
                # exit()
                try:
                    o = llm_regression(llm, x_train, x_test, y_train, y_test, encoding_type='vanilla', model_name=model_name, add_instr_prefix=True)
                    outputs.append(
                        {
                            **scores(**o), 
                            'full_outputs': o['full_outputs'],
                            'seed'   : seed,
                            'dataset': dataset,
                            'x_train': x_train.to_dict('records'),
                            'x_test' : x_test.to_dict('records'),
                            'y_train': y_train.to_list(),
                            'y_test' : y_test.to_list(),
                        }
                    )
                except KeyboardInterrupt:
                    exit()
                except Exception as e:
                    print('-'*10)
                    print(e)
                    print(dataset, seed)
                    print('-'*10)
                    # print(f"Reached maximum context at {i}.")
                    return
            # print(seed)
            
            # Sometimes the API fails, so if we ever end up re-running, we try to skip over what it was already done
            if seed not in seeds_done:
                run()
            else:
                print(f"Seed {seed} is already done")

        Path(f"results/regression_performance/{model_name}/").mkdir(parents=True, exist_ok=True)
        with open(f'results/regression_performance/{model_name}/{dataset}.jsonl', 'w+') as fout:
            for line in outputs:
                _ = fout.write(json.dumps(line))
                _ = fout.write('\n')

