"""

python -m src.experiments.regression_performance.regression_performance_fireworks
"""
from src.regressors.fireworks_llm_regressor import *
from src.dataset_utils import get_dataset
from src.score_utils import scores
import tqdm
import json
import os
from pathlib import Path

if 'FIREWORKS_API_KEY' not in os.environ:
    print("No Fireworks API key found in environment variables. Will attempt to read from `api_fireworks_personal.api`.")
    if os.path.exists('api_fireworks_personal.api'):
        with open('api_fireworks_personal.api') as fin:
            os.environ['FIREWORKS_API_KEY'] = fin.readlines()[0].strip()
    else:
        print("No `api_fireworks_personal.api` file found. Please create one with your Fireworks API key or set the `FIREWORKS_API_KEY` variable.")
        exit()


for (llm, model_name) in [
    (Fireworks(model="accounts/fireworks/models/dbrx-instruct", temperature=0, max_retries=7), 'dbrxinstruct'),
    (Fireworks(model="accounts/fireworks/models/mixtral-8x22b", temperature=0, max_retries=7), 'mixtral8x22B'),
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

        for seed in tqdm.tqdm(range(1, 101)):
            ((x_train, x_test, y_train, y_test), y_fn) = get_dataset(dataset)(max_train=50, max_test=1, noise=0, random_state=seed, round=True, round_value=2)
            def run():
                # fspt = construct_few_shot_prompt(x_train[:i], y_train[:i], x_train[i:(i+1)], encoding_type='vanilla')
                # fspt.format(**x_train[i:(i+1)].to_dict('records')[0])
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
                except:
                    # print(f"Reached maximum context at {i}.")
                    return
                
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

