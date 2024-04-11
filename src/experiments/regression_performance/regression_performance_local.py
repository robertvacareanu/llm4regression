from src.regressors.local_llm_regressor import *
from src.dataset_utils import get_dataset
from src.score_utils import scores
import tqdm
import json
import os
import warnings
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hf_model_name', type=str, default='RWKV/rwkv-raven-14b', help='What model to use (from Huggingface)')
parser.add_argument('--short_name', type=str, default='rwkv-v4-14b', help='A short name to save the results')

# Parse the arguments
args = vars(parser.parse_args())



for ((llm, tokenizer), model_name) in [
    ((AutoModelForCausalLM.from_pretrained(args['hf_model_name'], torch_dtype=torch.float16).to(0), AutoTokenizer.from_pretrained(args['hf_model_name'])), args['short_name']),
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
        for seed in tqdm.tqdm(range(1, 101)):
            ((x_train, x_test, y_train, y_test), y_fn) = get_dataset(dataset)(max_train=50, max_test=1, noise=0, random_state=seed, round=True, round_value=2)
            def run():
                # fspt = construct_few_shot_prompt(x_train[:i], y_train[:i], x_train[i:(i+1)], encoding_type='vanilla')
                # fspt.format(**x_train[i:(i+1)].to_dict('records')[0])
                try:
                    o = llm_regression(llm, tokenizer, x_train, x_test, y_train, y_test, encoding_type='vanilla', model_name=model_name, add_instr_prefix=True)
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
                
            run()

        Path(f"results/regression_performance/{model_name}/").mkdir(parents=True, exist_ok=True)
        with open(f'results/regression_performance/{model_name}/{dataset}.jsonl', 'w+') as fout:
            for line in outputs:
                _ = fout.write(json.dumps(line))
                _ = fout.write('\n')

