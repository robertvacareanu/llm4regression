from src.regressors.remote_llm_regressor import *
from src.dataset_utils import get_dataset
from src.score_utils import scores
from langchain_community.llms import DeepInfra
import tqdm
import json
import os
import warnings
from pathlib import Path

with open('api_deepinfra_personal.key') as fin:
    os.environ['DEEPINFRA_API_TOKEN'] = fin.readlines()[0].strip()

    
for (llm, model_name) in [
    (DeepInfra(model_id='mistralai/Mixtral-8x7B-Instruct-v0.1'), 'mixtral8x7B'    ),
    (DeepInfra(model_id='meta-llama/Llama-2-70b-chat-hf'),       'llama270bchathf'),
    (DeepInfra(model_id='01-ai/Yi-34B-Chat'),                    'yi34chat'       ),
    (DeepInfra(model_id='mistralai/Mistral-7B-Instruct-v0.1'),   'mixtral7B'      ),
    (DeepInfra(model_id='mistralai/Mistral-7B-Instruct-v0.2'),   'mixtral7Bv2'      ),
    (DeepInfra(model_id='codellama/CodeLlama-70b-Instruct-hf'),  'codellama70b'   ),
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
                
            run()

        Path(f"results/regression_performance/{model_name}/").mkdir(parents=True, exist_ok=True)
        with open(f'results/regression_performance/{model_name}/{dataset}.jsonl', 'w+') as fout:
            for line in outputs:
                _ = fout.write(json.dumps(line))
                _ = fout.write('\n')

