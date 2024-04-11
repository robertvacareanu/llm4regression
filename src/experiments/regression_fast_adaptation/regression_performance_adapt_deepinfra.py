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
    (DeepInfra(model_id='codellama/CodeLlama-70b-Instruct-hf'),  'codellama70b'   ),
]:
    for dataset in [
        'regression_ni22',
        'regression_ni13',

        'original1',
        'original2',
        'friedman1',
        'friedman2',
        'friedman3',

    ]:
        for seed in [1, 2, 3]:
            outputs = []
            ((x_train, _, y_train, _), y_fn) = get_dataset(dataset)(max_train=101, noise=0, random_state=seed, round=True, round_value=2)
            def run():
                # fspt = construct_few_shot_prompt(x_train, y_train, x_test, encoding_type='vanilla')
                # fspt.format(**x_train[i:(i+1)].to_dict('records')[0])
                # fspt = construct_few_shot_prompt(x_train, y_train, x_test, encoding_type='vanilla')
                # print(fspt.format(**x_test.to_dict('records')[0]))
                # print(y_test)
                # exit()
                for i in tqdm.tqdm(range(1, 101)):
                    try:
                        o = llm_regression(llm, x_train[:i], x_train[i:(i+1)], y_train[:i], y_train[i:(i+1)], encoding_type='vanilla', model_name=model_name, add_instr_prefix=True)
                        outputs.append(
                            {
                                **scores(**o), 
                                'full_outputs': o['full_outputs'],
                                'dataset': dataset,
                                'x_train': x_train[:i].to_dict('records'),
                                'x_test' : x_train[i:(i+1)].to_dict('records'),
                                'y_train': y_train[:i].to_list(),
                                'y_test' : y_train[i:(i+1)].to_list(),
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

            Path(f"results/online_learning_regression/seed_{seed}/{model_name}/").mkdir(parents=True, exist_ok=True)
            with open(f'results/online_learning_regression/seed_{seed}/{model_name}/{dataset}.jsonl', 'w+') as fout:
                for line in outputs:
                    _ = fout.write(json.dumps(line))
                    _ = fout.write('\n')
