from src.regressors.llm_regressor import *
from src.dataset_utils import get_dataset
from src.score_utils import scores
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import tqdm
import json
import os
import warnings
from pathlib import Path

with open('api_personal.key') as fin:
    os.environ['OPENAI_API_KEY'] = fin.readlines()[0].strip()


llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)
model_name = 'gpt4-turbo'
with get_openai_callback() as cb:
    for dataset in [
        'original1',
        'original2',
        'friedman1',
        'friedman2',
        'friedman3',
    ]:
        for max_train in [20, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500]:
            outputs = []
            for seed in range(1, 21):
                ((x_train, x_test, y_train, y_test), y_fn) = get_dataset(dataset)(max_train=max_train, max_test=1, noise=0, random_state=seed, round=True, round_value=2)
                def run():
                    # fspt = construct_few_shot_prompt(x_train, y_train, x_test, encoding_type='vanilla')
                    # fspt.format(**x_train[i:(i+1)].to_dict('records')[0])
                    # fspt = construct_few_shot_prompt(x_train, y_train, x_test, encoding_type='vanilla')
                    # print(fspt.format(**x_test.to_dict('records')[0]))
                    # print(y_test)
                    # exit()
                    try:
                        o = llm_regression(llm, x_train, x_test, y_train, y_test, encoding_type='vanilla', add_instr_prefix=True)
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
            print(cb)

            Path(f"results/regression_performance_plateau/{max_train}/{model_name}/").mkdir(parents=True, exist_ok=True)
            with open(f'results/regression_performance_plateau/{max_train}/{model_name}/{dataset}.jsonl', 'w+') as fout:
                for line in outputs:
                    _ = fout.write(json.dumps(line))
                    _ = fout.write('\n')
