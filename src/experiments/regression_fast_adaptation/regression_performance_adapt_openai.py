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

if 'OPENAI_API_KEY' not in os.environ:
    print("No OpenAI API key found in environment variables. Will attempt to read from `api.key`.")
    if os.path.exists('api.key'):
        with open('api.key') as fin:
            os.environ['OPENAI_API_KEY'] = fin.readlines()[0].strip()
    else:
        print("No `api.key` file found. Please create one with your OpenAI API key or set the `OPENAI_API_KEY` variable.")
        exit()


# llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)
# model_name = 'gpt4_0125_preview'
# llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)
# model_name = 'gpt4-turbo'
# model_name = 'gpt4_with_prefixinstr'
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
model_name = 'chatgpt'
# llm = OpenAI(model_name="davinci-002", temperature=0)
# model_name = 'davinci002'
with get_openai_callback() as cb:
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
                        o = llm_regression(llm, x_train[:i], x_train[i:(i+1)], y_train[:i], y_train[i:(i+1)], encoding_type='vanilla', add_instr_prefix=True)
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
            print(cb)

            Path(f"results/online_learning_regression/seed_{seed}/{model_name}/").mkdir(parents=True, exist_ok=True)
            with open(f'results/online_learning_regression/seed_{seed}/{model_name}/{dataset}.jsonl', 'w+') as fout:
                for line in outputs:
                    _ = fout.write(json.dumps(line))
                    _ = fout.write('\n')
