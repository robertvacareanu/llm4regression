from src.regressors.openrouter_llm_regressor import *
from src.dataset_utils import get_dataset
from src.score_utils import scores
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import DeepInfra
from langchain.callbacks import get_openai_callback, tracing_v2_enabled
import tqdm
import json
import os
import warnings
from pathlib import Path
import inspect

with open('api_openrouter_personal.key') as fin:
    os.environ['OPENROUTER_API_KEY'] = fin.readlines()[0].strip()

for (llm, model_name) in [
    (ChatOpenRouter(model_name='anthropic/claude-3-opus', temperature=0, max_retries=5), 'claude3opus'),
]:
    for dataset in [
        'regression_ni11',
        'regression_ni22',
        'regression_ni33',
        'regression_ni13',

        'original1',
        'original2',
        'friedman1',
        'friedman2',
        'friedman3',

    ]:
        outputs = []

        seeds_done = set([x['seed'] for x in outputs])

        # ' '.join([ for x in outputs])

        # Run only a couple of seeds
        for seed in range(1, 3):
            ((x_train, x_test, y_train, y_test), y_fn) = get_dataset(dataset)(max_train=50, max_test=1, noise=0, random_state=seed, round=True, round_value=2, print_coeffs=True)
            # Increase max tokens to allow for long explanations; Do not add instr_prefix
            o = llm_regression(llm, x_train, x_test, y_train, y_test, encoding_type='vanilla', model_name=model_name, max_tokens=1000, add_instr_prefix=False)
            outputs.append(
                {
                    'full_outputs': o['full_outputs'],
                    'seed'   : seed,
                    'dataset': dataset,
                    'x_train': x_train.to_dict('records'),
                    'x_test' : x_test.to_dict('records'),
                    'y_train': y_train.to_list(),
                    'y_test' : y_test.to_list(),
                    'fn'     : inspect.getsourcelines(y_fn),
                }
            )
            

        Path(f"results/regression_justification_analysis/{model_name}/").mkdir(parents=True, exist_ok=True)
        with open(f'results/regression_justification_analysis/{model_name}/{dataset}.jsonl', 'w+') as fout:
            for line in outputs:
                _ = fout.write(json.dumps(line))
                _ = fout.write('\n')

