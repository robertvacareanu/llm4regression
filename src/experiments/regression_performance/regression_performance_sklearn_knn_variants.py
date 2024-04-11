from src.regressors.sklearn_regressors import knn_regression_search
from src.regressors.remote_llm_regressor import *
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


for (model, model_name) in [(x, f'knn_search_{i}') for i, x in enumerate(knn_regression_search())]:
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
        for seed in range(1, 101):
            ((x_train, x_test, y_train, y_test), y_fn) = get_dataset(dataset)(max_train=50, max_test=1, noise=0, random_state=seed, round=True, round_value=2)
            def run():
                    o = model(x_train, x_test, y_train, y_test)
                    outputs.append(
                        {
                            **scores(**o), 
                            'seed'   : seed,
                            'dataset': dataset,
                            'x_train': x_train.to_dict('records'),
                            'x_test' : x_test.to_dict('records'),
                            'y_train': y_train.to_list(),
                            'y_test' : y_test.to_list(),
                        }
                    )        
            run()
        Path(f"results/regression_performance_knn_variants/sklearn/{model_name}/").mkdir(parents=True, exist_ok=True)
        with open(f'results/regression_performance_knn_variants/sklearn/{model_name}/{dataset}.jsonl', 'w+') as fout:
            for line in outputs:
                _ = fout.write(json.dumps(line))
                _ = fout.write('\n')

