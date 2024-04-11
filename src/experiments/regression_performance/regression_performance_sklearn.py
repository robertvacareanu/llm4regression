from src.regressors.sklearn_regressors import (linear_regression,
            ridge, 
            lasso, 
            mlp_universal_approximation_theorem1, 
            mlp_universal_approximation_theorem2, 
            mlp_universal_approximation_theorem3, 
            mlp_deep1, 
            mlp_deep2, 
            mlp_deep3, 
            random_forest, 
            bagging, 
            gradient_boosting, 
            adaboost, 
            voting, 
            baseline,
            svm_regression,
            svm_and_scaler_regression,
            knn_regression,
            kernel_ridge_regression,
            lr_with_polynomial_features_regression,
            spline_regression,
            knn_regression_v2,
            knn_regression_v3,
)
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


for (model, model_name) in [
    (linear_regression,                    'lr'),
    (ridge,                                'ridge'),
    (lasso,                                'lasso'),
    (mlp_universal_approximation_theorem1, 'mlp_uat1'),
    (mlp_universal_approximation_theorem2, 'mlp_uat2'),
    (mlp_universal_approximation_theorem3, 'mlp_uat3'),
    (mlp_deep1,                            'mlp_deep1'),
    (mlp_deep2,                            'mlp_deep2'),
    (mlp_deep3,                            'mlp_deep3'),
    (random_forest,                        'random_forest'),
    (bagging,                              'bagging'),
    (gradient_boosting,                    'gb'),
    (adaboost,                             'ab'),
    (svm_regression,                       'svm'),
    (svm_and_scaler_regression,            'svm_w_s'),
    (knn_regression,                       'knn'),
    (kernel_ridge_regression,              'kr'),

    (lr_with_polynomial_features_regression, 'lr_with_polynomial'),
    (spline_regression,                      'spline'),

    (knn_regression_v2,                    'knn_v2'),
    (knn_regression_v3,                    'knn_v3'),

    # # More baselines
    (lambda x_train, x_test, y_train, y_test: baseline(x_train, x_test, y_train, y_test, 'average'), 'baseline_average'),
    (lambda x_train, x_test, y_train, y_test: baseline(x_train, x_test, y_train, y_test, 'last'),    'baseline_last'),
    (lambda x_train, x_test, y_train, y_test: baseline(x_train, x_test, y_train, y_test, 'random'),  'baseline_random'),
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
        Path(f"results/regression_performance/sklearn/{model_name}/").mkdir(parents=True, exist_ok=True)
        with open(f'results/regression_performance/sklearn/{model_name}/{dataset}.jsonl', 'w+') as fout:
            for line in outputs:
                _ = fout.write(json.dumps(line))
                _ = fout.write('\n')

