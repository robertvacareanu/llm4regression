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
)
from src.regressors.remote_llm_regressor import *
from src.dataset_utils import get_dataset
from src.score_utils import scores
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
    (spline_regression                       , 'spline'),

    # More baselines
    (lambda x_train, x_test, y_train, y_test: baseline(x_train, x_test, y_train, y_test, 'average'), 'baseline_average'),
    (lambda x_train, x_test, y_train, y_test: baseline(x_train, x_test, y_train, y_test, 'last'),    'baseline_last'),
    (lambda x_train, x_test, y_train, y_test: baseline(x_train, x_test, y_train, y_test, 'random'),  'baseline_random'),
]:
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
            Path(f"results/regression_performance_plateau/{max_train}/sklearn/{model_name}/").mkdir(parents=True, exist_ok=True)
            with open(f'results/regression_performance_plateau/{max_train}/sklearn/{model_name}/{dataset}.jsonl', 'w+') as fout:
                for line in outputs:
                    _ = fout.write(json.dumps(line))
                    _ = fout.write('\n')

