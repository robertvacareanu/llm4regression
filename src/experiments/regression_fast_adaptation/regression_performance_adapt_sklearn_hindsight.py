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

for seed in [1, 2, 3]:
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

        # More baselines
        (lambda x_train, x_test, y_train, y_test: baseline(x_train, x_test, y_train, y_test, 'average'), 'baseline_average'),
        (lambda x_train, x_test, y_train, y_test: baseline(x_train, x_test, y_train, y_test, 'last'),    'baseline_last'),
        (lambda x_train, x_test, y_train, y_test: baseline(x_train, x_test, y_train, y_test, 'random'),  'baseline_random'),
        (lambda x_train, x_test, y_train, y_test: baseline(x_train, x_test, y_train, y_test, 'constant_prediction', constant_prediction_value=0.5),  'baseline_constant_prediction'),
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
            ((x_train, _, y_train, _), y_fn) = get_dataset(dataset)(max_train=101, noise=0, random_state=seed, round=True, round_value=2)

            y_predict = model(x_train, x_train, y_train, y_train)['y_predict'].tolist()[1:] # Skip first because for LLMs we never predict the first

            Path(f"results/online_learning_regression/seed_{seed}/hindsight/{model_name}/").mkdir(parents=True, exist_ok=True)
            with open(f'results/online_learning_regression/seed_{seed}/hindsight/{model_name}/{dataset}.jsonl', 'w+') as fout:
                _ = fout.write(json.dumps(y_predict))
