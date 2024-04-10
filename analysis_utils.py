
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.dataset_utils import get_dataset
from src.score_utils import scores
import glob
import json
import pandas as pd
import tqdm
import os
import seaborn as sns

hatching_patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

name_to_short = {
    'chatgpt'        : 'Chat GPT',
    'codellama70b'   : 'Code Llama 70B',
    'gpt4'           : 'GPT-4 (original)',
    'gpt4-turbo'     : 'GPT-4',
    'llama270bchathf': 'Llama2 70B Chat HF',
    'mixtral7B'      : 'Mistral 7B',
    'mixtral8x7B'    : 'Mixtral 8x7B',
    'pythia12B'      : 'Pythia 12B',
    'yi34chat'       : 'Yi 34B Chat',

    'davinci002'     : 'Davinci 002',
    'babbage002'     : 'Babbage 002',

    'claude3opus'    : 'Claude 3 Opus',
    'claude3sonnet'  : 'Claude 3 Sonnet',
    'claude3haiku'   : 'Claude 3 Haiku',
    'claude21'       : 'Claude 2.1',
    'claude20'       : 'Claude 2.0',
    'claude12'       : 'Claude 1.2',

    'coherecommand'       : 'Cohere Command',
    'coherecommand-r'     : 'Cohere Command R',
    'coherecommand-r-plus': 'Cohere Command R Plus',

    'mistralmedium'    : 'Mistral Medium',
    'mistrallarge'     : 'Mistral Large',
    'geminipro'        : 'Gemini Pro',
    'palm2bison'       : 'PaLM2 Bison',
    'dbrxinstruct'     : 'DBRX',
    'falcon40b'        : 'Falcon 40B',
    'falcon40binstruct': 'Falcon 40B Instruct',
    'falcon7b'        : 'Falcon 7B',
    'falcon7binstruct': 'Falcon 7B Instruct',
    'rwkv-v4-14b'      : 'RWKV V4 14B',

    'stripedhyena-nous-7b': 'Striped Hyena 7B',

    'gpt4-turbo-20240409': 'GPT-4 (20240409)',

    'mixtral8x22B'    : 'Mixtral 8x22B',

    'gpt4_with_prefixinstr': 'GPT 4 (+ Prefix Instr)',


    'lr'                : 'Linear Regression',
    'ridge'             : 'Ridge',
    'lasso'             : 'Lasso',
    'mlp_uat1'          : 'MLP Wide 1',
    'mlp_uat2'          : 'MLP Wide 2',
    'mlp_uat3'          : 'MLP Wide 3',
    'mlp_deep1'         : 'MLP Deep 1',
    'mlp_deep2'         : 'MLP Deep 2',
    'mlp_deep3'         : 'MLP Deep 3',
    'random_forest'     : 'Random Forest',
    'bagging'           : 'Bagging',
    'gb'                : 'Gradient Boosting',
    'ab'                : 'AdaBoost',
    'svm'               : 'SVM',
    'svm_w_s'           : 'SVM + Scaler',
    'knn'               : 'KNN',
    'knn_v2'            : 'KNN v2',
    'knn_v3'            : 'KNN v3',
    'knn_v4'            : 'KNN v4',
    'knn_v5_adaptable'  : 'KNN v5',
    'kr'                : 'Kernel Ridge',
    'lr_with_polynomial': 'Linear Regression + Poly',
    'spline'            : 'Spline',

    'baseline_average' : 'Average',
    'baseline_last'    : 'Last',
    'baseline_random'  : 'Random',
}

name_to_group = {
    'chatgpt'        : 'LLM',
    'codellama70b'   : 'LLM',
    'gpt4'           : 'LLM',
    'gpt4-turbo'     : 'LLM',
    'llama270bchathf': 'LLM',
    'mixtral7B'      : 'LLM',
    'mixtral8x7B'    : 'LLM',
    'pythia12B'      : 'LLM',
    'yi34chat'       : 'LLM',
    'davinci002'     : 'LLM',
    'babbage002'     : 'LLM',
    'claude3opus'    : 'LLM',
    'claude3sonnet'  : 'LLM',
    'claude3haiku'   : 'LLM',
    'claude21'       : 'LLM',
    'claude20'       : 'LLM',
    'claude12'       : 'LLM',
    'mistralmedium'  : 'LLM',
    'mistrallarge'   : 'LLM',
    'geminipro'      : 'LLM',
    'palm2bison'     : 'LLM',
    'dbrxinstruct'   : 'LLM',

    'falcon40b'        : 'LLM',
    'falcon40binstruct': 'LLM',

    'coherecommand'       : 'LLM',
    'coherecommand-r'     : 'LLM',
    'coherecommand-r-plus': 'LLM',

    'rwkv-v4-14b'         : 'LLM',
    'stripedhyena-nous-7b': 'LLM',
    

    'gpt4-turbo-20240409'  : 'LLM',
    'gpt4_with_prefixinstr': 'LLM',

    'mixtral8x22B'    : 'LLM',

    'lr'                : 'Traditional Supervised Model',
    'ridge'             : 'Traditional Supervised Model',
    'lasso'             : 'Traditional Supervised Model',
    'mlp_uat1'          : 'Traditional Supervised Model',
    'mlp_uat2'          : 'Traditional Supervised Model',
    'mlp_uat3'          : 'Traditional Supervised Model',
    'mlp_deep1'         : 'Traditional Supervised Model',
    'mlp_deep2'         : 'Traditional Supervised Model',
    'mlp_deep3'         : 'Traditional Supervised Model',
    'random_forest'     : 'Traditional Supervised Model',
    'bagging'           : 'Traditional Supervised Model',
    'gb'                : 'Traditional Supervised Model',
    'ab'                : 'Traditional Supervised Model',
    'svm'               : 'Traditional Supervised Model',
    'svm_w_s'           : 'Traditional Supervised Model',
    'knn'               : 'Traditional Supervised Model',
    'knn_v2'            : 'Traditional Supervised Model',
    'knn_v3'            : 'Traditional Supervised Model',
    'kr'                : 'Traditional Supervised Model',
    'lr_with_polynomial': 'Traditional Supervised Model',
    'spline'            : 'Traditional Supervised Model',


    'baseline_average' : 'Unsupervised Model',
    'baseline_last'    : 'Unsupervised Model',
    'baseline_random'  : 'Unsupervised Model',
}

shortname_to_group = {name_to_short[x]:name_to_group[x] for x in name_to_group.keys()}

dataset_to_name = {
    'regression_ni1'   : 'Regression NI 1/5', 
    'regression_ni2'   : 'Regression NI 2/5', 

    'regression_ni11'  : 'Regression NI 1/1', 
    'regression_ni12'  : 'Regression NI 1/2', 
    'regression_ni13'  : 'Regression NI 1/3', 
    'regression_ni22'  : 'Regression NI 2/2', 
    'regression_ni23'  : 'Regression NI 2/3', 
    'regression_ni33'  : 'Regression NI 3/3', 
    
    'regression_ni1_10': 'Regression NI 1/10', 
    'regression_ni2_10': 'Regression NI 2/10', 
    'regression_ni3_10': 'Regression NI 3/10', 
    
    'original1'        : 'Original 1', 
    'original2'        : 'Original 2', 
    'original3'        : 'Original 3',
    'original4'        : 'Original 4',
    'original5'        : 'Original 5',
    
    'friedman1'        : 'Friedman 1',
    'friedman2'        : 'Friedman 2', 
    'friedman3'        : 'Friedman 3', 

    'simple_random_nn1'      : 'Simple NN 1', 
    'simple_random_nn2'      : 'Simple NN 2', 
    'simple_random_nn3'      : 'Simple NN 3', 
    
    'simple_random_nn1_scaled': 'Simple Scaled NN 1', 

    'more_complex_random_nn1': 'More Complex NN 1', 
    'more_complex_random_nn2': 'More Complex NN 2', 
    'more_complex_random_nn3': 'More Complex NN 3', 

    'transformer1': 'Transformer 1',
    'transformer2': 'Transformer 2',
    'transformer3': 'Transformer 3',

    'character_regression1': 'Character Regression 1',

    'unlearnable1': 'Unlearnable 1',
    'unlearnable2': 'Unlearnable 2',
}

dataset_to_order = {
    'Regression NI 1/5': 1,
    'Regression NI 2/5': 2,
    'Original 1'     : 3,
    'Original 2'     : 4,
    'Original 3'     : 5,
    'Original 4'     : 6,
    'Original 5'     : 7,
    'Friedman 1'     : 8,
    'Friedman 2'     : 9,
    'Friedman 3'     : 10,
    'Regression NI 1/10': 11,
    'Regression NI 2/10': 12,
    'Regression NI 3/10': 13,
    'Simple NN 1'       : 14,
    'Simple NN 2'       : 15,
    'Simple NN 3'       : 16,
    'More Complex NN 1' : 17,
    'More Complex NN 2' : 18,
    'More Complex NN 3' : 19,
    'Regression NI 1/1': 20,
    'Regression NI 2/2': 21,
    'Regression NI 3/3': 22,
    'Regression NI 1/2': 23,
    'Regression NI 1/3': 24,
    'Regression NI 2/3': 25,
    'Unlearnable 1': 26,
    'Unlearnable 2': 27,
    'Simple Scaled NN 1': 28,
}

model_to_order = {
    # LLMs
    'mixtral7B'      : 1,
    'mixtral8x7B'    : 2,
    'codellama70b'   : 3,
    'llama270bchathf': 4,
    'pythia12B'      : 5,
    'yi34chat'       : 6,
    'davinci002'     : 7,
    'chatgpt'        : 8,
    'gpt4'           : 9,
    

    # Traditional Supervised Methods
    'lr'                : 10,
    'ridge'             : 11,
    'lasso'             : 12,
    'mlp_uat1'          : 13,
    'mlp_uat2'          : 14,
    'mlp_uat3'          : 15,
    'mlp_deep1'         : 16,
    'mlp_deep2'         : 17,
    'mlp_deep3'         : 18,
    'random_forest'     : 19,
    'bagging'           : 20,
    'gb'                : 21,
    'ab'                : 22,
    'svm'               : 23,
    'svm_w_s'           : 24,
    'knn'               : 25,
    'kr'                : 26,
    'lr_with_polynomial': 27,
    'spline'            : 28,

    # Unsupervised Baselines
    'baseline_average' : 29,
    'baseline_last'    : 30,
    'baseline_random'  : 31,
}


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit_curves(d):

    # Assuming 'd' is a list of cumulative regret values for each time step T
    # For demonstration, let's create a sample 'd' with random values
    np.random.seed(0)  # for reproducible results

    # Time steps (T = 1 to len(d))
    T = np.arange(1, len(d) + 1)

    # Define the models for curve fitting
    def linear_model(T, a, b):
        return a * T + b

    def sqrt_model(T, a, b):
        return a * np.sqrt(T) + b

    def log_model(T, a, b):
        return a * np.log(T) + b

    # Curve fitting for each model
    params_linear, _ = curve_fit(linear_model, T, d)
    params_sqrt, _ = curve_fit(sqrt_model, T, d)
    params_log, _ = curve_fit(log_model, T, d)

    # Generate fitted values
    fitted_linear = linear_model(T, *params_linear)
    fitted_sqrt = sqrt_model(T, *params_sqrt)
    fitted_log = log_model(T, *params_log)

    r2_linear = r2_score(d, fitted_linear)
    r2_sqrt = r2_score(d, fitted_sqrt)
    r2_log = r2_score(d, fitted_log)

    return [T, d, fitted_linear, fitted_sqrt, fitted_log, r2_linear, r2_sqrt, r2_log]

    # # Plotting
    # plt.figure(figsize=(12, 6))
    # plt.plot(T, d, label='Cumulative Regret', color='blue')
    # plt.plot(T, fitted_linear, label=f'Linear Fit ({np.round(r2_linear, 3)})', linestyle=':', color='red')
    # plt.plot(T, fitted_sqrt, label=f'Sqrt Fit ({np.round(r2_sqrt, 3)})', linestyle='-.', color='green')
    # plt.plot(T, fitted_log, label=f'Log Fit ({np.round(r2_log, 3)})', linestyle='--', color='purple')
    # plt.xlabel('Time Step (T)')
    # plt.ylabel('Cumulative Regret')
    # plt.title(f'Cumulative Regret and Fitted Curves ({dataset_to_name[dataset]})')
    # plt.legend()
    # plt.show()


def output_to_number(string, silent=True):
    valid  = True
    output = string.split('\n')[0].strip() 
    output = '.'.join(output.split('.')[:2]).strip()
    output = output.split(" ")[0].strip()
    try:
        output = float(output)
    except Exception as e:
        valid = False
        if not silent:
            print(e)
            print(output)
        output = 0.0

    return (output, valid)