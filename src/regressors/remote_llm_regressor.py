"""
Mostly DeepInfra
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import time
import random
import tqdm
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from src.regressors.prompts import construct_few_shot_prompt

def llm_regression(llm, x_train, x_test, y_train, y_test, encoding_type, model_name, add_instr_prefix=False, instr_prefix='The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.\n\n\n\n\n'):
    examples_test = []
    for x1 in x_test.to_dict('records'):
        examples_test.append(x1)

    fspt = construct_few_shot_prompt(x_train, y_train, x_test, encoding_type=encoding_type)
    full_outputs = []
    outputs = []
    gold    = []
    for x, y in tqdm.tqdm(zip(examples_test, y_test)):
        gold.append(y)
        if add_instr_prefix:
            inpt = instr_prefix + fspt.format(**x)
        else:
            inpt = fspt.format(**x)
        output = llm(inpt, stop=['\n'], max_new_tokens=10, temperature=0)
        full_outputs.append(output)
        output = output.strip().split("\n")[0].strip()
        # time.sleep(0.5)
        if encoding_type == "two_binaries":
            if '.' not in output:
                output = output + ".0"
            l, r = output.split(".")[:2]
            if len(output.split(".")) > 2:
                print("More than 2 .", output)
            l = int(l, 2)
            r = int(r, 2)
            output = f"{l}.{r}"
        try:
            output = float(output)
        except Exception as e:
            print(e)
            print(output)
            output = 0.0
        outputs.append(output)

    y_predict = np.array(outputs)
    y_test    = np.array(gold)

    return {
        'model_name': model_name,
        'full_outputs': full_outputs,
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }



