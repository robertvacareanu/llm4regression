"""
For Fireworks
"""



import numpy as np
import time
from src.regressors.prompts import construct_few_shot_prompt

def llm_regression(llm, x_train, x_test, y_train, y_test, encoding_type, model_name, add_instr_prefix=False, instr_prefix='The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.\n\n\n\n\n'):
    examples_test = []
    for x1 in x_test.to_dict('records'):
        examples_test.append(x1)

    fspt = construct_few_shot_prompt(x_train, y_train, x_test, encoding_type=encoding_type)
    full_outputs = []
    outputs = []
    gold    = []
    for x, y in zip(examples_test, y_test):
        gold.append(y)
        if add_instr_prefix:
            inpt = instr_prefix + fspt.format(**x)
        else:
            inpt = fspt.format(**x)
        output = llm(inpt, stop=['\n'], temperature=0, max_tokens=12).strip().split('\n')[0].strip() # For everything else
        full_outputs.append(output)
        # output = llm(inpt, stop=['\n'], max_tokens=12).strip().split('\n')[0].strip() # For davinci
        time.sleep(0.1)
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

