"""
For OpenRouter
"""

from langchain_community.chat_models import ChatOpenAI

import numpy as np
import os
import tqdm
from langchain.chat_models import ChatOpenAI
from src.regressors.prompts import construct_few_shot_prompt
from typing import Optional

class ChatOpenRouter(ChatOpenAI):
    """
    OpenRouter uses same API as OpenAI
    See: https://medium.com/@gal.peretz/openrouter-langchain-leverage-opensource-models-without-the-ops-hassle-9ffbf0016da7
    """
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self,
                 model_name: str,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)


def llm_regression(llm, x_train, x_test, y_train, y_test, encoding_type, model_name, max_tokens=64, add_instr_prefix=False, instr_prefix='The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.\n\n\n\n\n'):
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
        
        output = llm.call_as_llm(inpt, max_tokens=max_tokens)
        
        full_outputs.append(output)
        output=output.strip().split('\n')[0].strip() # Similar to OpenAI

        # time.sleep(1.0)
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



