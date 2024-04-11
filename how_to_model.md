# How to add a new model

Out of the box, there is code for the following types of models.

API Requests
- (1) OpenAI
- (2) DeepInfra
- (3) OpenRouter
- (4) Fireworks

Local Models
- (1) Deployed with TGI
- (2) Deployed as `AutoModelForCausalLM`


Performing (additional) experiments with models from one of the services above, even if different models, is easy. For example, the code below uses `gpt-4-0125-preview`. To change this to `gpt-3.5-turbo-1106`, just change the parameter of `ChatOpenAI`.
```python
llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)
model_name = 'gpt4-turbo'
```

To use non-chat models, please use `OpenAI(model_name="davinci-002", temperature=0)`. Note, however, that this might entail additional changes.

## Add a new model (LLM as a service)

In the case that you want to add a model from a service that is not in the code, you will need to write a new file in `src/regressors`. The structure is that it contains a function with the following signature
```python
def llm_regression(llm, x_train, x_test, y_train, y_test, encoding_type, add_instr_prefix=False, instr_prefix='The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.\n\n\n\n\n'):
```

The first parameter is the llm. In this function, you will call the llm with the prompt:
```python
if add_instr_prefix:
    inpt = instr_prefix + fspt.format(**x)
else:
    inpt = fspt.format(**x)
output = llm.call_as_llm(inpt, stop=['\n'], max_tokens=12).strip().split('\n')[0].strip()
```

Depending on the llm, it might not work with `llm.call_as_llm`, or you might need to write a particular class in case that service is not supported out of the box by langchain. There is an example of this in `src/regressors/openrouter_llm_regressor.py`, where I wrote specific code for openrouter (`ChatOpenRouter`).