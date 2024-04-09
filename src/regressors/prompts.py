from langchain import PromptTemplate, FewShotPromptTemplate

def construct_examples_prompt(x_train, y_train):
    template = [feature + ": {" + f"{feature}" + "}" for feature in x_train.columns] + [y_train.name + ": {" + f"{y_train.name}" + "}"]
    template = "\n".join(template)
    example_prompt = PromptTemplate(
        template=template,
        input_variables=x_train.columns.to_list() + [y_train.name],
    )

    return example_prompt

def construct_few_shot_suffix_and_iv(x_train, y_train):
    template = [feature + ": {" + f"{feature}" + "}" for feature in x_train.columns] + [y_train.name + ":"]
    template = "\n".join(template)

    input_variables=x_train.columns.to_list()

    return (template, input_variables)

def construct_few_shot_prompt(x_train, y_train, x_test, encoding_type='vanilla'):
    examples = []
    for x1, x2 in zip(x_train.to_dict('records'), y_train):
        if encoding_type == "vanilla":
            output = x2
        else:
            raise ValueError(f"Unknown {encoding_type}")

        examples.append({**x1, y_train.name: output})

    examples_test = []
    for x1 in x_test.to_dict('records'):
        examples_test.append(x1)

    example_prompt = construct_examples_prompt(x_train, y_train)
    (template, input_variables) = construct_few_shot_suffix_and_iv(x_train, y_train)
    fspt = FewShotPromptTemplate(
        examples        =  examples,
        example_prompt  =  example_prompt,
        suffix          =  template,
        input_variables = input_variables,
    )

    return fspt

