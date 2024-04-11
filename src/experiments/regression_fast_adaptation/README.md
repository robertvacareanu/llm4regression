## What
Experiment to analyze how the performance of LLMs scale with the number of in-context examples.

## Why
Borrowing from the online learning community, we check whether LLMs are able to improve their prediction, approaching the performance of the best model in hindsight. Ideally, a strong LLM should improve its prediction with the number of examples in-context. Intuitively, when you do not have many (input, output) examples for a regression task, there are many potential functions that could have generated that dataset. But by increasing the number of examples, we decrease the number of potential functions. We investigate whether the LLMs are capable of capturing this.

## How
Generate a dataset of 101 examples. Give the first one to the model, and ask to predict the output corresponding to example #2. Then give first two examples and ask the model to predict the output corresponding to example #3. This process is similarly repeated for all 101 examples.
