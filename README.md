# LLMs Can Do Regression
This project explores the extent to which LLMs can do regression.

## Models
We use three types of models:
- Large Language Models (e.g., GPT-4, Claude 3, DBRX, Llama, etc)
- Traditional Supervised Methods (e.g., Linear Regression, Gradient Boosting, Random Forest, KNN, etc)
- Unsupervised Heuristics (e.g., just predict the average, etc)

We describe them in greater detail below.



### LLM
We use over 10 large language models (LLMs), either through pay-per-token services or deployed locally.
We also show their rank, as compared with *all* other models. The use 🥇 for the best LLM, 🥈 for the second best, and 🥉 for the third best. 

Note: 🏆 Claude 3 Opus 🏆 is the best model **overall** on non-linear datasets, outperforming all other models (LLMs or supervised). It ranks second place overall (across all datasets), only behind `Linear Regression + Poly`.

| LLM                             | How was used                                      | Additional details                                                              | Average Rank Across Linear Datasets | Average rank Across Non-Linear Datasets |
| ------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------- | --------------------------------------- |
| GPT-4                           | OpenAI API                                        | `gpt-4-0125-preview`                                                            | 🥈12.50🥈                               | 🥉12.90🥉                                   |
| Chat GPT                        | OpenAI API                                        | `gpt-3.5-turbo-1106`                                                            | 27.50                               | 27.70                                   |
| Davinci 002                     | OpenAI API                                        | `davinci-002`                                                                   | 24.50                               | 24.75                                   |
| Babbage 002                     | OpenAI API                                        | `babbage-002`                                                                   | 37.16                               | 31.00                                   |
| Claude 3 Opus                   | OpenRouter                                        | `anthropic/claude-3-opus`                                                       | 🥇7.66🥇                                | 🥇7.80🥇                                    |
| Claude 3 Sonnet                 | OpenRouter                                        | `anthropic/claude-3-sonnet`                                                     | 🥉12.66🥉                               | 🥈9.40🥈                                    |
| Claude 3 Haiku                  | OpenRouter                                        | `anthropic/claude-3-haiku`                                                      | 12.83                               | 19.00                                   |
| Claude 2.1                      | OpenRouter                                        | `anthropic/claude-2.1`                                                          | 42.00                               | 41.28                                   |
| Claude 2.0                      | OpenRouter                                        | `anthropic/claude-2.0`                                                          | 40.50                               | 36.14                                   |
| Claude 1.2                      | OpenRouter                                        | `anthropic/claude-1.2`                                                          | 42.16                               | 38.42                                   |
| Gemini Pro                      | OpenRouter                                        | `google/gemini-pro`                                                             | 15.66                               | 17.70                                   |
| Mistral Medium                  | OpenRouter                                        | `mistralai/mistral-medium`                                                      | 21.33                               | 20.00                                   |
| Cohere Command                  | OpenRouter                                        | `cohere/command`                                                                | 44.33                               | 46.42                                   |
| Cohere Command R                | OpenRouter                                        | `cohere/command-r`                                                              | 36.50                               | 32.85                                   |
| Cohere Command R Plus           | OpenRouter                                        | `cohere/command-r-plus`                                                         | 23.00                               | 26.30                                   |
| StripedHyena Nous 7B            | OpenRouter                                        | `togethercomputer/stripedhyena-nous-7b`                                         |                                     |                                         |
| DBRX                            | Fireworks                                         | `accounts/fireworks/models/dbrx-instruct`                                       | 17.33                               | 15.50                                   |
| Mixtral Mixture of Experts 8x7B | DeepInfra                                         | `mistralai/Mixtral-8x7B-Instruct-v0.1`                                          | 25.33                               | 22.50                                   |
| Mistral 7B                      | DeepInfra                                         | `mistralai/Mistral-7B-Instruct-v0.1`                                            | 34.33                               | 33.80                                   |
| Llama 2 70B Chat                | DeepInfra                                         | `meta-llama/Llama-2-70b-chat-hf`                                                | 29.66                               | 30.10                                   |
| Code Llama 2 70B Instruct       | DeepInfra                                         | `codellama/CodeLlama-70b-Instruct-hf`                                           | 22.50                               | 21.50                                   |
| Yi 34B Chat                     | DeepInfra                                         | `01-ai/Yi-34B-Chat`                                                             | 26.00                               | 22.20                                   |
| Falcon 40B                      | Locally with TGI                                  | `tiiuae/falcon-40b` quantized to 8bits with `bitsandbytes` through TGI          | 31.66                               | 20.00                                   |
| Falcon 40B Instruct             | Locally with TGI                                  | `tiiuae/falcon-40b-instruct` quantized to 8bits with `bitsandbytes` through TGI | 34.33                               | 23.00                                   |
| RWKV v4 14B                     | Locally with Huggingface (`AutoModelForCausalLM`) | `rwkv-v4-14b`                                                                   |                                     |                                         |



### Traditional Supervised Methods
We use traditional supervised methods typically used for regression. We use models found in sklearn. We include in additional details the model name and any default parameter changes.
We used `<..>` for some parameters that are omitted for brevity (e.g., random state)
| Model Name               | Additional Details                                                                                                    | Average Rank Across Linear Datasets | Average rank Across Non-Linear Datasets |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | --------------------------------------- |
| Linear Regression        | `LinearRegression`                                                                                                    | 🥇1.16🥇                                | 18.20                                   |
| Ridge                    | `Ridge`                                                                                                               | 14.66                               | 17.60                                   |
| Lasso                    | `Lasso`                                                                                                               | 12.83                               | 26.10                                   |
| MLP Wide 1               | `MLPRegressor(hidden_layer_sizes=(10, ), activation='relu', <..>)`                                                    | 🥉4.66🥉                                | 16.00                                   |
| MLP Wide 2               | `MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', <..>)`                                                   | 7.83                                | 19.10                                   |
| MLP Wide 3               | `MLPRegressor(hidden_layer_sizes=(1000, ), activation='relu', <..>)`                                                  | 6.16                                | 19.00                                   |
| MLP Deep 1               | `MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', <..>)`                                                  | 5.66                                | 18.50                                   |
| MLP Deep 2               | `MLPRegressor(hidden_layer_sizes=(10, 20, 10), activation='relu', <..>)`                                              | 8.33                                | 17.60                                   |
| MLP Deep 3               | `MLPRegressor(hidden_layer_sizes=(10, 20, 30, 20, 10), activation='relu', <..>)`                                      | 10.50                               | 16.90                                   |
| Random Forest            | `RandomForestRegressor(max_depth=3, <..>)`                                                                            | 29.66                               | 15.30                                   |
| Bagging                  | `BaggingRegressor`                                                                                                    | 25.16                               | 🥉12.00🥉                                   |
| Gradient Boosting        | `GradientBoostingRegressor`                                                                                           | 23.00                               | 🥇8.40🥇                                    |
| AdaBoost                 | `AdaBoostRegressor(n_estimators=100, <..>)`                                                                           | 27.16                               | 15.10                                   |
| SVM                      | `SVR`                                                                                                                 | 42.00                               | 25.60                                   |
| SVM + Scaler             | `make_pipeline(StandardScaler(), SVR())`                                                                              | 42.66                               | 24.70                                   |
| KNN v1                   | `KNeighborsRegressor`                                                                                                 | 27.66                               | 17.00                                   |
| KNN v2                   | `KNeighborsRegressor(weights='distance')`                                                                             | 26.50                               | 18.80                                   |
| Kernel Ridge             | `KernelRidge`                                                                                                         | 13.50                               | 27.50                                   |
| Linear Regression + Poly | `Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', LinearRegression())])`                             | 🥈2.50🥈                                | 🥈9.80🥈                                    |
| Spline                   | `Pipeline([('spline', SplineTransformer(n_knots=n_knots, degree=degree)), ('linear', LinearRegression())])`           | 13.33                               | 19.7                                    |
| KNN v3                   | `KNeighborsRegressor(n_neighbors=3, weights='distance')`                                                              |                                     |                                         |
| KNN v4                   | `KNeighborsRegressor(n_neighbors=1, weights='distance')`                                                              |                                     |                                         |
| KNN v5                   | `KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')` (`n_neigbors` depends on the number of datapoints) |                                     |                                         |

### Unsupervised Heuristics
We use heuristic-inspired baselines.
| Name    | Additional Details                                                                   |
| ------- | ------------------------------------------------------------------------------------ |
| Average | Predict the average output of the train partition                                    |
| Last    | Predict the value corresponding to the last value in the train partition             |
| Random  | Predict the value corresponding to a randomly sampled value from the train partition |


## Datasets
| Name              | Additional Details                                                                                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Regression NI 1/1 | A random linear regression dataset with 1 informative variable and 1 total variable ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html))   |
| Regression NI 1/2 | A random linear regression dataset with 1 informative variable and 2 total variables ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html))  |
| Regression NI 1/3 | A random linear regression dataset with 1 informative variable and 3 total variables ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html))  |
| Regression NI 2/2 | A random linear regression dataset with 2 informative variables and 2 total variables ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)) |
| Regression NI 2/3 | A random linear regression dataset with 2 informative variables and 3 total variables ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)) |
| Regression NI 3/3 | A random linear regression dataset with 3 informative variables and 3 total variables ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)) |
| Friedman #1       | The Friedman #1 dataset ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html))                                                                |
| Friedman #2       | The Friedman #2 dataset ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman2.html))                                                                |
| Friedman #3       | The Friedman #3 dataset ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman3.html))                                                                |
| Original #1       | A dataset with a single input variable, similar to a line with oscillations (by adding `sin` and `cos`)                                                                                    |
| Original #2       | A dataset inspired by Friedman #2, but changing the domain of the input variable and some operants (e.g., $^2 \rightarrow $^4)                                                             |
| Original #3       | Trying more operands (e.g., $e^x$)                                                                                                                                                         |
| Original #4       | Trying more operands together (sin, cos, log, sqrt, fractions)                                                                                                                             |
| Original #5       | Trying softmax                                                                                                                                                                             |
| Simple NN 1       | Initializing a random neural network and running it over random input. The output is considered gold                                                                                       |
| Transformer 1     | Initializing a random transformer encoder block and running random data. The output is considered gold                                                                                     |
| Character         | Mapping random characters (e.g., `a`) to a numeric value. Then sampling a vector to map back the characters                                                                                |

## Results At A Glance
Overall, LLMs *generally* outperform the unsupervised heuristics, suggesting that the in-context learning mechanism is more complex than such simple heuristics.

Selected LLMs, both private (e.g., Claude 3 Opus, GPT-4) and open (e.g., DBRX) can outperform supervised methods such as KNN, Gradient Boosting, or Random Forest. For example, except on the datasets derived from neural networks (and `Original 4`), Claude 3 Opus outperforms KNN, Gradient Boosting, and Random Forest on **all** datasets.

### Rank Heatmap

![Rank Heatmap Datasets](heatmap_all.png "Rank Heatmap")


### Adaptation


## How to

### How to add a new dataset?
Please check `hot_to_dataset.md`

### How to add a new model?
Please check `hot_to_model.md`