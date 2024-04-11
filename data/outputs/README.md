The outputs of all experiments, aggregated. For example, `lr_nlr.json` contains the data for linear and non-linear regression experiments.

Description of each file:
- `lr_nlr.json`: Data for linear and non-linear regression experiments, all models
- `knn_search.json`: Data of for variants of KNN on `Regression NI 2/2`, `Regression NI 1/3`, `Original 1`, `Original 2`, `Friedman 1`, `Friedman 2`, `Friedman 3`; Useful to verify the extent to which the performance of LLMs is comparable with *any* KNN
- `lr_nlr_rounding_to_5.json`: Results with GPT-4 and various sklearn models on `Friedman #1` and `Friedman #2`, but this time rounding everything to `5` decimals instead of `2`. The idea is to verify whether the results are affected by the number of decimals or not
- `hindsight.json`: The performance of sklearn models with hindsight information for the online learning experiments
- `online_learning`: The data for LLMs, Traditional Supervised Methods, and Unsupervised Methods when varying the number of examples from 1 to 100 (note that some LLMs cannot handle that, such as Llama2) 
- `plateauing`: The data for the experiments where we analyze how the performance of LLMs (GPT-4) scales with the number of examples. We ran up to 500 in-context examples.


In order to read it, you can use:
```python
import pandas as pd
df = pd.read_json('data/outputs/lr_nlr.json')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.groupby(by=['model', 'dataset']).agg({'l1': 'mean'}).unstack('model').T)
```

This should print `2.488320` for `AdaBoost` on `friedman1`, and `2.001900` for `Claude 3 Opus` on `friedman1`.

This dataframe can then be used to create many of the Figures presented in the pdf.

For example, to create a barplot similar to Figure 1, which contains the following models: `Claude 3`, `GPT-4`, `DBRX`, `Linear Regression`, `Gradient Boosting`, `KNN`, one can do:
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_json('data/outputs/lr_nlr.json')
# Subset of models
cdf = df[df['model'].isin(['Claude 3 Opus', 'GPT-4', 'DBRX', 'Linear Regression', 'Random Forest', 'Gradient Boosting', 'KNN'])]
# Only one dataset
cdf = cdf[cdf['dataset'] == 'regression_ni12']

sns.barplot(data=cdf, x='model', y='l1')
plt.xticks(rotation=30, ha='right')
plt.xlabel('Model', fontsize=14)
plt.ylabel('Mean Absolute Error', fontsize=14)
```

More details below.


### Checking the best model in hindsight

```python
hindsight_df = pd.read_json('data/outputs/hindsight.json')
cdf = hindsight_df.groupby(by=['dataset', 'model']).agg({'l1_avg': 'mean'})
idx = cdf.groupby('dataset')['l1_avg'].idxmin()
best_models = cdf.loc[idx].reset_index()
dataset_to_best_model = {x['dataset']: x for x in best_models.to_dict('records')}
display(best_models)
```
Best models should look like:
| dataset         | model              | l1_avg    |
| --------------- | ------------------ | --------- |
| friedman1       | mlp_uat3           | 0.075393  |        
| friedman2       | lr_with_polynomial | 4.122403  |        
| friedman3       | gb                 | 0.007092  |        
| original1       | gb                 | 0.027750  |        
| original2       | gb                 | 16.537801 |        
| regression_ni13 | mlp_uat3           | 0.016838  |        
| regression_ni22 | mlp_uat3           | 0.074845  |        


### How to do a heatmap

```python
# Part 1: Create the heatmap data
##########################

# What models to consider
model_order = [
    # LLMs
    'Claude 3 Opus',
    'Claude 3 Sonnet',
    'GPT-4',
    'Chat GPT',
    'Gemini Pro',
    'Mistral Medium',
    'Mixtral 8x22B',
    'Mixtral 8x7B',
    'Mistral 7B',
    'Code Llama 70B',
    'Llama2 70B Chat HF',
    'Yi 34B Chat',
    'DBRX',
    
    # Traditional Supervised Methods
    'AdaBoost',
    'Bagging',
    'Gradient Boosting',
    'KNN',
    'Kernel Ridge',
    'Lasso',
    'Linear Regression',
    'Linear Regression + Poly',
    'MLP Deep 1',
    'MLP Deep 2',
    'MLP Deep 3',
    'MLP Wide 1',
    'MLP Wide 2',
    'MLP Wide 3',
    'Random Forest',
    'Ridge',
    'SVM',
    'SVM + Scaler',
    'Spline',

    # Unsupervised Methods
    'Average',
    'Last',
    'Random',
]
 
# What datasets to consider
datasets = ['original1', 'original2', 'friedman1', 'friedman2', 'friedman3']

# Create a current dataframe (`cdf`) to hold only the necessary information
cdf = df[df['dataset'].isin(datasets)]
cdf = cdf[cdf['model'].isin(model_order)]
current_data = []

# Calculate the ranks
for (key, group) in cdf.groupby(by=['dataset', 'model']).agg({'l1': 'mean'}).reset_index().groupby(by=['dataset']):
    for idx, line in enumerate(group.sort_values(by=['l1']).to_dict('records')):
        current_data.append({
            'dataset':line['dataset'],
            'model': line['model'],
            'rank': idx + 1, # `+ 1` because we want the first element (i.e., `0`) to correspond to the first rank (the best)
        })

##########################

########################## Plotting
# Create a new dataframe with the ranks
cdf = pd.DataFrame(current_data)

# Create the heatmap data
heatmap_data = cdf.pivot(index='dataset', columns='model', values='rank')

# Create the heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(heatmap_data.reindex(columns=model_order), cmap="RdYlGn_r", annot=True, fmt="d", linewidths=0.5, cbar=False, annot_kws={"size": 14})

# Rotate the ticks
plt.xticks(fontsize=14, rotation=45, ha="right")
plt.yticks(fontsize=14, rotation=0)
plt.xlabel('Method', fontsize=16)
plt.ylabel('Dataset', fontsize=16)

plt.tight_layout()
ax = plt.gca()

# Annotate, to make it easier to read
plt.annotate("LLM", fontsize=12, rotation=0, xy=(0.23, 0.99), xycoords='figure fraction')
plt.annotate("Traditional Supervised Methods", fontsize=12, rotation=0, xy=(0.59, 0.99), xycoords='figure fraction')
plt.annotate("Unsupervised Methods", fontsize=12, rotation=0, xy=(0.901, 0.99), xycoords='figure fraction')

# number of models
number_of_llms = 13
number_of_tsm  = 19
number_of_um   = 3

from matplotlib.patches import Rectangle
eps = 0.01
rect1 = Rectangle((0, 0), number_of_llms, len(datasets), linewidth=7, edgecolor='black', facecolor='none')
rect2 = Rectangle((number_of_llms, 0), number_of_tsm, len(datasets), linewidth=7, edgecolor='black', facecolor='none')
rect3 = Rectangle((number_of_llms + number_of_tsm, 0), number_of_um, len(datasets), linewidth=7, edgecolor='black', facecolor='none')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)


plt.show()
########################## 


```