# How to add a new dataset

At a high-level, the process is as follows:

(1) Create a new dataset function in `dataset_utils.py`

(2) Modify the dictionary in `get_dataset` from `dataset_utils.py` to contain that new dataset. For example:
```python
'my_new_function': lambda random_state=1, max_train=64, max_test=32, **kwargs: my_new_function(random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),

```

## Analyzing an existing function
Let's take the definition of `get_original1(<..>)` and analyze it:
```python
def get_original1(random_state=1, max_train=64, max_test=32, **kwargs):
    generator = np.random.RandomState(random_state)

    n_samples = max_train + max_test

    x = generator.uniform(size=(n_samples, 1), low=0, high=100)

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x, round_value)
        y_fn = lambda x: np.round(x[0] + 10*np.sin(x[0]/100 * np.pi * 5) + 10*np.cos(x[0]/100 * np.pi * 6), round_value)
    else:
        y_fn = lambda x: x[0] + 10*np.sin(x[0]/100 * np.pi * 5) + 10*np.cos(x[0]/100 * np.pi * 6)

    y = np.array([y_fn(point) for point in x])

    r_data   = x
    r_values = y

    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_test, random_state=random_state)

    x_train = x_train.iloc[:max_train]
    y_train = y_train.iloc[:max_train]
    x_test  = x_test.iloc[:max_test]
    y_test  = y_test.iloc[:max_test]

    if kwargs.get('sort_data'):
        df_train = pd.concat([x_train, pd.DataFrame(y_train, columns=['Output'])], axis=1).sort_values('Feature 0')
        df_test  = pd.concat([x_test, pd.DataFrame(y_test, columns=['Output'])], axis=1).sort_values('Feature 0')
        x_train  = df_train.drop(['Output'], axis=1)
        y_train  = df_train['Output']
        x_test   = df_test.drop(['Output'], axis=1)
        y_test   = df_test['Output']
    

    
    return ((x_train, x_test, y_train, y_test), y_fn)


```

It works as follows. 

(1) Create x and y
In the case of synthetic data, sample x, then write a custom y_fn, then create y.
For example:
```python
x = generator.uniform(size=(n_samples, 1), low=0, high=100)
y_fn = lambda x: x[0] + 10*np.sin(x[0]/100 * np.pi * 5) + 10*np.cos(x[0]/100 * np.pi * 6)
y = np.array([y_fn(point) for point in x])
```

(2) Create a pandas dataframe, then shuffle
```python
# Create dataframe
df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
# Create train and test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_test, random_state=random_state)
# Only keep the maximum amount of items asked by the user
x_train = x_train.iloc[:max_train]
y_train = y_train.iloc[:max_train]
x_test  = x_test.iloc[:max_test]
y_test  = y_test.iloc[:max_test]
```

(3) Return data + y_fn
```python
return ((x_train, x_test, y_train, y_test), y_fn)
```


## Adding a random synthetic dataset
The easiest way is to simply copy the `get_original1`, then modify `y_fn`. Please notice that `y_fn` appears twice, in `if kwargs.get('round', False)` branch and in the `else` branch. For example:
```python
if kwargs.get('round', False):
    round_value = kwargs.get('round_value', 2)
    x = np.round(x, round_value)
    y_fn = lambda x: np.round(x[0] + x[0] ** 2 + np.sin(x[0]), round_value)
else:
    y_fn = lambda x: x[0] + x[0] ** 2 + np.sin(x[0])
```

## Adding a dataset from a file
Say that you have a `csv` file that you want to play with. The way to do it is as follows:

(1) Read the data
```python
data = pd.read_csv(<..>)
```

(2) Select x and y
```python
x = data[['House Age', 'Latitude', 'Longitude']].to_numpy()
y = data[['Price']].to_numpy().reshape(-1)
```

After this, most of the code from any function from `dataset_utils.py` can be used as inspiration:
```python
r_data   = x
r_values = y
<..>
```