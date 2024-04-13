import random
import string
import pandas as pd
import numpy as np
import scipy.special as spp
from sklearn.datasets import make_regression, make_friedman1, make_friedman2, make_friedman3, make_sparse_uncorrelated
from sklearn.model_selection import train_test_split
from scipy.special import softmax


def get_real_estate_data(path='data/real_estate/real_estate.csv', random_state=1, max_train=64, max_test=32, **kwargs):
    df = pd.read_csv(path).drop(['No'], axis=1)
    df.rename(columns={
        'X1 transaction date'                   : 'Transaction date',
        'X2 house age'                          : 'House age',
        'X3 distance to the nearest MRT station': 'Distance to nearest station',
        'X4 number of convenience stores'       : 'Number of convenience stores',
        'X5 latitude'                           : 'Latitude',
        'X6 longitude'                          : 'Longitude',
        'Y house price of unit area'            : 'Price',
    }, inplace=True)

    x=df.drop(['Price', 'Latitude', 'Longitude', 'Transaction date'], axis=1)
    y=df['Price']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_test, random_state=random_state)

    x_train = x_train.iloc[:max_train]
    y_train = y_train.iloc[:max_train]
    x_test  = x_test.iloc[:max_test]
    y_test  = y_test.iloc[:max_test]
    
    return ((x_train, x_test, y_train, y_test), None)


def get_regression(n_features=5, n_informative=1, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs):
    r_data, r_values, coeffs = make_regression(n_samples=max_train + max_test, n_features=n_features, n_informative=n_informative, noise=noise, random_state=random_state, coef=True)
    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x, round_value)
        y_fn = lambda x: np.round(np.dot(x, coeffs), round_value)
        y    = np.round(y, round_value)
    else:
        y_fn = lambda x: np.dot(x, coeffs)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_test, random_state=random_state)

    x_train = x_train.iloc[:max_train]
    y_train = y_train.iloc[:max_train]
    x_test  = x_test.iloc[:max_test]
    y_test  = y_test.iloc[:max_test]

    if kwargs.get('print_coeffs', False):
        print(coeffs)
    
    return ((x_train, x_test, y_train, y_test), y_fn)

def get_friedman1(n_features=5, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs):
    r_data, r_values = make_friedman1(n_samples=max_train + max_test, n_features=n_features, noise=noise, random_state=random_state)

    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})

    x = df.drop(['Output'], axis=1)
    y = df['Output']

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x    = np.round(x, round_value)
        y_fn = lambda x: np.round(10 * np.sin(np.pi * x[0] * x[1]) + 20 * (x[2] - 0.5) ** 2 + 10 * x[3] + 5 * x[4], round_value)
        y    = np.round(y, round_value)
    else:
        y_fn = lambda x: 10 * np.sin(np.pi * x[0] * x[1]) + 20 * (x[2] - 0.5) ** 2 + 10 * x[3] + 5 * x[4]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_test, random_state=random_state)

    x_train = x_train.iloc[:max_train]
    y_train = y_train.iloc[:max_train]
    x_test  = x_test.iloc[:max_test]
    y_test  = y_test.iloc[:max_test]
    
    
    
    return ((x_train, x_test, y_train, y_test), y_fn)

def get_friedman2(noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs):
    r_data, r_values = make_friedman2(n_samples=max_train + max_test, noise=noise, random_state=random_state)

    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x    = np.round(x, round_value)
        y_fn = lambda x: np.round((x[0] ** 2 + (x[1] * x[2] - 1 / (x[1] * x[3])) ** 2) ** 0.5, round_value)
        y    = np.round(y, round_value)
    else:
        y_fn = lambda x: (x[0] ** 2 + (x[1] * x[2] - 1 / (x[1] * x[3])) ** 2) ** 0.5


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_test, random_state=random_state)

    x_train = x_train.iloc[:max_train]
    y_train = y_train.iloc[:max_train]
    x_test  = x_test.iloc[:max_test]
    y_test  = y_test.iloc[:max_test]
    
    

    return ((x_train, x_test, y_train, y_test), y_fn)

def get_friedman3(noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs):
    r_data, r_values = make_friedman3(n_samples=max_train + max_test, noise=noise, random_state=random_state)

    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x    = np.round(x, round_value)
        y_fn = lambda x: np.round(np.arctan((x[1] * x[2] - 1 / (x[1] * x[3])) / x[0]), round_value)
        y    = np.round(y, round_value)
    else:
        y_fn = lambda x: np.arctan((x[1] * x[2] - 1 / (x[1] * x[3])) / x[0])


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_test, random_state=random_state)

    x_train = x_train.iloc[:max_train]
    y_train = y_train.iloc[:max_train]
    x_test  = x_test.iloc[:max_test]
    y_test  = y_test.iloc[:max_test]


    return ((x_train, x_test, y_train, y_test), y_fn)

def get_sparse_uncorrelated(random_state=1, max_train=64, max_test=32, shuffle_columns=False):
    r_data, r_values = make_sparse_uncorrelated(n_samples=max_train + max_test, random_state=random_state)
    if shuffle_columns:
        r = random.Random(random_state)
        r_data = r_data[:, r.choices(range(r_data.shape[1]), k=r_data.shape[1])]
    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_test, random_state=random_state)

    x_train = x_train.iloc[:max_train]
    y_train = y_train.iloc[:max_train]
    x_test  = x_test.iloc[:max_test]
    y_test  = y_test.iloc[:max_test]
    
    return ((x_train, x_test, y_train, y_test), None)


def get_original1_deprecated(random_state=1, max_train=64, max_test=32, **kwargs):
    """
    Like a line, but with some giggles
    Deprecated this one in favor of one which has a broader domain for x.
    """
    generator = np.random.RandomState(random_state)

    n_samples = max_train + max_test

    x = generator.uniform(size=(n_samples, 1))

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x, round_value)
        y_fn = lambda x: np.round(x[0] * 10 + np.sin(x[0] * np.pi * 5) + np.cos(x[0] * np.pi * 6), round_value)
    else:
        y_fn = lambda x: x[0] * 10 + np.sin(x[0] * np.pi * 5) + np.cos(x[0] * np.pi * 6)

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

def get_original1(random_state=1, max_train=64, max_test=32, **kwargs):
    """
    Like a line, but with some giggles
    """
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

def get_original2(random_state=1, max_train=64, max_test=32, noise_level=0.0, **kwargs):
    """
    Adapted from Friedman 2
    """
    generator = np.random.RandomState(random_state)

    n_samples = max_train + max_test

    x = generator.uniform(size=(n_samples, 4))
    x[:, 0] *= 3
    x[:, 1] *= 52 * np.pi
    x[:, 1] += 4 * np.pi
    x[:, 2] *= 2
    x[:, 3] *= 10
    x[:, 3] += 1

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x, round_value)
        y_fn = lambda x: np.round((x[0] ** 4 + (x[1] * x[2] - 2 / (np.sqrt(x[1]) * np.sqrt(x[3]))) ** 2) ** 0.75, round_value)
    else:
        y_fn = lambda x: (x[0] ** 4 + (x[1] * x[2] - 2 / (np.sqrt(x[1]) * np.sqrt(x[3]))) ** 2) ** 0.75


    y = np.array([y_fn(point) for point in x]) + noise_level * generator.standard_normal(size=(n_samples))

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
    
    return ((x_train, x_test, y_train, y_test), y_fn)

def get_random_nn1(random_state=1, max_train=64, max_test=32, **kwargs):
    """
    Initialize a neural network. This will serve as as the regression function f
    Create a dataset by randomly sampling data, then running it through f (i.e., the neural network).
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(random_state)

    n_samples = max_train + max_test

    input_dimension         = kwargs.get('input_dimension', 5)
    hidden_dimension        = kwargs.get('hidden_dimension', 5)
    output_dimension        = kwargs.get('output_dimension', 1)
    number_of_layers = kwargs.get('layers', 1)
    assert(number_of_layers > 0)

    sequential_nn = []

    # First Layer
    sequential_nn += [
        nn.Linear(in_features=input_dimension, out_features=hidden_dimension),
        nn.ReLU(),
    ]


    for layer in range(number_of_layers-1):
        sequential_nn += [
            nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension),
            nn.ReLU(),
        ]

    
    network = nn.Sequential(
        *sequential_nn,
        nn.Linear(in_features=hidden_dimension, out_features=output_dimension),
    ).eval()

    x = torch.randn(n_samples, input_dimension)
    
    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x.detach().cpu().numpy(), round_value)
        y = np.round(network(torch.tensor(x)).squeeze(dim=1).detach().cpu().numpy().tolist(), round_value)
        x = np.round(x.tolist(), 2)
        y_fn = lambda x: np.round(network(x).detach().cpu().numpy(), round_value)
    else:
        y = network(x).squeeze(dim=1).detach().cpu().numpy()
        y_fn = lambda x: network(x).detach().cpu().numpy()
        
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
    
    
    return ((x_train, x_test, y_train, y_test), y_fn)

def get_random_nn1_scaled(random_state=1, max_train=64, max_test=32, **kwargs):
    """
    Initialize a neural network. This will serve as as the regression function f
    Create a dataset by randomly sampling data, then running it through f (i.e., the neural network).
    Different from `get_random_nn1`, scale the output
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(random_state)

    n_samples = max_train + max_test

    input_dimension         = kwargs.get('input_dimension', 5)
    hidden_dimension        = kwargs.get('hidden_dimension', 5)
    output_dimension        = kwargs.get('output_dimension', 1)
    number_of_layers = kwargs.get('layers', 1)
    assert(number_of_layers > 0)

    sequential_nn = []

    # First Layer
    sequential_nn += [
        nn.Linear(in_features=input_dimension, out_features=hidden_dimension),
        nn.ReLU(),
    ]


    for layer in range(number_of_layers-1):
        sequential_nn += [
            nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension),
            nn.ReLU(),
        ]

    
    network = nn.Sequential(
        *sequential_nn,
        nn.Linear(in_features=hidden_dimension, out_features=output_dimension),
    ).eval()

    x = torch.randn(n_samples, input_dimension)
    x *= 10
    
    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x.detach().cpu().numpy(), round_value)
        y = np.round((10 * network(torch.tensor(x))).squeeze(dim=1).detach().cpu().numpy().tolist(), round_value)
        x = np.round(x.tolist(), 2)
        y_fn = lambda x: np.round((10 * network(x)).detach().cpu().numpy(), round_value)
    else:
        y = (10 * network(x)).squeeze(dim=1).detach().cpu().numpy()
        y_fn = lambda x: (10 * network(x)).detach().cpu().numpy()
        
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
    
    
    return ((x_train, x_test, y_train, y_test), y_fn)

def get_random_nn2(random_state=1, max_train=64, max_test=32, **kwargs):
    """
    Similar to `get_random_nn1`, but define fancier neural networks (e.g., with skip connections, layer normalization, etc.)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LinearLayerWithExtras(nn.Module):
        def __init__(self, in_features, intermediary_features, out_features, activation='relu', **kwargs):
            super().__init__()
            self.ll1 = nn.Linear(in_features=in_features, out_features=intermediary_features, **kwargs.get('ll_kwargs', {}))
            self.ll2 = nn.Linear(in_features=intermediary_features, out_features=out_features, **kwargs.get('ll_kwargs', {}))
            self.activation_fn_name = activation
            if activation == 'relu':
                self.activation_fn = F.relu
            elif activation == 'gelu':
                self.activation_fn = F.gelu
            else:
                raise ValueError(f"Only `relu` and `gelu` are supported, but the value is {activation}. Is everything ok?")

            self.extra_args = kwargs

            if self.extra_args.get('ln', False):
                self.ln = nn.LayerNorm(intermediary_features)

        def forward(self, x):
            y1 = self.activation_fn(self.ll1(x))
            if self.extra_args.get('ln', False):
                y1 = self.ln(y1)

            if self.extra_args.get('skip', False):
                y2 = x + self.ll2(y1)
            else:
                y2 = self.ll2(y1)

            return self.activation_fn(y2)

    torch.manual_seed(random_state)

    n_samples = max_train + max_test

    input_dimension         = kwargs.get('input_dimension', 5)
    hidden_dimension        = kwargs.get('hidden_dimension', 5)
    output_dimension        = kwargs.get('output_dimension', 1)

    number_of_layers = kwargs.get('layers', 1)
    add_skip         = kwargs.get('skip', False)
    add_ln           = kwargs.get('ln', False)

    assert(number_of_layers > 0)
    assert(add_skip | add_ln) # Does not make sense to not use any of the special operations

    sequential_nn = []
    for layer in range(number_of_layers):
        sequential_nn.append(
            LinearLayerWithExtras(input_dimension, hidden_dimension, input_dimension, 'gelu', **{'skip': add_skip, 'ln': add_ln})
        )
    
    network = nn.Sequential(
        *sequential_nn,
        nn.Linear(in_features=input_dimension, out_features=output_dimension),
    ).eval()

    x = torch.randn(n_samples, input_dimension)
    
    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x.detach().cpu().numpy(), round_value)
        y = np.round(network(torch.tensor(x)).squeeze(dim=1).detach().cpu().numpy().tolist(), round_value)
        x = np.round(x.tolist(), 2)
        y_fn = lambda x: np.round(network(x).detach().cpu().numpy(), round_value)
    else:
        y = network(x).squeeze(dim=1).detach().cpu().numpy()
        y_fn = lambda x: network(x).detach().cpu().numpy()
        
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
    
    
    return ((x_train, x_test, y_train, y_test), y_fn)


def get_random_transformer(random_state=1, max_train=64, max_test=32, **kwargs):
    """
    Initialize a random transformer encoder block. This will serve as as the regression function f
    Sample random data, then run it through f (i.e., the transformer) to create the dataset
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(random_state)

    n_samples = max_train + max_test

    input_dimension         = kwargs.get('input_dimension', 5)
    output_dimension        = kwargs.get('output_dimension', 1)

    nhead = kwargs.get('nhead', 1)

    number_of_layers = kwargs.get('layers', 1)
    
    tel = nn.TransformerEncoderLayer(d_model=input_dimension, nhead=nhead)
    te  = nn.TransformerEncoder(tel, num_layers=number_of_layers)


    network = nn.Sequential(
        te,
        nn.Linear(in_features=input_dimension, out_features=output_dimension),
    ).eval()

    x = torch.randn(n_samples, input_dimension)
    
    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x.detach().cpu().numpy(), round_value)
        y = np.round(network(torch.tensor(x)).squeeze(dim=1).detach().cpu().numpy().tolist(), round_value)
        x = np.round(x.tolist(), 2)
        y_fn = lambda x: np.round(network(x).detach().cpu().numpy(), round_value)
    else:
        y = network(x).squeeze(dim=1).detach().cpu().numpy()
        y_fn = lambda x: network(x).detach().cpu().numpy()
        
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
    
    
    return ((x_train, x_test, y_train, y_test), y_fn)


def get_character_regression(random_state=1, max_train=64, max_test=32, **kwargs):
    """
    Instead of having numbers as input and number as output, have characters as input and number as output
    Essentially, sample a subset of characters (to keep the number of necessary input examples small), 
    then assign a random numeric value to each character
    Then, sample a random weight vector
    The output is created by doing a dot product
    """
    r = random.Random(random_state)
    # r.sample(<..>, k=len(<..>)) -> Like shuffle, but shuffle does not return anything; Here, it does.
    letter_to_index = [(c, i) for (c, i) in enumerate(r.sample(list(string.ascii_lowercase), k=len(string.ascii_lowercase)))]
    generator = np.random.RandomState(random_state)

    n_samples = max_train + max_test
    max_number_of_variables = kwargs.get('max_number_of_variables', 5)

    letters_to_use = r.sample(letter_to_index, max_number_of_variables)
    sample = r.choices(letters_to_use, k=max_number_of_variables * n_samples)
    sample = np.array(sample)[:, 1].reshape(-1, max_number_of_variables)

    weight_vector = 10 * generator.uniform(size=(max_number_of_variables))


    index_to_letter = dict(letters_to_use)
    letter_to_index = {v:k for (k, v) in index_to_letter.items()}

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        y_fn = lambda x: np.round(weight_vector @ np.array([letter_to_index[c] for c in x]), round_value)
    else:
        y_fn = lambda x: weight_vector @ np.array([letter_to_index[c] for c in x])
    
    y = [y_fn(x) for x in sample]

    r_data   = sample
    r_values = y

    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_test, random_state=random_state)

    x_train = x_train.iloc[:max_train]
    y_train = y_train.iloc[:max_train]
    x_test  = x_test.iloc[:max_test]
    y_test  = y_test.iloc[:max_test]

    
    return ((x_train, x_test, y_train, y_test), y_fn)

def get_unlearnable1(random_state=1, max_train=64, max_test=32, **kwargs):
    """
    An unlearnable dataset
    The best you can do is to predict the mean
    """
    generator = np.random.RandomState(random_state)

    n_samples = max_train + max_test

    x = generator.uniform(size=(n_samples, 2))
    x[:, 0] *= 10
    x[:, 1] *= 10
    
    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x, round_value)

    y  = x[:, 1]

    r_data   = x[:, :1]
    r_values = y

    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_test, random_state=random_state)

    x_train = x_train.iloc[:max_train]
    y_train = y_train.iloc[:max_train]
    x_test  = x_test.iloc[:max_test]
    y_test  = y_test.iloc[:max_test]
    

    return ((x_train, x_test, y_train, y_test), None)

def get_unlearnable2(random_state=1, max_train=64, max_test=32, **kwargs):
    """
    An unlearnable dataset (similar to `get_unlearnable1`), but using numbers from random.org
    The best you can do is to predict the mean
    """

    with open('data/randomorg1.txt') as fin:
        x = fin.readlines()
        x = x[29:-7]
        x = [float(number.strip()) for number in x]

    with open('data/randomorg2.txt') as fin:
        y = fin.readlines()
        y = y[29:-7]
        y = [float(number.strip()) for number in y]

    r = random.Random(random_state)
    x = np.array(r.sample(x, k=len(x)))
    y = np.array(r.sample(y, k=len(y)))


    n_samples = max_train + max_test
    
    r_data   = x[:, np.newaxis]
    r_values = y

    df = pd.DataFrame({**{f'Feature {i}': r_data[:, i] for i in range(r_data.shape[1])}, 'Output': r_values})
    x = df.drop(['Output'], axis=1)
    y = df['Output']


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_test, random_state=random_state)

    x_train = x_train.iloc[:max_train]
    y_train = y_train.iloc[:max_train]
    x_test  = x_test.iloc[:max_test]
    y_test  = y_test.iloc[:max_test]
    

    return ((x_train, x_test, y_train, y_test), None)

def get_original3(random_state=1, max_train=64, max_test=32, noise_level=0.0, **kwargs):
    """
    A new original function
    Experimenting with other operators (e.g., e^x[0])
    """
    generator = np.random.RandomState(random_state)

    n_samples = max_train + max_test

    x = generator.uniform(size=(n_samples, 4))
    x[:, 0] *= 2
    x[:, 0] += 1
    x[:, 1] *= 9
    x[:, 1] += 1
    x[:, 2] *= 10
    x[:, 3] *= 19
    x[:, 3] += 1

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x, round_value)
        y_fn = lambda x: np.round(np.e ** x[0] + (x[1] * x[2]) / np.sqrt(x[3]) + ((x[3] * x[0]) ** 1.5), round_value)
    else:
        y_fn = lambda x: np.e ** x[0] + (x[1] * x[2]) / np.sqrt(x[3]) + ((x[3] * x[0]) ** 1.5)


    y = np.array([y_fn(point) for point in x]) + noise_level * generator.standard_normal(size=(n_samples))

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
    
    return ((x_train, x_test, y_train, y_test), y_fn)


def get_original4(random_state=1, max_train=64, max_test=32, noise_level=0.0, **kwargs):
    """
    A new original function
    Using many operators together (e.g., sin, cos, sqrt, log, fractions)
    """
    generator = np.random.RandomState(random_state)

    n_samples = max_train + max_test

    x = generator.uniform(size=(n_samples, 2))
    x *= 98
    x += 2

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x, round_value)
        y_fn = lambda x: np.round((x[1]/10) * np.sin(x[0]) + (x[0]/10) * np.cos(x[1]) + (np.sqrt(x[0]) * np.log(x[1])) / (np.sqrt(x[1]) * np.log(x[0])), round_value)
    else:
        y_fn = lambda x: (x[1]/10) * np.sin(x[0]) + (x[0]/10) * np.cos(x[1]) + (np.sqrt(x[0]) * np.log(x[1])) / (np.sqrt(x[1]) * np.log(x[0]))


    y = np.array([y_fn(point) for point in x]) + noise_level * generator.standard_normal(size=(n_samples))

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
    
    return ((x_train, x_test, y_train, y_test), y_fn)

def get_original5(random_state=1, max_train=64, max_test=32, noise_level=0.0, **kwargs):
    """
    A new original function
    Experimenting with softmax
    """
    generator = np.random.RandomState(random_state)

    n_samples = max_train + max_test

    x = generator.uniform(size=(n_samples, 3))
    x *= 50
    x -= 25

    if kwargs.get('round', False):
        round_value = kwargs.get('round_value', 2)
        x = np.round(x, round_value)
        y_fn = lambda x: np.round(100 * softmax(x/10, axis=-1).max(axis=-1), round_value)
    else:
        y_fn = lambda x: 100 * softmax(x/10, axis=-1).max(axis=-1)


    y = np.array([y_fn(point) for point in x]) + noise_level * generator.standard_normal(size=(n_samples))

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
    
    return ((x_train, x_test, y_train, y_test), y_fn)




def get_dataset(name: str):
    """
    Get a dataset by name

    :param name (str): The name of the dataset
    :returns a lambda function which, typically, given `random_state`, `max_train`, `max_test` will return a new dataset
    """

    dataset_to_fn = {
        'real_estate'    : lambda path='data/real_estate/real_estate.csv', random_state=1, max_train=64, max_test=32: get_real_estate_data(path=path, random_state=random_state, max_train=max_train, max_test=max_test),

        'regression'     : lambda n_features=5, n_informative=1, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=n_informative, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        # 1 informative out of 5
        'regression_ni1' : lambda n_features=5, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=1, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        # 2 informative out of 5
        'regression_ni2' : lambda n_features=5, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=2, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        # 3 informative out of 5
        'regression_ni3' : lambda n_features=5, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=3, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        # 'regression_ni3' : lambda n_features=5, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=3, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        # 1 informative out of 10
        'regression_ni1_10': lambda n_features=10, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=1, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        # 2 informative out of 10
        'regression_ni2_10': lambda n_features=10, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=2, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        # 3 informative out of 10
        'regression_ni3_10': lambda n_features=10, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=3, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),

        # 1 informative out of 1
        'regression_ni11' : lambda n_features=1, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=1, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        # 2 informative out of 2
        'regression_ni22' : lambda n_features=2, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=2, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        # 3 informative out of 3
        'regression_ni33' : lambda n_features=2, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=3, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),

        # 1 informative out of 2
        'regression_ni12' : lambda n_features=2, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=1, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),

        # 1 informative out of 3
        'regression_ni13' : lambda n_features=3, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=1, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        # 2 informative out of 3
        'regression_ni23' : lambda n_features=3, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_regression(n_features=n_features, n_informative=2, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),


        'friedman1'      : lambda n_features=5, noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs: get_friedman1(n_features=n_features, noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        'friedman2'      : lambda noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs              : get_friedman2(noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        'friedman3'      : lambda noise=0.5, random_state=1, max_train=64, max_test=32, **kwargs              : get_friedman3(noise=noise, random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),

        'sparse_uncorrelated': lambda random_state=1, max_train=64, max_test=32: get_sparse_uncorrelated(random_state=random_state, max_train=max_train, max_test=max_test),

        'original1': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_original1(random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        'original2': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_original2(random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        'original3': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_original3(random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        'original4': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_original4(random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        'original5': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_original5(random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        
        'simple_random_nn1': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_random_nn1(random_state=random_state, max_train=max_train, max_test=max_test, input_dimension=5, hidden_dimension=5, output_dimension=1, layers=1, **kwargs),
        'simple_random_nn2': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_random_nn1(random_state=random_state, max_train=max_train, max_test=max_test, input_dimension=5, hidden_dimension=10, output_dimension=1, layers=1, **kwargs),
        'simple_random_nn3': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_random_nn1(random_state=random_state, max_train=max_train, max_test=max_test, input_dimension=5, hidden_dimension=10, output_dimension=1, layers=2, **kwargs),

        'simple_random_nn1_scaled': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_random_nn1_scaled(random_state=random_state, max_train=max_train, max_test=max_test, input_dimension=5, hidden_dimension=5, output_dimension=1, layers=1, **kwargs),

        'more_complex_random_nn1': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_random_nn2(random_state=random_state, max_train=max_train, max_test=max_test, input_dimension=5, hidden_dimension=10, output_dimension=1, layers=1, skip=True, ln=True, **kwargs),
        'more_complex_random_nn2': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_random_nn2(random_state=random_state, max_train=max_train, max_test=max_test, input_dimension=5, hidden_dimension=10, output_dimension=1, layers=2, skip=True, ln=True, **kwargs),
        'more_complex_random_nn3': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_random_nn2(random_state=random_state, max_train=max_train, max_test=max_test, input_dimension=5, hidden_dimension=10, output_dimension=1, layers=3, skip=True, ln=True, **kwargs),

        'transformer1': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_random_transformer(random_state=random_state, max_train=max_train, max_test=max_test, input_dimension=5, output_dimension=1, nhead=1, layers=1, **kwargs),
        'transformer2': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_random_transformer(random_state=random_state, max_train=max_train, max_test=max_test, input_dimension=5, output_dimension=1, nhead=1, layers=2, **kwargs),

        
        'character_regression1': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_character_regression(random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        
        'unlearnable1': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_unlearnable1(random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        'unlearnable2': lambda random_state=1, max_train=64, max_test=32, **kwargs: get_unlearnable2(random_state=random_state, max_train=max_train, max_test=max_test, **kwargs),
        
    }

    return dataset_to_fn[name]




if __name__ == "__main__":
    """
    Simple example of how to use `get_dataset`.
    """
    (x_train, x_test, y_train, y_test), y_fn = get_dataset('real_estate')()
    print(x_train)
    print("\n")
    (x_train, x_test, y_train, y_test), y_fn = get_dataset('friedman1')()
    print(x_train)
    print("\n")
    (x_train, x_test, y_train, y_test), y_fn = get_dataset('original2')()
    print(x_train)
    print("\n")
    (x_train, x_test, y_train, y_test), y_fn = get_dataset('friedman1')(noise=0, shuffle_columns=False)
    print(y_fn(x_train.iloc[0].to_list()))
    print(y_train.iloc[0])