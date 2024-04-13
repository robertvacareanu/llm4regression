import random
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, SplineTransformer
from sklearn.kernel_ridge import KernelRidge


def linear_regression(x_train, x_test, y_train, y_test, random_state=1):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'linear_regression',
        'x_train'  : x_train,
        'x_test'   : x_test,
        'y_train'  : y_train,
        'y_test'   : y_test,
        'y_predict': y_predict,
    }


def ridge(x_train, x_test, y_train, y_test, random_state=1):
    model = Ridge(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'ridge',
        'x_train'  : x_train,
        'x_test'   : x_test,
        'y_train'  : y_train,
        'y_test'   : y_test,
        'y_predict': y_predict,
    }

def lasso(x_train, x_test, y_train, y_test, random_state=1):
    model = Lasso(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'lasso',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

# See `Minimum Width for Universal Approximation` (says it's max(input_dim+1, output_dim))

def mlp_universal_approximation_theorem1(x_train, x_test, y_train, y_test, random_state=1):
    """
    Wide MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(10, ), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_uat_1',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def mlp_universal_approximation_theorem2(x_train, x_test, y_train, y_test, random_state=1):
    """
    Wide MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_uat_2',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }


def mlp_universal_approximation_theorem3(x_train, x_test, y_train, y_test, random_state=1):
    """
    Wide MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(1000, ), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_uat_3',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def mlp_deep1(x_train, x_test, y_train, y_test, random_state=1):
    """
    Deep MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_deep1',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def mlp_deep2(x_train, x_test, y_train, y_test, random_state=1):
    """
    Deep MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(10, 20, 10), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_deep2',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def mlp_deep3(x_train, x_test, y_train, y_test, random_state=1):
    """
    Deep MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(10, 20, 30, 20, 10), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_deep3',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def random_forest(x_train, x_test, y_train, y_test, random_state=1):
    """
    Random Forest Regressor
    """
    model = RandomForestRegressor(max_depth=3, random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'random_forest',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def bagging(x_train, x_test, y_train, y_test, random_state=1):
    """
    Bagging Regressor
    """
    model = BaggingRegressor(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'bagging',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def gradient_boosting(x_train, x_test, y_train, y_test, random_state=1):
    """
    Gradient Boosting Regressor
    """
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'gradient_boosting',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }


def adaboost(x_train, x_test, y_train, y_test, random_state=1):
    """
    AdaBoost Regressor
    """
    model = AdaBoostRegressor(n_estimators=100, random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'adaboost',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }


def voting(x_train, x_test, y_train, y_test, random_state=1):
    """
    Voting Regressor
    """
    model = VotingRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'voting',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }


# def stacking(x_train, x_test, y_train, y_test, random_state=1):
#     """
#     Stacking Regressor
#     """
#     model = StackingRegressor()
#     model.fit(x_train, y_train)
#     y_predict = model.predict(x_test)
#     y_test    = y_test.to_numpy()

#     return {
#         'model_name': 'stacking',
#         'x_train'   : x_train,
#         'x_test'    : x_test,
#         'y_train'   : y_train,
#         'y_test'    : y_test,
#         'y_predict' : y_predict,
#     }

def bayesian_regression1(x_train, x_test, y_train, y_test, random_state=1):
    model = make_pipeline(
        PolynomialFeatures(degree=10, include_bias=False),
        StandardScaler(),
        BayesianRidge(),
    )

    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'bayesian_regression',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def svm_regression(x_train, x_test, y_train, y_test, random_state=1):
    model = SVR()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'svm',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def svm_and_scaler_regression(x_train, x_test, y_train, y_test, random_state=1):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), SVR())
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'svm_w_s',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression(x_train, x_test, y_train, y_test, random_state=1):
    model = KNeighborsRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'knn',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_v2(x_train, x_test, y_train, y_test, random_state=1):
    model = KNeighborsRegressor(weights='distance')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'knn_v2',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_v3(x_train, x_test, y_train, y_test, random_state=1):
    model = KNeighborsRegressor(n_neighbors=3, weights='distance')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'knn_v3',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_v4(x_train, x_test, y_train, y_test, random_state=1):
    model = KNeighborsRegressor(n_neighbors=1, weights='distance')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'knn_v4',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_v5_adaptable(x_train, x_test, y_train, y_test, random_state=1):
    """
    The idea behind this function is to have a KNN model that adapts to the size of the training set
    Presumably, when you have very little training data, you want to use a small number of neighbors
    As the number of examples increase, a larger numbers of neighbors is fine.
    """
    if x_train.shape[0] < 3:
        n_neighbors=1
    elif x_train.shape[0] < 7:
        n_neighbors=3
    else:
        n_neighbors=5

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'knn_v5_adaptable',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_generic(x_train, x_test, y_train, y_test, model_name, knn_kwargs):
    # knn_args = {**knn_kwargs}
    # if 'n_neighbors' in knn_args:
    #     knn_args['n_neighbors'] = min(knn_args['n_neighbors'], len(x_train))
    model = KNeighborsRegressor(**knn_kwargs)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': model_name,
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_search():
    """
    A large number of KNN variants
    """
    idx = 0
    knn_fns = []
    for n_neighbors in [1, 2, 3, 5, 7, 9, 11]:
        for weights in ['uniform', 'distance']:
            for p in [0.25, 0.5, 1, 1.5, 2]:
                idx += 1
                knn_fns.append(
                    lambda x_train, x_test, y_train, y_test: knn_regression_generic(x_train, x_test, y_train, y_test, model_name=f'knn_search_{idx}', knn_kwargs={'n_neighbors': n_neighbors, 'weights': weights, 'p': p})
                )
    return knn_fns

def kernel_ridge_regression(x_train, x_test, y_train, y_test, random_state=1):
    model = KernelRidge()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'kernel_ridge',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def lr_with_polynomial_features_regression(x_train, x_test, y_train, y_test, random_state=1, **kwargs):
    degree = kwargs.get('degree', 2)
    
    # Create a pipeline that first transforms the data using PolynomialFeatures, then applies Linear Regression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'lr_with_polynomial_features',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def spline_regression(x_train, x_test, y_train, y_test, random_state=1, **kwargs):
    n_knots = kwargs.get('degree', 5) # Same defaults as SplineTransformer
    degree  = kwargs.get('degree', 3) # Same defaults as SplineTransformer
    
    # Create a pipeline that first transforms the data using PolynomialFeatures, then applies Linear Regression
    model = Pipeline([
        ('spline', SplineTransformer(n_knots=n_knots, degree=degree)),
        ('linear', LinearRegression())
    ])
    
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'spline',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }


def baseline(x_train, x_test, y_train, y_test, baseline_type, random_state=1, **kwargs):
    r = random.Random(random_state)
    if baseline_type == "average":
        pred   = np.mean(y_train)
        y_predict = np.array([pred for _ in y_test])
    elif baseline_type == "last":
        pred   = y_train.to_list()[-1]
        y_predict = np.array([pred for _ in y_test])
    elif baseline_type == "random":
        y_train_list = y_train.to_list()
        y_predict     = np.array([r.choice(y_train_list) for _ in y_test])
    elif baseline_type == 'constant_prediction':
        y_predict = np.array([kwargs['constant_prediction_value'] for _ in y_test])
    else:
        raise ValueError(f"Unknown {baseline_type}")

    y_test    = y_test.to_numpy()

    return {
        'model_name': baseline_type,
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }