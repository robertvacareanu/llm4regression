import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def scores(y_predict, y_test, model_name, **kwargs):

    lr_l1 = np.linalg.norm(y_test - y_predict, ord=1).item()
    
    return {
        'method': model_name,
        'preds': y_predict.tolist(),
        'gold': y_test.tolist(),
        'l1': np.abs(y_test - y_predict).item(),
        'r2': r2_score(y_test, y_predict),
    }
