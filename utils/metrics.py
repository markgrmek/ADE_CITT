import numpy as np
from sklearn.metrics import mean_squared_error

def MSE(
    measured: np.ndarray,
    predicted: np.ndarray
    ) -> float:

    return mean_squared_error(measured, predicted)

def RMSE(
    measured: np.ndarray,
    predicted: np.ndarray
    ) -> float:

    return np.sqrt(MSE(measured, predicted))