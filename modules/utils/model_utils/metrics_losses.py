import numpy as np

import tensorflow.keras.backend as K


def mae_np(y_true, y_pred, axis=None):
    """Function for computing the Mean Absolute Error (MAE)
    given numpy array:

    Args:
        - y_true: a numpy array, is the collection of ground truth values
        - y_pred: a numpy array, is the collection of predicted values

    Returns:
        - mean_absolute_eror: is a float, the MAE between y_true and y_pred
    """
    absolute_error = np.abs(y_true - y_pred)
    mean_absolute_eror = np.nanmean(absolute_error, axis=axis)
    return mean_absolute_eror


def smape_np(y_true, y_pred, axis=None):
    """Function for computing the Simmetric Mean Absolute Error (SMAPE)
    given numpy array:

    Args:
        - y_true: a numpy array, is the collection of ground truth values
        - y_pred: a numpy array, is the collection of predicted values

    Returns:
        - division: is a float, the SMAPE between y_true and y_pred
    """
    nominator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) + 1e-07
    division = np.nanmean(nominator / denominator, axis=axis)
    return division


def smape_k(y_true, y_pred):
    """Function for computing the Simmetric Mean Absolute Error (SMAPE)
    given keras tensors:

    Args:
        - y_true: a keras tensor, is the collection of ground truth values
        - y_pred: a keras tensor, is the collection of predicted values

    Returns:
        - division: is a float, the SMAPE between y_true and y_pred
    """
    y_true = K.cast(y_true, 'float32')
    nominator = K.abs(y_true - y_pred)
    denominator = (K.abs(y_true) + K.abs(y_pred) + K.epsilon())
    division = K.mean(nominator / denominator, axis=-1)
    return division
