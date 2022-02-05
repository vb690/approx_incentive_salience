import numpy as np


def compute_pooled_sd(var_1, var_2, n_1, n_2):
    """Compute pooled standard deviation

    Para:
        - var_1: float, variance group 1
        - var_2: float, variance group 2
        - n_1: int, size_group 2
        - n_2: int, size group 1

    Returns:
        - pooled_sd: float, pooled standard deviation
    """
    group_1 = var_1 * (n_1 - 1)
    group_2 = var_2 * (n_2 - 1)
    pooled_var = (group_1 + group_2) / (n_1 + n_2 - 2)
    pooled_sd = np.sqrt(pooled_var)
    return pooled_sd


def compute_cd_from_regression(beta, var_1, var_2, n_1, n_2):
    """Compute cohen d from unstandardized regression coefficient

    Para:
        - beta: float, unstandardized regression coefficient
        - var_1: float, variance group 1
        - var_2: float, variance group 2
        - n_1: int, size_group 2
        - n_2: int, size group 1

    Returns:
        - cd: float, cohen's d
    """
    pooled_sd = compute_pooled_sd(var_1, var_2, n_1, n_2)
    cd = beta / pooled_sd
    return cd
