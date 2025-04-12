from typing import Union
import numpy as np


def calc_t_test_statistic(tau: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """
    Calculate the test statistic t from the tau value
    """
    return -2 * tau
