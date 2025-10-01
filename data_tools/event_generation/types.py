from typing import TypeVar, Union

import numpy as np


FLOAT_OR_ARRAY = TypeVar('FLOAT_OR_ARRAY', bound=Union[float, np.ndarray])
