from typing import Union

import numpy as np

def get_digit(number: Union[int, np.ndarray[int]],
              base: Union[int, np.ndarray[int]],
              digit: Union[int, np.ndarray[int]]
             ) -> Union[int, np.ndarray[int]]:
    """
    Computes the specified `digit` of a `number` when the `number` is expressed
    in a specified `base`.

    Parameters
    ----------
    number: int | np.ndarray[int]
        The number(s) to find the digit(s) of
    base: int | np.ndarray[int]
        The base(s) to use
    digit: int | np.ndarray[int]
        The digit(s) to find

    Returns
    -------
    int | np.ndarray[int]
        The digit(s) fo the number(s) in the base(s)

    Note
    ----
    If arrays are passed instead of integers the output is an "outer operation"
    in which the first set of axes correspond to the ``number``, the second set
    of axes correspond to ``base`` and the third set of axes correspond to
    ``digit``. That is the shape of the output will be the concatenation of
    shapes of ``number``, ``base``, and ``digit``.
    """
    number = np.array(number)
    base = np.array(base)
    digit = np.array(digit)
    return np.floor_divide.outer(number,np.power.outer(base, digit)) \
           % base.reshape((1,)*number.ndim + base.shape + (1,)*digit.ndim)
