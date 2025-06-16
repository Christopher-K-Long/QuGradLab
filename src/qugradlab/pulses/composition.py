"""A collection of methods for composing functions and tensors."""

import typing
from typing import Callable

import numpy as np
import tensorflow as tf

def pad(x: np.ndarray[complex],
        value: complex = 0,
        left: bool = True,
        right: bool = True
       ) -> typing.Any:
    """Adds padding to a tensor.

    Parameters
    ----------
    x: NDArray[Shape[s := Any_Shape]]
        The tensor to pad
    value: number
        The value to pad the tensor with, by default ``0``
    left: bool
        Whether to prepend padding, by default ``True``
    right: bool
        Whether to append padding, by default ``True``

    Returns
    -------
    NDArray[Shape[``s[0]+left+right``, ``s[1:]``]]
       Padded tensor
    """
    if not left and not right:
        return x
    padding = value*tf.ones_like(x[0:1])
    if left and right:
        return tf.concat([padding, x, padding], axis=0)
    elif left and not right:
        return tf.concat([padding, x], axis=0)
    return tf.concat([x, padding], axis=0)

def concatenate_functions(functions: list[Callable]) -> Callable:
    """Generates a function that executes the listed functions and concatenates
    the outputs.

    Parameters
    ----------
    functions : list[Callable]
        A list of Callables that return concatenatable outputs.

    Returns
    -------
    Callable
        A function that executes the listed functions and concatenates the
        outputs.

    Example
    -------
    >>> def f1(x):
    ...     return [x + 1]
    >>> def f2(x):
    ...     return [x - 1]
    >>> a = f1(1)
    >>> b = f2(2)
    >>> c = tf.concat([a, b], axis=-1)
    >>> f3 = concatenate_functions([f1, f2])
    >>> d = f3([[1], [2]])
    >>> assert all(c == d)
    """
    def execute_and_concatenate(function_arguments):
        return tf.concat([d(*a) for d, a in zip(functions,
                                                function_arguments)],
                         axis=-1)
    return execute_and_concatenate
