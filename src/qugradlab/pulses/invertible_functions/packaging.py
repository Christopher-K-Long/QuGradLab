"""
A collection of methods for packaging and unpackaging varaibles into a single
tensor.
"""

import math
import typing

from typing import Iterable

import numpy as np
import tensorflow as tf

from . import InvertibleFunction as _InvertibleFunction

@_InvertibleFunction
def package_complex(x: np.ndarray[float]) -> typing.Any:
    """Takes the last axis of a real tensor and takes the first entry as the
    real part and the second as the imaginary.

    Parameters
    ----------
    x: NDArray[Shape[s := Any_Shape, 2], float]
        The tensor to reshape.

    Returns
    -------
    NDArray[Shape[s], complex]
        The reshaped tensor.

    Methods
    -------
    package_complex.inverse
        Takes a complex tensor and splits it into the real and imaginary parts
        as the final axis.

        PARAMETERS:
            * **z** (*NDArray[Shape[s := Any_Shape], complex]*) —
              The tensor to reshape.

        RETURNS:
            The reshaped tensor.

        RETURN TYPE:
            NDArray[Shape[s, 2], float]
    package_complex.compose
        Composes :func:`package_complex()` with another
        :class:`.InvertibleFunction` to create a new
        :class:`.InvertibleFunction` along with the composed inverse. That is
        the following assertions should hold::

            assert package_complex.compose(g)(x, *g_args, **g_kwargs) \
                == package_complex(g(x, *g_args, **g_kwargs))

        for all inputs ``x``.

        PARAMETERS:
            * **inner_invertible_function** (*InvertibleFunction*) —
              The :class:`.InvertibleFunction` be be called first. The output of
              this :class:`.InvertibleFunction` will be passed to
              :func:`package_complex()`.

        RETURNS:
            A new :class:`.InvertibleFunction` that is the composition of the
            two functions.

        RETURN TYPE:
            :class:`.InvertibleFunction`
    """
    x = tf.cast(x, dtype=tf.complex128)
    return x[..., 0] + 1j*x[..., 1]
@package_complex.set_inverse
def unpackage_complex(z: np.ndarray[complex]) -> typing.Any:
    return tf.stack([tf.math.real(z), tf.math.imag(z)], axis=-1)
unpackage_complex = package_complex.inverse
"""Takes a complex tensor and splits it into the real and imaginary parts as the
final axis.

Parameters
----------
z: NDArray[Shape[s := Any_Shape], complex]
    The tensor to reshape.

Returns
-------
NDArray[Shape[s, 2], float]
    The reshaped tensor.

Methods
-------
unpackage_complex.inverse
    Takes the last axis of a real tensor and takes the first entry as the real
    part and the second as the imaginary.

    PARAMETERS:
        * **x** (*NDArray[Shape[s := Any_Shape, 2], float]*) —
          The tensor to reshape.

    RETURNS:
        The reshaped tensor.

    RETURN TYPE:
        NDArray[Shape[s], complex]
unpackage_complex.compose
    Composes the :func:`unpackage_complex()` with another
    :class:`.InvertibleFunction` to create a new :class:`.InvertibleFunction`
    along with the composed inverse. That is the following assertions should
    hold::

        assert package_complex.compose(g)(z, *g_args, **g_kwargs) \
            == package_complex(g(z, *g_args, **g_kwargs))

    for all inputs ``z``.

    PARAMETERS:
        * **inner_invertible_function** (*InvertibleFunction*) —
          The :class:`.InvertibleFunction` be be called first. The output of
          this :class:`.InvertibleFunction` will be passed to
          :func:`unpackage_complex()`.

    RETURNS:
        A new :class:`.InvertibleFunction` that is the composition of the two
        functions.

    RETURN TYPE:
        :class:`.InvertibleFunction`
"""

class ViewPop():
    """
    A class that allows for a view of an iterable to be popped.

    Example
    -------
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> v = ViewPop(x)
    >>> v(2)
    [1, 2]
    >>> v(2)
    [3, 4]
    >>> v(1)
    [5]
    >>> v(2)
    [6]
    """

    iterator: Iterable
    """
    The iterator to be viewed
    """

    index: int
    """
    The current index of the iterator
    """
    
    def __init__(self, iterator: Iterable):
        """
        Initialises the view of the `iterator`.

        Parameters
        ----------
        iterator : Iterable
            The iterator to be viewed
        """
        self.iterator = iterator
        self.index = 0
    def __call__(self, number_of_elements_to_pop: int = 1) -> Iterable:
        """
        Pops a view of the `iterator` of the given size.
        
        Parameters
        ----------
        number_of_elements_to_pop : int
            The number of elements to pop from the iterator
            
        Returns
        -------
        Iterable
            A view of the iterator of the given size
        """
        new_index = self.index+number_of_elements_to_pop
        view = self.iterator[self.index:new_index]
        self.index = new_index
        return view

@_InvertibleFunction
def unpack(x: Iterable, shapes: Iterable[Iterable[int]]) -> list:
    """
    Unpacks an input iterable `x` into
    `TensorFlow <https://www.tensorflow.org>`__ tensors of the given shapes.

    Parameters
    ----------
    x: NDArray[Shape[length], complex]
        The iterable to be unpacked.
    shapes: Iterable[Iterable[int]]
        The shapes of the tensors to be unpacked into.

    Returns
    -------
    list
        A list of `TensorFlow <https://www.tensorflow.org>`__
        tensors of the given shapes.

    Methods
    -------
    unpack.inverse
        Packs the list of input tensors into a single
        `TensorFlow <https://www.tensorflow.org>`__ tensor.

        PARAMETERS:
            * **tensors** (*list*) —
              The `TensorFlow <https://www.tensorflow.org>`__ tensors to be
              packed
            * **shapes** (*Iterable[Iterable[int]]*) —
              A placeholder parameter such that shapes can be pre-specified for
              :func:`unpack()`

        RETURNS:
            A `TensorFlow <https://www.tensorflow.org>`__ tensor with 1 axis
            consisting of all the input tensors flattened and concatenated.

        RETURN TYPE:
            Any
    unpack.specify_parameters
        Allows the ``shapes`` to be pre-specified. This removes shapes from the
        call signature.

        PARAMETERS:
            * **shapes** (*Iterable[Iterable[int]], optional*) —
              The shapes of the tensors to be unpacked into.

        RETURNS:
            A new :class:`.InvertibleFunction` with the specified parameters
            pre-specified.

        RETURN TYPE:
            InvertibleFunction   
    unpack.compose
        Composes the :func:`unpack` with another :class:`.InvertibleFunction`
        to create a new :class:`.InvertibleFunction` along with the composed
        inverse. That is the following assertions should hold::

            assert unpack.compose(g, *args, **kwargs)(x, *g_args, **g_kwargs) \
                == unpack(g(x, *g_args, **g_kwargs), *args, **kwargs)

        for all inputs ``x``.

        PARAMETERS:
            * **inner_invertible_function** (*InvertibleFunction*) —
              The :class:`.InvertibleFunction` be be called first. The output of
              this :class:`.InvertibleFunction` will be passed to
              :func:`unpack()`.
            * **shapes** (*Iterable[Iterable[int]]*) —
              The shapes of the tensors to be unpacked into.

        RETURNS:
            A new :class:`.InvertibleFunction` that is the composition of the
            two functions.

        RETURN TYPE:
            InvertibleFunction
    """
    v = ViewPop(x)
    return [tf.reshape(v(math.prod(s)), s) for s in shapes]
@unpack.set_inverse
def pack(tensors: list, shapes: Iterable[Iterable[int]] = []) -> typing.Any:
    return tf.concat([tf.reshape(a, (-1,)) for a in tensors], axis=0)
pack = unpack.inverse
"""
Packs the list of input tensors into a single
`TensorFlow <https://www.tensorflow.org>`__ tensor.

Parameters
----------
tensors: list
    The list of `TensorFlow <https://www.tensorflow.org>`__ tensors to be packed
shapes: Iterable[Iterable[int]]
    A placeholder parameter such that shapes can be pre-specified for
    :func:`unpack()`

Returns
-------
`TensorFlow <https://www.tensorflow.org>`__ Tensor
    A `TensorFlow <https://www.tensorflow.org>`__ tensor with 1 axis consisting
    of all the input tensors flattened and concatenated.

Methods
-------
unpack.inverse
    Unpacks an input iterable `x` into
    `TensorFlow <https://www.tensorflow.org>`__ tensors of the given shapes.

    PARAMETERS:
        * **x** (*Iterable*) —
          The iterable to be unpacked.
        * **shapes** (*Iterable[Iterable[int]]*) —
          The shapes of the tensors to be unpacked into.

    RETURNS:
        A `TensorFlow <https://www.tensorflow.org>`__ tensor with 1 axis
        consisting of all the input tensors flattened and concatenated.

    RETURN TYPE:
        list
unpack.specify_parameters
    Allows the ``shapes`` to be pre-specified. This removes shapes from the
    call signature of :func:`inverse()`.

    PARAMETERS:
        * **shapes** (*Iterable[Iterable[int]], optional*) —
          The shapes of the tensors to be unpacked into.

    RETURNS:
        A new :class:`.InvertibleFunction` with the specified parameters
        pre-specified.

    RETURN TYPE:
        InvertibleFunction   
unpack.compose
    Composes the :func:`pack()` with another :class:`.InvertibleFunction`
    to create a new :class:`.InvertibleFunction` along with the composed
    inverse. That is the following assertions should hold::

        assert pack.compose(g, *args, **kwargs)(x, *g_args, **g_kwargs) \
            == pack(g(x, *g_args, **g_kwargs), *args, **kwargs)

    for all inputs ``x``.

    PARAMETERS:
        * **inner_invertible_function** (*InvertibleFunction*) —
          The :class:`.InvertibleFunction` be be called first. The output of
          this :class:`.InvertibleFunction` will be passed to :func:`pack`.
        * **shapes** (*Iterable[Iterable[int]]*) —
          The shapes of the tensors to be unpacked into.

    RETURNS:
        A new :class:`.InvertibleFunction` that is the composition of the two
        functions.

    RETURN TYPE:
        InvertibleFunction
"""