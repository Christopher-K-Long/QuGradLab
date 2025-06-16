"""
A class for representing invertible functions.
"""

from functools import partial
from typing import Callable, Optional

from qugrad.pulses import compose

class InvertibleFunction():
    """A class representing an invertible function."""
    
    _inverse: Optional["InvertibleFunction"] = None
    """
    The inverse of the function

    Parameters
    ----------
    input
        The input to the inverse function
    *args
        Any additional positional arguments to pass to the inverse function.
    **kwargs
        Keyword arguments to pass to the inverse function.

    Returns
    -------
    Any
        The result of the inverse function call.

    Note
    ----
    `_inverse` should satisfy the following assertions::

        assert self._inverse(self(x, *args, **kwargs), *args, **kwargs) == x
        assert self(self._inverse(x, *args, **kwargs), *args, **kwargs) == x

    for all inputs ``x``.
    """
    
    def __init__(self, func: Callable):
        """
        Wraps the a function in an InvertibleFunction object so that an inverse
        can be associated with it.

        Parameters
        ----------
        func : Callable
            The function to be wrapped
        """
        self._func = func
    def __call__(self, input, *args, **kwargs):
        """
        Call the function with the given arguments.

        Parameters
        ----------
        input
            The input to the function
        *args
            Any additional positional arguments to pass to the function.
        **kwargs
            Keyword arguments to pass to the function.

        Returns
        -------
        Any
            The result of the function call.
        """
        return self._func(input, *args, **kwargs)
    @property
    def inverse(self):
        """
        The inverse of the function

        Parameters
        ----------
        input
            The input to the inverse function
        *args
            Any additional positional arguments to pass to the inverse function.
        **kwargs
            Keyword arguments to pass to the inverse function.

        Returns
        -------
        Any
            The result of the inverse function call.

        Note
        ----
        `inverse` should satisfy the following assertions::

            assert self.inverse(self(x, *args, **kwargs), *args, **kwargs) == x
            assert self(self.inverse(x, *args, **kwargs), *args, **kwargs) == x

        for all inputs ``x``.
        """
        if self._inverse is not None:
            return self._inverse
        raise NotImplementedError("This function has no inverse.")
    def set_inverse(self, inverse_func: Callable):
        """
        Sets the inverse of the function.

        Parameters
        ----------
        inverse_func : Callable
            The inverse function

        Note
        ----
        ``inverse_func`` should satisfy the following assertions::

            assert inverse_func(self(x, *args, **kwargs), *args, **kwargs) == x
            assert self(inverse_func(x, *args, **kwargs), *args, **kwargs) == x

        for all inputs ``x``.
        """
        self._inverse = InvertibleFunction(inverse_func)
        self.inverse._inverse = self
    def specify_parameters(self, **kwargs) -> "InvertibleFunction":
        """
        Allows keyword parameters for the function and inverse function to be
        pre-specified. This generates a new :class:`InvertibleFunction` without
        the specified parameters in the call signatures.

        Parameters
        ----------
        **kwargs
            Keyword arguments to pre-specify for the function and inverse
            function.

        Returns
        -------
        InvertibleFunction
            A new :class:`InvertibleFunction` with the specified parameters
            pre-specified.
        """
        func = partial(self.__call__, **kwargs)
        inverse = partial(self.inverse.__call__, **kwargs)
        new_invertible_function = InvertibleFunction(func)
        new_invertible_function.set_inverse(inverse)
        return new_invertible_function
    def compose(self,
                inner_invertible_function: "InvertibleFunction",
                *args,
                **kwargs
               ) -> "InvertibleFunction":
        """
        Composes the :class:`InvertibleFunction` with another
        :class:`InvertibleFunction` to create a new :class:`InvertibleFunction`
        along with the composed inverse. That is the following assertions should
        hold::

            assert f.compose(g, *f_args, **f_kwargs)(x, *g_args, **g_kwargs) \
                == f(g(x, *g_args, **g_kwargs), *f_args, **f_kwargs)

        for all inputs ``x``.
        
        Parameters
        ----------
        inner_invertible_function : InvertibleFunction
            The :class:`InvertibleFunction` be be called first. The output of
            this :class:`InvertibleFunction` will be passed to the current
            :class:`InvertibleFunction`.
        *args
            Any additional positional arguments to pass to the function.

        **kwargs
            Keyword arguments to pass to the function.

        Note
        ----
        Enough positional arguments and keyword arguments need to be passed that
        the calling :class:`InvertibleFunction` has only one remaining parameter
        (``input``).

        Returns
        -------
        InvertibleFunction
            A new :class:`InvertibleFunction` that is the composition of the two
            functions.
        """
        new_invertible_function = InvertibleFunction(compose(self.__call__,
                                          inner_invertible_function.__call__,
                                          *args,
                                          **kwargs))
        if (inner_invertible_function._inverse is not None
        and self._inverse is not None):
            inverse = compose(inner_invertible_function.inverse.__call__,
                            self.inverse.__call__,
                            *args,
                            **kwargs)
            new_invertible_function.set_inverse(inverse)
        return new_invertible_function