"""A collection of reversible rescalings for contructing pulse sequences."""

import numpy as np

from . import InvertibleFunction as _InvertibleFunction

@_InvertibleFunction
def linear_rescaling(x: np.ndarray[complex],
                     min: complex,
                     max: complex
                    ) -> np.ndarray[complex]:
    """
    Rescales an input `x` in the range [-1, 1] to the range [`min`, `max`] with
    -1 being mapped to `min` and 1 being mapped to `max`.

    Parameters
    ----------
    x: NDArray[Shape[s := Any_Shape], complex]
        The input to be rescaled
    min: complex
        The minimum value of the rescaled range
    max: complex
        The maximum value of the rescaled range

    Returns
    -------
    NDArray[Shape[s], complex]
        The rescaled input

    Note
    ----
    :func:`linear_rescaling()` is an instance of :class:`.InvertibleFunction`.

    Methods
    -------
    linear_rescaling.inverse
        The inverse linear rescaling

        PARAMETERS:
            * **x** (*NDArray[Shape[s := Any_Shape], complex]*) —
              The rescaled value to be unscaled
            * **min** (*complex*) —
              The minimum value of the rescaled range
            * **max** (*complex*) —
              The maximum value of the rescaled range

        RETURNS:
            The unscaled value

        RETURN TYPE:
            NDArray[Shape[s], complex]
    linear_rescaling.specify_parameters
        Allows the maximum and minimum values to be pre-specified. This removes
        them from the call signature. If only one of ``max`` or ``min`` is
        passed, then only the value speficied is removed from the call
        signature.

        PARAMETERS:
            * **min** (*complex, optional*) —
              The minimum value of the rescaled range
            * **max** (*complex, optional*) —
              The maximum value of the rescaled range

        RETURNS:
            A new :class:`.InvertibleFunction` with the specified parameters
            pre-specified.
            
        RETURN TYPE:
            InvertibleFunction
                
    linear_rescaling.compose
        Composes the :func:`linear_rescaling()` with another
        :class:`.InvertibleFunction` to create a new
        :class:`.InvertibleFunction` along with the composed inverse.
        That is the following assertions should hold::

            assert linear_rescaling.compose(g, *args, **kwargs)(x, *g_args, **g_kwargs) \
                == linear_rescaling(g(x, *g_args, **g_kwargs), *args, **kwargs)

        for all inputs ``x``.

        PARAMETERS:
            * **inner_invertible_function** (*InvertibleFunction*) —
              The :class:`.InvertibleFunction` be be called first. The output of
              this :class:`.InvertibleFunction` will be passed to
              :func:`linear_rescaling()`.
            * **min** (*complex*) —
              The minimum value of the rescaled range
            * **max** (*complex*) —
              The maximum value of the rescaled range

        RETURNS:
            A new :class:`.InvertibleFunction` that is the composition of the
            two functions.

        RETURN TYPE:
            InvertibleFunction
    """
    return 0.5*(min + max + (max-min)*x)
@linear_rescaling.set_inverse
def linear_rescaling_inverse(y: np.ndarray[complex],
                             min: complex,
                             max: complex
                     ) -> np.ndarray[complex]:
    return(2*y - min - max)/(max - min)
