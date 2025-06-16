"""Methods and classes for producing an array of sample points."""

import numpy as np
import tensorflow as tf

import typing

def get_sample_points(T: int,
                      number_sample_points: int
                     ) -> tuple[np.ndarray[np.float64], np.float64]:
    """Generates an array of `number_sample_points` equally spaced sample points
    between 0 and `T` and also returns the step size.

    Parameters
    ----------
    T : int
        The final point.
    number_sample_points : int
        The number of sample points between 0 and `T`.

    Returns
    -------
    tuple[NDArray[Shape[``number_sample_points``], np.float64], np.float64]
        A tuple containing the array of sample points and the step size.
    """
    t = np.linspace(0, T, number_sample_points, dtype=np.float64)
    return t, t[1]

class SampleTimes():
    """Allows for easy generation of equally spaced sample times from a variety
    of input data
    """

    _T: float
    """The total time"""
    
    _dt: float
    """The time step"""
    
    _number_sample_points: int
    """The number of sample times between 0 and :attr:`T`"""
    
    _t: np.ndarray[np.float64]
    """An array of the sample times"""
    
    _KEYWORDS = ["T", "dt", "number_sample_points"]
    """The keywords that are allowed to be passed to the constructor"""
    
    def __init__(self, **kwargs):
        """Initialises an instance of `SampleTimes`. Exactly two of the three
        optional keyword arguments must be passed.

        Parameters
        ----------
        T : float, optional
            The total time.
        dt : float, optional
            The time step.
        number_sample_points : int, optional
            The number of sample times between 0 and `T`.

        Raises
        ------
        TypeError
            Raised when more or less than two keyword arguments are provided.
        TypeError
            Raised when a keyword arguement other than `T`, `dt`, or
            `number_sample_points` is passed.
        """
        if len(kwargs) != 2:
            raise TypeError("Must be exactly two arguments.")
        for key, value in kwargs.items():
            if key not in self._KEYWORDS:
                raise TypeError(f"Only {self._KEYWORDS} are allowed arguments.")
            setattr(self, "_"+key, value)
        if "T" not in kwargs.keys():
            self._T = self._dt*(self._number_sample_points-1)
            self._t = np.linspace(0, self._T, self._number_sample_points)
        elif "dt" not in kwargs.keys():
            self._t, self._dt = get_sample_points(self._T, self._number_sample_points)
        else:
            assert np.floor(self._T/self._dt)*self._dt == self._T
            self._number_sample_points = 1 + int(self._T/self._dt)
            self._t = np.linspace(0, self._T, self._number_sample_points)
        self._t.flags.writeable = False
    @property
    def T(self) -> float:
        """
        The total time. If updated, so is :attr:`dt` while
        :attr:`number_sample_points` is kept constant.
        """
        return self._T
    @T.setter
    def T(self, value: float):
        self._T = value
        self._t, self._dt = get_sample_points(self._T, self._number_sample_points)
        self._t.flags.writeable = False
    @property
    def dt(self) -> float:
        """
        The time step. If updated, so is :attr:`T` while
        :attr:`number_sample_points` is kept constant.
        """
        return self._dt
    @dt.setter
    def dt(self, value: float):
        self._dt = value
        self._T = self._dt*(self._number_sample_points-1)
        self._t = np.linspace(0, self._T, self._number_sample_points)
        self._t.flags.writeable = False
    @property
    def number_sample_points(self) -> int:
        """
        The number of sample times between 0 and `T`. If updated, so is
        :attr:`dt` while :attr:`T` is kept constant."""
        return self._number_sample_points
    @number_sample_points.setter
    def number_sample_points(self, value: int):
        self._number_sample_points = value
        self._t, self._dt = get_sample_points(self._T, self._number_sample_points)
        self._t.flags.writeable = False
    @property
    def t(self) -> np.ndarray[np.float64]:
        """An array of the sample times"""
        return self._t

def sample_from_piecewise_linear(signal: np.ndarray[complex],
                                 fractional_indices: np.ndarray[float]
                                ) -> typing.Any:
    """
    Samples from a piecewise linear signal.

    Parameters
    ----------
    signal : NDArray[Shape[n_points, Any_Shape], complex]
        The piecewise linear signal to sample from
    fractional_indices : NDArray[Shape[n_sample_points], complex]
        Each entry will correspond to a sample. The integral part of each entry
        corresponds to the index ``i`` in the signal and the fractional part
        corresponds to the convex combination of the points at indices ``i`` and
        ``i+1``.

    Returns
    -------
    NDArray[Shape[n_sample_points, ``signal.shape[1:]``], complex]
        The sampled signal.
    """
    ceil_indices = tf.cast(tf.math.ceil(fractional_indices), dtype=tf.int32)
    floor_indices = tf.cast(tf.math.floor(fractional_indices), dtype=tf.int32)
    sample_ceil = tf.gather(signal, ceil_indices)
    sample_floor = tf.gather(signal, floor_indices)
    return sample_floor \
           + tf.einsum("i,i...->i...",
                       fractional_indices % 1,
                       sample_ceil - sample_floor)