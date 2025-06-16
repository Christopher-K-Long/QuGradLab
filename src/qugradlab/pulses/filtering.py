"""A collection of methods for filtering contorl amplitudes"""

import typing
import functools

from typing import Callable

import numpy as np
import tensorflow as tf

import scipy.special

from . import composition

def get_fixed_filter(n_pieces_in_spline: int,
                     piece_length: float,
                     samples_per_piece: int,
                     low_pass_time_constant: float,
                     high_pass_time_constant: float,
                     low_pass_order: float,
                     spline_order: int,
                     high_pass_order: float
                    ) -> Callable[[np.ndarray[np.complex128]], typing.Any]:
    r"""
    Generates a Callable that applies the defined filter to the input signal
    represented by a spline. The filter has a transfer function of the form
    $$
    T(\omega)=\frac{(i\omega\tau')^m}{(1+i\omega\tau)^n}.
    $$
    where $\tau'$ corresponds to `high_pass_time_constant`, $m$ corresponds to
    `high_pass_order`, $\tau$ corresponds to `low_pass_time_constant`, and $n$
    corresponds to `low_pass_order`.

    Consider the spline

    $$
    \begin{aligned}
    f\left(t\right)&\coloneqq\sum_{i=0}^L\sqcap_i\left(t\right)\sum_{j=0}^N
    f^-_{ij}\left(t-t_i\right)^j\quad\textrm{where }\sqcap_i\left(t\right)
    \coloneqq\begin{cases}
    1&t_i\le t< t_{i+1},\\
    0&\textrm{otherwise},
    \end{cases}\\
    &\equiv\sum_{i=0}^L\sqcap_i\left(t\right)\sum_{j=0}^N
    f^+_{ij}\left(t-t_{i+1}\right)^j\quad
    \textrm{where }f^\pm_{ij}\equiv\sum_{k=j}^N
    {k\choose j}\left(\pm[t_{i+1}-t_i]\right)^{j-k}f^\mp_{ik}
    \end{aligned}
    $$

    where $L$ correspodns to `n_pieces_in_spline`, $N$ corresponds to
    `spline_order`, and $t_{i+1}-t_i$ corresponds to `piece_length`.

    If we apply the transfer function $T(\omega)$ to the spline $f(t)$ we find
    the filtered spline is given by

    $$
    f'\left(t\right)=\tau^{-n}\tau'^{m}\sum_{i=0}^L\sum_{j=0}^N\Delta f_{ij}
    \Theta\left(t-t_i\right)\frac{\Gamma\left(j+1\right)
    \left(t-t_i\right)^{n-m+j}}{\Gamma\left(n-m+j+1\right)}
    {_1F_1}\left[\begin{matrix}
    n\\
    n-m+j+1
    \end{matrix}\:; -\left(t-t_i\right)\tau^{-1}\right]
    $$
    where $\Delta f_{ij}\coloneqq f^-_{ij}-f^+_{\left(i-1\right)j}$, $\Theta$ is
    the Heaviside step function, $\Gamma$ is the Gamma function, and ${_1F_1}$
    is the confluent hypergeometric function.

    Parameters
    ----------
    n_pieces_in_spline : int
        The number of pieces in the spline
    piece_length : float
        The length of each piece in the spline
    samples_per_piece : int
        The number of samples per piece in the spline
    low_pass_time_constant : float
        The time constant of the low pass type filter. To apply a pure low
        pass filter set ``high_pass_order_constant = 0``.
    high_pass_time_constant : float
        The time constant of the high pass type filter. To apply a pure high
        pass filter `high_pass_time_constant` should equal
        ``low_pass_time_constant``.
    low_pass_order : float
        The order of the low pass type filter. To apply a pure low pass
        filter set ``high_pass_order = 0``.
    spline_orders : NDArray[Shape[n_orders], int]
        The orders of the spline. 0 corresponds to piecewise constant, 1
        corresponds to piecewise linear, etc.
    high_pass_order : float
        The order of the high pass type filter. To apply a pure high pass
        filter `high_pass_time_constant` should equal ``low_pass_time_constant``
        and ``high_pass_order`` should equal ``low_pass_order``.

    Returns
    -------
    Callable[[NDArray[np.complex128]], Any]
        A callable that takes a spline and returns the filtered spline.

        PARAMETERS:
            * **time_spline_matrix** (*NDArray[np.complex128]*) —
              This matrix defines the spline and corresponds to $\Delta f_{ij}$.

        RETURNS:
            The filtered spline $f'(t)$ evaluated at ``samples_per_piece``
            equally spaced points per piece of the splice. The shape will be
            ``(..., samples_per_piece * n_peices_in_spline)``.

        RETURN TYPE:
            `TensorFlow <https://www.tensorflow.org>`__ Tensor
    """
    piece_sample_times = np.linspace(0, piece_length, samples_per_piece+1)[:-1]
    piece_points = piece_length*np.arange(n_pieces_in_spline)
    sample_times = np.add.outer(piece_sample_times, piece_points)
    time_response = get_time_response(sample_times,
                                      low_pass_time_constant,
                                      high_pass_time_constant,
                                      low_pass_order,
                                      spline_order,
                                      high_pass_order)
    frequency_response = get_frequency_response(time_response)
    return functools.partial(apply_filtering_transform,
                             frequency_response=frequency_response)

def apply_filtering_transform(time_spline_matrix: np.ndarray[np.complex128],
                              frequency_response: np.ndarray[np.complex128]
                             ) -> np.ndarray[np.complex128]:
    r"""
    Filters a spline
    
    $$
    \begin{aligned}
    f\left(t\right)&\coloneqq\sum_{i=0}^L\sqcap_i\left(t\right)\sum_{j=0}^N
    f^-_{ij}\left(t-t_i\right)^j\quad\textrm{where }\sqcap_i\left(t\right)
    \coloneqq\begin{cases}
    1&t_i\le t< t_{i+1},\\
    0&\textrm{otherwise},
    \end{cases}\\
    &\equiv\sum_{i=0}^L\sqcap_i\left(t\right)\sum_{j=0}^N
    f^+_{ij}\left(t-t_{i+1}\right)^j\quad
    \textrm{where }f^\pm_{ij}\equiv\sum_{k=j}^N
    {k\choose j}\left(\pm[t_{i+1}-t_i]\right)^{j-k}f^\mp_{ik}
    \end{aligned}
    $$

    using the `frequency_response` to the transfer function
    
    $$
    T(\omega)=\frac{(i\omega\tau')^m}{(1+i\omega\tau)^n}.
    $$

    The filtered spline will be
    $$
    f'\left(t\right)=\tau^{-n}\tau'^{m}\sum_{i=0}^L\sum_{j=0}^N\Delta f_{ij}
    \Theta\left(t-t_i\right)\frac{\Gamma\left(j+1\right)
    \left(t-t_i\right)^{n-m+j}}{\Gamma\left(n-m+j+1\right)}
    {_1F_1}\left[\begin{matrix}
    n\\
    n-m+j+1
    \end{matrix}\:; -\left(t-t_i\right)\tau^{-1}\right]
    $$
    where $\Delta f_{ij}\coloneqq f^-_{ij}-f^+_{\left(i-1\right)j}$ corresponds
    to `time_spline_matrix`, $\Theta$ is
    the Heaviside step function, $\Gamma$ is the Gamma function, and ${_1F_1}$
    is the confluent hypergeometric function.

    Parameters
    ----------
    time_spline_matrix : NDArray[Shape[n_pieces_in_spline, n_orders, ...], np.complex128]
        This matrix defines the spline and corresponds to $\Delta f_{ij}$.
    frequency_response : NDArray[Shape[n_orders, samples_per_piece, ``2*n_pieces_in_spline``], np.complex128]
        The frequency response to the transfer function. This should be computed
        using :func:`get_frequency_response`.

    Returns
    -------
    `TensorFlow <https://www.tensorflow.org>`__ Tensor
        The filtered spline $f'(t)$ evaluated at ``samples_per_piece``
        equally spaced points per piece of the splice. The shape will be
        ``(..., samples_per_piece * n_peices_in_spline)``.
    """
    M = tf.shape(time_spline_matrix)[0]
    remaining_shape = tf.shape(time_spline_matrix)[2:]
    time_spline_matrix = tf.cast(time_spline_matrix, tf.complex128)
    time_spline_matrix = tf.transpose(time_spline_matrix)
    padding = tf.zeros_like(time_spline_matrix)
    time_spline_matrix = tf.concat([padding, time_spline_matrix], axis=-1)
    time_spline_matrix = tf.signal.fft(time_spline_matrix)
    return tf.reshape(tf.transpose(tf.signal.ifft(tf.einsum("...jt,jdt->...dt", time_spline_matrix, frequency_response))[..., M:]), tf.concat([[-1], remaining_shape], axis=0))

def get_time_response(sample_times: np.ndarray[np.float64],
                      low_pass_time_constant: float,
                      high_pass_time_constant: float,
                      low_pass_order: int,
                      mononomial_order: int,
                      high_pass_order: int
                     ) -> np.ndarray[np.float64]:
    r"""Generates the response of the transfer function
    
    $$
    T(\omega)=
    \frac{(i\omega\tau')^m}{(1+i\omega\tau)^n}.
    $$
    
    to a Heaviside step function multiplied by a mononomial of order
    `mononomial_order`
    
    $$
    g\left(t\right)=\tau^{-n}\tau'^{m}\Theta\left(t\right)
    \frac{\Gamma\left(j+1\right)t^{n-m+j}}{\Gamma\left(n-m+j+1\right)}
    {_1F_1}\left[\begin{matrix}
    n\\
    n-m+j+1
    \end{matrix}\:; -t\tau^{-1}\right].
    $$
    
    where $\tau$ corresponds to `low_pass_time_constant`, $\tau'$ corresponds to
    `high_pass_time_constant`, $m$ corresponds to `high_pass_order`, $n$
    corresponds to `low_pass_order`, $j$ corresponds to
    `mononomial_order`, $\Theta$ is
    the Heaviside step function, $\Gamma$ is the Gamma function, and ${_1F_1}$
    is the confluent hypergeometric function.

    Parameters
    ----------
    sample_times : NDArray[Shape[n_samples], np.float64]
        The times at which the response function is evaluated.
    low_pass_time_constant : float
        The time constant of the low pass type filter. To apply a pure low
        pass filter set ``high_pass_order_constant = 0``.
    high_pass_time_constant : float
        The time constant of the high pass type filter. To apply a pure high
        pass filter `high_pass_time_constant` should equal
        ``low_pass_time_constant``.
    low_pass_order : int
        The order of the low pass type filter. To apply a pure low pass
        filter set ``high_pass_order = 0``.
    mononomial_order : NDArray[Shape[n_orders], int]
        The order of the mononomial. 0 corresponds constant, 1 linear, etc.
    high_pass_order : int
        The order of the high pass type filter. To apply a pure high pass
        filter ``high_pass_time_constant`` should equal
        ``low_pass_time_constant`` and ``high_pass_order`` should equal
        ``low_pass_order``.

    Returns
    -------
    NDArray[Shape[n_orders, n_samples], np.float64]
        The response function evaluated at ``sample_times``.
    """
    sample_times = np.array(sample_times)
    tndim = sample_times.ndim
    mononomial_order = np.array(mononomial_order)
    jndim = mononomial_order.ndim
    for _ in range(jndim): sample_times = np.expand_dims(sample_times, axis=0)
    for _ in range(tndim):
        mononomial_order = np.expand_dims(mononomial_order, axis=-1)
    order = (low_pass_order - high_pass_order) + mononomial_order
    heaviside_step = (sample_times >= 0)
    polynomial = np.power(sample_times, order)
    pre_factor = (scipy.special.factorial(mononomial_order)
                  /scipy.special.factorial(order)) \
                * np.power(high_pass_time_constant, high_pass_order) \
                / np.power(low_pass_time_constant, low_pass_order)
    decay = scipy.special.hyp1f1(low_pass_order,
                                 order+1,
                                 -sample_times/low_pass_time_constant)
    return pre_factor*heaviside_step*polynomial*decay

def get_frequency_response(time_response: np.ndarray) -> typing.Any:
    r"""Adds zeros as padding to a time signal before performing descrete
    Fourier transform. The zero padding allows the discrete convolution theorem
    to be used.

    Parameters
    ----------
    time_response : NDArray[Shape[..., n_samples], np.float64]
        The time response function to be transformed.

    Returns
    -------
    NDArray[Shape[..., ``2*n_samples``], np.complex128]
        The frequency response function.
    """
    time_response = tf.cast(time_response, tf.complex128)
    time_response = tf.concat([time_response,
                               tf.zeros_like(time_response)],
                              axis=-1)
    return tf.signal.fft(time_response)

def const_spline_time_spline_matrix(points: np.ndarray) -> typing.Any:
    """
    Generates a spline matrix of a piecewise constant spline from an array
    of `points` that represent each of the value of each of the constant pieces
    of the piecewise constant spline.

    Parameters
    ----------
    points : NDArray[Shape[n_pieces, ...], np.float64]
        The values of the piecewise constant spline.
        
    Returns
    -------
    NDArray[Shape[n_pieces, n_orders, ...], np.float64]
        The spline matrix of the piecewise constant spline.
    """
    return tf.expand_dims(points-composition.pad(points, right=False)[:-1],
                          axis=1)
