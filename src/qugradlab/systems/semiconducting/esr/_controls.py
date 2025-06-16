"""The control scheme for electron spin resonance (ESR) devices"""

from typing import Callable

import numpy as np
import tensorflow as tf

from qugrad import QuantumSystem

from ._device import Device
from ....pulses.composition import concatenate_functions
from ....pulses.invertible_functions.scaling import linear_rescaling

class Controls(Device):
    """A class representing an ESR device with its control scheme.

    Important
    ---------
    This class should not be used directly but instead inherited along with
    :class:`qugrad.QuantumSystem` or any of its child classes.

    Warning
    -------
    This class does not specify the Hamiltonian model as the control scheme will
    be the same for all ESR device Hamiltonian models. The class is seperated
    out from the parameters in :class:`Device` so that a minimal stub file can
    be utilised to document the new parameters for
    :class:`qugrad.QuantumSystem` which will be inherited by child classes.

    See Also
    --------
    * :class:`qugradlab.physical_systems.semiconducting.esr._systems.SpinSystem`
    * :class:`qugradlab.physical_systems.semiconducting.esr._systems.SpinAngledDriveSystem`
    * :class:`qugradlab.physical_systems.semiconducting.esr._systems.ValleySystem`
    """
    
    _rescale_and_concatenate: Callable[[np.ndarray[complex]], np.ndarray[complex]]
    """A function that rescales the Rabi drives and exchange couplings before
    concatenating them into a single array.

    Parameters
    ----------
    drive_ctrl_amp : NDArray[Shape[n_time_steps, n_drive_ctrl], complex]
        The control amplitudes for the global Rabi-drive Hamiltonian. These
        values should be in the range [``-1``, ``1``] and will be linearly
        rescaled to the range
        [− :attr:`max_drive_strength`, :attr:`max_drive_strength`] where
        ``-1`` corresponds to the − :attr:`max_drive_strength` and ``1`` to the
        :attr:`max_drive_strength`.
    J_ctrl_amp : NDArray[Shape[n_time_steps, n_J_ctrl], complex]
        The control amplitudes for the exchange Hamiltonians. These values
        should be in the range [``-1``, ``1``] and will be linearly rescaled to
        the range [:attr:`J_min`, :attr:`J_max`] where ``-1`` corresponds to the
        :attr:`J_min` and ``1`` to the :attr:`J_max`.

    Returns
    -------
    NDArray[Shape[n_time_steps, n_drive_ctrl + n_J_ctrl], complex]
        The rescaled and concatenated control amplitudes in the form expected by
        :meth:`qugrad.QuantumSystem._pre_processing()`.
    """
    
    def __init__(self,
                 zeeman_splittings: np.ndarray[complex],
                 max_drive_strength: float = 1,
                 J_min: float = 1,
                 J_max: float = 1
                ):
        """Initialises the control scheme for the ESR device.

        Parameters
        ----------
        zeeman_splittings : NDArray[Shape[spins], number]
            The Zeeman splittings of the spins
        max_drive_strength : float
            The maximum drive strength that can be applied at a specific
            frequency and quadrature. That is if their are ``n_drive_ctrl``
            frequencies and both quadratures are used then the maximum amplitude
            of the drive that can be applied to the device is::

                np.sqrt(2) * n_drive_ctrl * max_drive_strength
        J_min : float
            The minimum value of the exchange coupling $J$
        J_max : float
            The maximum value of the exchange coupling $J$
        """
        rescale_rabi_drive = \
            linear_rescaling.specify_parameters(min=-max_drive_strength,
                                                max=max_drive_strength)
        rescale_J = linear_rescaling.specify_parameters(min=J_min, max=J_max)
        self._rescale_and_concatenate = \
            concatenate_functions([rescale_rabi_drive, rescale_J])
        super().__init__(zeeman_splittings, max_drive_strength, J_min, J_max)
    def _pre_processing(self,
                        drive_ctrl_amp: np.ndarray[complex],
                        drive_frequencies: np.ndarray[complex],
                        J_ctrl_amp: np.ndarray[complex],
                        initial_state: np.ndarray[complex],
                        dt: float
                       ) -> tuple:
        """
        When calling any evolution method (listed in the
        :ref:`See also section <pre_processing_see_also>`)
        :meth:`_pre_processing()` is executed on the arguments before the
        control amplitudes are modulated by the frequencies (during
        :meth:`_envolope_processing()`) and then finally the modulated control
        amplitudes are used by the evolution method.

        :meth:`_pre_processing()` can be overridden to produce desired pulse
        shapes. You can either override :meth:`_pre_processing()` directly by
        creating a child class, or you can use :meth:`pulse_form()`.

        For :meth:`gradient()` to function correctly :meth:`_pre_processing()`
        should be written in `TensorFlow <https://www.tensorflow.org>`__.

        Parameters
        ----------
        drive_ctrl_amp : NDArray[Shape[n_time_steps, n_drive_ctrl], complex]
            The control amplitudes for the global Rabi-drive Hamiltonian. These
            values should be in the range [``-1``, ``1``] and will be linearly
            rescaled to the range
            [− :attr:`max_drive_strength`, :attr:`max_drive_strength`] where
            ``-1`` corresponds to the − :attr:`max_drive_strength` and ``1`` to
            the :attr:`max_drive_strength`.
        drive_frequencies : NDArray[Shape[n_drive_ctrl], complex]
            The frequencies to modulate the control amplitude of the global
            Rabi-drive Hamiltonian with
        J_ctrl_amp : NDArray[Shape[n_time_steps, n_J_ctrl], complex]
            The control amplitudes for the exchange Hamiltonians. These
            values should be in the range [``-1``, ``1``] and will be linearly
            rescaled to the range
            [:attr:`J_min`, :attr:`J_max`] where ``-1`` corresponds to the
            :attr:`J_min` and ``1`` to the :attr:`J_max`.
        initial_state : NDArray[Shape[dim], complex]
            The initial state for the integrator
        dt : float
            The itegration time step

        Returns
        -------
        tuple[tf.Tensor[Shape[n_time_steps, total_n_channels], tf.complex128], tf.Tensor[Shape[:attr:`state_shape`], tf.complex128], float, tf.Tensor[Shape[n_time_steps, total_n_channels], tf.complex128], list[int]]
            A tuple of
            1. The control amplitude envolopes
            2. The initial state
            3. The integrator time step
            4. The frequencies to modulate the control amplitude envolopes with
            5. A list of the number of channels for each control Hamiltonian

            Warning
            -------
            The number of channels for each control Hamiltonian must be stored
            as a ``list`` and not an ``NDArray`` or a
            `TensorFlow <https://www.tensorflow.org>`__ tensor.


        .. _pre_processing_see_also:
        
        See Also
        --------
        * :meth:`propagate()`
        * :meth:`propagate_collection()`
        * :meth:`propagate_all()`
        * :meth:`evolved_expectation_value()`
        * :meth:`evolved_expectation_value_all()`
        * :meth:`get_driving_pulses()`
        * :meth:`gradient()`
        """
        frequencies = tf.concat([drive_frequencies,
                                 [0]*(drive_frequencies.shape[0]-1)],
                                axis=0)
        number_channels = [drive_frequencies.shape[0]] \
                         +[1] * (drive_frequencies.shape[0] - 1)
        ctrl_amp = self._rescale_and_concatenate([(drive_ctrl_amp,),
                                                  (J_ctrl_amp,)])
        return QuantumSystem._pre_processing(self,
                                             ctrl_amp,
                                             initial_state,
                                             dt,
                                             frequencies,
                                             number_channels)