"""The control scheme for electron spin resonance (ESR) devices"""

from typing import Callable

import numpy as np

from ._device import Device

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
            The maximum drive strength that can be applied to the system
        J_min : float
            The minimum value of the exchange coupling $J$
        J_max : float
            The maximum value of the exchange coupling $J$
        """
        ...
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
        initial_state : NDArray[Shape[:attr:`dim`], complex]
            The initial state for the integrator
        dt : float
            The itegration time step

        Returns
        -------
        tuple[tf.Tensor[Shape[n_time_steps, total_n_channels], tf.complex128], tf.Tensor[Shape[:attr::attr:`state_shape`], tf.complex128], float, tf.Tensor[Shape[n_time_steps, total_n_channels], tf.complex128], list[int]]
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
        ...
    def propagate(self,
                        drive_ctrl_amp: np.ndarray[complex],
                        drive_frequencies: np.ndarray[complex],
                        J_ctrl_amp: np.ndarray[complex],
                        initial_state: np.ndarray[complex],
                        dt: float
                 ) -> np.ndarray[complex]:
        """
        Evolves a state vector under the time-dependent Hamiltonian defined by
        the control amplitudes using
        :meth:`~py_ste.evolvers.DenseUnitaryEvolver.propagate()`
        from `PySTE <https://PySTE.readthedocs.io>`__.

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
        initial_state : NDArray[Shape[:attr:`dim`], complex]
            The initial state for the integrator
        dt : float
            The itegration time step

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        NDArray[Shape[:attr:`state_shape`], complex]
            The final state

        See Also
        --------
        * :meth:`propagate_collection()`
        * :meth:`propagate_all()`
        """
        ...
    def propagate_collection(self,
                             drive_ctrl_amp: np.ndarray[complex],
                             drive_frequencies: np.ndarray[complex],
                             J_ctrl_amp: np.ndarray[complex],
                             initial_state: np.ndarray[complex],
                             dt: float
                            ) -> np.ndarray[complex]:
        """
        Evolves a collection of state vectors under the time-dependent
        Hamiltonian defined by the control amplitudes using
        :meth:`~py_ste.evolvers.DenseUnitaryEvolver.propagate_collection()`
        from `PySTE <https://PySTE.readthedocs.io>`__.

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
        initial_state : NDArray[Shape[:attr:`dim`], complex]
            The initial state for the integrator
        dt : float
            The itegration time step

            Warning
            -------
            This must be a ``list`` and not an ``NDArray`` or a
            `TensorFlow <https://www.tensorflow.org>`__ tensor.

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        NDArray[Shape[n_states, :attr:`state_shape`], complex]
            The final state

        See Also
        --------
        * :meth:`propagate()`
        * :meth:`propagate_all()`
        """
        ...
    def propagate_all(self,
                      drive_ctrl_amp: np.ndarray[complex],
                      drive_frequencies: np.ndarray[complex],
                      J_ctrl_amp: np.ndarray[complex],
                      initial_state: np.ndarray[complex],
                      dt: float
                     ) -> np.ndarray[complex]:
        """
        Evolves a state vector under the time-dependent Hamiltonian defined by
        the control amplitudes using
        :meth:`~py_ste.evolvers.DenseUnitaryEvolver.propagate_all()`
        from `PySTE <https://PySTE.readthedocs.io>`__ and returns the state at
        each time-step.

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
        initial_state : NDArray[Shape[:attr:`dim`], complex]
            The initial state for the integrator
        dt : float
            The itegration time step

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        NDArray[Shape[n_time_steps+1, :attr:`state_shape`], complex]
            The state at each integrator time step (including the initial
            state).

        See Also
        --------
        * :meth:`propagate()`
        * :meth:`propagate_collection()`
        """
        ...
    def evolved_expectation_value(self,
                                  drive_ctrl_amp: np.ndarray[complex],
                                  drive_frequencies: np.ndarray[complex],
                                  J_ctrl_amp: np.ndarray[complex],
                                  initial_state: np.ndarray[complex],
                                  dt: float,
                                  observable : np.ndarray[complex]
                                 ) -> complex:
        """
        Evolves a state vector under the time-dependent Hamiltonian defined by
        the control amplitudes and computes the expectation value of a specified
        observable with respect to the final state using
        :meth:`~py_ste.evolvers.DenseUnitaryEvolver.evolved_expectation_value()`
        from `PySTE <https://PySTE.readthedocs.io>`__.

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
        initial_state : NDArray[Shape[:attr:`dim`], complex]
            The initial state for the integrator
        dt : float
            The itegration time step
        observable : NDArray[Shape[:attr:`dim`, :attr:`dim`], complex]
            The observable to take the expectation value of.

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        complex
            The expectation value.

        See Also
        --------
        * :meth:`evolved_expectation_value_all()`
        * :meth:`gradient()`
        """
        ...
    def evolved_expectation_value_all(self,
                                      drive_ctrl_amp: np.ndarray[complex],
                                      drive_frequencies: np.ndarray[complex],
                                      J_ctrl_amp: np.ndarray[complex],
                                      initial_state: np.ndarray[complex],
                                      dt: float,
                                      observable : np.ndarray[complex]
                                     ) -> np.ndarray[complex]:
        """
        Evolves a state vector under the time-dependent Hamiltonian defined by
        the control amplitudes and computes the expectation value of a specified
        observable with respect to the state at each time-step using
        :meth:`~py_ste.evolvers.DenseUnitaryEvolver.evolved_expectation_value_all()`
        from `PySTE <https://PySTE.readthedocs.io>`__.

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
        initial_state : NDArray[Shape[:attr:`dim`], complex]
            The initial state for the integrator
        dt : float
            The itegration time step
        observable : NDArray[Shape[:attr:`dim`, :attr:`dim`], complex]
            The observable to take the expectation value of.

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        NDArray[Shape[n_time_steps+1], complex]
            The state at each integrator time step (including the initial
            state).

        See Also
        --------
        * :meth:`evolved_expectation_value()`
        * :meth:`gradient()`
        """
        ...
    def get_driving_pulses(self,
                           drive_ctrl_amp: np.ndarray[complex],
                           drive_frequencies: np.ndarray[complex],
                           J_ctrl_amp: np.ndarray[complex],
                           initial_state: np.ndarray[complex],
                           dt: float
                          ) -> tuple[np.ndarray[complex], np.ndarray[complex], float]:
        """
        When calling any evolution method (listed in the See also section`)
        `get_driving_pulses()` is executed on the arguements before the
        evolution method.

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
        initial_state : NDArray[Shape[:attr:`dim`], complex]
            The initial state for the integrator
        dt : float
            The itegration time step

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        tuple[NDArray[Shape[n_time_steps, `n_ctrl`], complex], NDArray[Shape[:attr:`state_shape`], complex], float]
            A tuple of:
            1. Control amplitudes
            2. Initial state
            3. Integrator time step


        .. _get_driving_pulses_see_also:
        
        See Also
        --------
        * :meth:`propagate()`
        * :meth:`propagate_collection()`
        * :meth:`propagate_all()`
        * :meth:`evolved_expectation_value()`
        * :meth:`evolved_expectation_value_all()`
        * :meth:`gradient()`
        """
        ...
    def _eager_processing(self,
                          drive_ctrl_amp: np.ndarray[complex],
                          drive_frequencies: np.ndarray[complex],
                          J_ctrl_amp: np.ndarray[complex],
                          initial_state: np.ndarray[complex],
                          dt: float
                         ) -> tuple:
        """
        Executes `_pre_processing()` followed by
        `_envolope_processing()` eagerly (i.e. without using a
        `TensorFlow <https://www.tensorflow.org>`__ graph). Nonetheless,
        `_eager_processing()` is still auto differentiable.

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
        initial_state : NDArray[Shape[:attr:`dim`], complex]
            The initial state for the integrator
        dt : float
            The itegration time step

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        tuple[tf.Tensor[Shape[n_time_steps, `n_ctrl`], complex], tf.Tensor[Shape[:attr:`state_shape`], complex], tf.Tensor[Shape[], float]]
            A tuple of:
            1. Control amplitudes
            2. Initial state
            3. Integrator time step
        """
        ...
    def _traceable_eager_processing(self,
                                   drive_ctrl_amp: np.ndarray[complex],
                                   drive_frequencies: np.ndarray[complex],
                                   J_ctrl_amp: np.ndarray[complex],
                                   initial_state: np.ndarray[complex],
                                   dt: float
                                  ) -> tuple:
        """
        A function that will be traced by
        `TensorFlow <https://www.tensorflow.org>`__ to produce a graph of
        `_pre_processing()` followed by `_envolope_processing()`.

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
        initial_state : NDArray[Shape[:attr:`dim`], complex]
            The initial state for the integrator
        dt : float
            The itegration time step

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        tuple[tf.Tensor[Shape[n_time_steps, `n_ctrl`], complex], tf.Tensor[Shape[:attr:`state_shape`], complex], tf.Tensor[Shape[], float]]
            A tuple of:
            1. Control amplitudes
            2. Initial state
            3. Integrator time step
        """
        ...
    def gradient(self,
                 drive_ctrl_amp: np.ndarray[complex],
                 drive_frequencies: np.ndarray[complex],
                 J_ctrl_amp: np.ndarray[complex],
                 initial_state: np.ndarray[complex],
                 dt: float,
                 observable : np.ndarray[complex]
                ) -> tuple[float, np.ndarray[float]]:
        """
        Evolves a state vector under the time-dependent Hamiltonian defined by
        the control amplitudes and computes the expectation value of a specified
        observable with respect to the final state and then computes the
        gradient of the final state with respect to the first argument
        (``args[0]``) using
        :meth:`~py_ste.evolvers.DenseUnitaryEvolver.switching_function()`
        from `PySTE <https://PySTE.readthedocs.io>`__.

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
        initial_state : NDArray[Shape[:attr:`dim`], complex]
            The initial state for the integrator
        dt : float
            The itegration time step
        observable : NDArray[Shape[:attr:`dim`, :attr:`dim`], complex]
            The observable to take the expectation value of.

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        tuple[complex, NDArray[Shape[n_parameters], float]]
            A tuple of the expectation value and the gradient.

        See Also
        --------
        * :meth:`evolved_expectation_value()`
        * :meth:`evolved_expectation_value_all()`
        """
        ...