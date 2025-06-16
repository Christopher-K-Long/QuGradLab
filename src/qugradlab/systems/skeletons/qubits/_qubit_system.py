"""
:class:`~qugradlab.systems.skeletons.SkeletalSystem` s of qubits.
"""

import numpy as np

from . import qubit_skeleton
from .. import SkeletalSystem
from ....hilbert_spaces import _qudit_hilbert_spaces

class QubitSystem(SkeletalSystem):
    r"""A :class:`qugrad.QuantumSystem` for a system of qubits.
    The Hamiltonian is constructed in the form:
    $$
    H(t) = \sum_{i,j}t_{ij}\sigma^{(i)}_j
    +\sum_{i,j,k,l}U_{ijkl}\sigma^{(i)}_k\sigma^{(j)}_l
    +\sum_{m,i,j}a_m(t)h_{mij}\sigma^{(i)}_j
    +\sum_{m,i,j,k,l}b_m(t)J_{mijkl}\sigma^{(i)}_k\sigma^{(j)}_l
    $$
    where $\sigma^{(i)}_j$ is the $j$th Pauli operator acting on the $i$th
    qubit, $t_{ij}$ are the single-qubit coefficients for the drift Hamiltonian,
    $U_{ijkl}$ are the two-qubit coefficients for the drift Hamiltonian,
    $h_{mij}$ are the single-qubit coefficients for the $m$th control
    Hamiltonian, $J_{mijk}$ are the single-qubit coefficients for the $m$th
    control Hamiltonian, $a_m(t)$ are the time-dependent control amplitdues that
    modulate the $m$th single-qubit control Hamiltonian, and $b_m(t)$ are the
    time-dependent control amplitdues that modulate the $m$th two-qubit control
    Hamiltonian.
    """
    def __init__(self,
                 hilbert_space: _qudit_hilbert_spaces.QubitSpace,
                 single_qubit_drift_coefficients: np.ndarray[complex],
                 two_qubit_drift_coefficients: np.ndarray[complex],
                 single_qubit_ctrl_coefficients: np.ndarray[complex],
                 two_qubit_ctrl_coefficients: np.ndarray[complex],
                 use_graph: bool = True):
        r"""Creates an instance of a `QubitSystem`. The Hamiltonian is
        constructed in the form:
        $$
        H(t) = \sum_{i,j}t_{ij}\sigma^{(i)}_j
        +\sum_{i,j,k,l}U_{ijkl}\sigma^{(i)}_k\sigma^{(j)}_l
        +\sum_{m,i,j}a_m(t)h_{mij}\sigma^{(i)}_j
        +\sum_{m,i,j,k,l}b_m(t)J_{mijkl}\sigma^{(i)}_k\sigma^{(j)}_l
        $$
        where $\sigma^{(i)}_j$ is the $j$th Pauli operator acting on the $i$th
        qubit, $t_{ij}$ corresponds to `single_qubit_drift_coefficients`,
        $U_{ijkl}$ corresponds to `two_qubit_drift_coefficients`, $h_{mij}$
        corresponds to `single_qubit_ctrl_coefficients`, $J_{mijk}$ corresponds
        to `two_qubit_ctrl_coefficients`, $a_m(t)$ are the time-dependent
        control amplitdues that modulate the $m$th single-qubit control
        Hamiltonian, and $b_m(t)$ are the time-dependent control amplitdues that
        modulate the $m$th two-qubit control Hamiltonian.
        
        Parameters
        ----------
        hilbert_space: QubitSpace,
            The Hilbert space of the system of qubits
        single_qubit_drift_coefficients : NDArray[Shape[hilbert_space.qubits, 3], complex]
            The single-qubit coefficients for the drift Hamiltonian
        two_qubit_drift_coefficients : NDArray[Shape[hilbert_space.qubits, hilbert_space.qubits, 3, 3], complex]
            The two-qubit coefficients for the drift Hamiltonian
        single_qubit_ctrl_coefficients : NDArray[Shape[n_single_qubit_ctrl, hilbert_space.qubits, 3], complex]
            The single-qubit coefficients for the control Hamiltonian
        two_qubit_ctrl_coefficients : NDArray[Shape[n_two_qubit_ctrl, hilbert_space.qubits, hilbert_space.qubits, 3, 3], complex]
            The two-qubit coefficients for the control Hamiltonian
        use_graph : bool
            Whether to use `TensorFlow <https://www.tensorflow.org>`__ graphs
            during computation, by default ``True``
        """
        t = qubit_skeleton.second_order_tensor(hilbert_space)
        u = qubit_skeleton.fourth_order_tensor(hilbert_space)
        skeletons = [t, u]
        super().__init__(drift_coefficients = [single_qubit_drift_coefficients,
                                               two_qubit_drift_coefficients],
                         drift_skeletons  = skeletons,
                         ctrl_coefficients =  [single_qubit_ctrl_coefficients,
                                               two_qubit_ctrl_coefficients],
                         ctrl_skeletons = skeletons,
                         hilbert_space = hilbert_space,
                         use_graph = use_graph)
        del skeletons
        del u
        del t
