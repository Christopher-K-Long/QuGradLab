"""
:class:`~qugradlab.systems.skeletons.SkeletalSystem` s of fermions.
"""

import numpy as np

from . import fermionic_fock_skeleton
from .. import SkeletalSystem
from ....hilbert_spaces import fermionic

class FermionicSystem(SkeletalSystem):
    r"""A :class:`qugrad.QuantumSystem` for a system of fermions.
    The Hamiltonian is constructed in the form:
    $$
    H(t) = \sum_{i,j}t_{ij}c_i^\dagger c_j
    +\sum_{i,j,k,l}U_{ijkl}c_i^\dagger c_j^\dagger c_k c_l
    +\sum_{m,i,j}a_m(t)h_{mij}c_i^\dagger c_j
    $$
    where $c_i^\dagger$ and $c_i$ are the fermionic creation and anhilation
    operators acting on the $i$th orbital, $t_{ij}$ are the hoppings for the
    drift Hamiltonian, $U_{ijkl}$ corresponds to the Coulomb integrals for the
    drift Hamiltonian, $h_{mij}$ corresponds to hoppings for the $m$th control
    Hamiltonian, and $a_m(t)$ are the time-dependent control amplitdues that
    modulate the $m$th control Hamiltonian.
    """
    def __init__(self,
                 hilbert_space: fermionic.FermionSpace,
                 drift_hoppings: np.ndarray[complex],
                 coulomb_integrals: np.ndarray[complex],
                 ctrl_hoppings: np.ndarray[complex],
                 use_graph: bool = True):
        r"""Creates an instance of a `FermionicSystem`. The Hamiltonian is
        constructed in the form:
        $$
        H(t) = \sum_{i,j}t_{ij}c_i^\dagger c_j
        +\sum_{i,j,k,l}U_{ijkl}c_i^\dagger c_j^\dagger c_k c_l
        +\sum_{m,i,j}a_m(t)h_{mij}c_i^\dagger c_j
        $$
        where $c_i^\dagger$ and $c_i$ are the fermionic creation and anhilation
        operators acting on the $i$th orbital, $t_{ij}$ corresponds to
        `drift_hoppings`, $U_{ijkl}$ corresponds to `coulomb_integrals`,
        $h_{mij}$ corresponds to `ctrl_hoppings`, and $a_m(t)$ are the
        time-dependent control amplitdues.
        
        Parameters
        ----------
        hilbert_space : FermionSpace
            The Hilbert space of the system of fermions
        drift_hoppings : NDArray[Shape[``hilbert_space.n_single_particle_states``, ``hilbert_space.n_single_particle_states``"], complex]
            The hopping coefficients for the drift Hamiltonian
        coulomb_integrals : NDArray[Shape[``hilbert_space.n_single_particle_states``, ``hilbert_space.n_single_particle_states``, ``hilbert_space.n_single_particle_states``, ``hilbert_space.n_single_particle_states``], complex]
            The Coulomb integrals for the drift Hamiltonian
        ctrl_hoppings : NDArray[Shape[:attr:`n_ctrl`, ``hilbert_space.n_single_particle_states``, ``hilbert_space.n_single_particle_states``], complex]
            An array of hopping coefficients for the control Hamiltonians
        use_graph : bool
            Whether to use `TensorFlow <https://www.tensorflow.org>`__ graphs
            during computation, by default ``True``
        """
        t = fermionic_fock_skeleton.second_order_tensor(hilbert_space)
        u = fermionic_fock_skeleton.fourth_order_tensor(hilbert_space)
        super().__init__(drift_coefficients=[drift_hoppings, coulomb_integrals],
                         drift_skeletons   =[t,              u],
                         ctrl_coefficients =[ctrl_hoppings],
                         ctrl_skeletons    =[t],
                         hilbert_space     =hilbert_space,
                         use_graph         =use_graph)
        del u
        del t