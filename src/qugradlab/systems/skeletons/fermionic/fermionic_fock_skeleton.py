"""
Defines functions for constructing the Fermionic Fock skeleton out of second and
fourth order terms in creation and anihilation operators.
"""

import numpy as np

from ....hilbert_spaces.fermionic._hamming_weight_operations import \
    hamming_weight as _hamming_weight
from ._fermionic_fock_operations import is_empty as _is_empty,\
                                        is_occupied as _is_occupied,\
                                        select_orbitals as _select_orbitals,\
                                        flips as _flips
from ....hilbert_spaces import fermionic

def second_order_tensor(fock_space: fermionic.FermionSpace
                       ) -> np.ndarray[int]:
    r"""
    Generates a tensor given by the following expression
    $$
    t_{i,j,\alpha,\beta}\coloneqq \left(c^\dagger_ic_j\right)_{\alpha,\beta}
    $$
    where $c^\dagger_i$ and $c_i$ are the fermionic creation and anhilation
    operators acting on the $i$th orbital, and $\alpha$ and $\beta$ are the rows
    and columns of the matrix representation of the operator $c^\dagger_ic_j$ in
    the Fock basis.

    Parameters
    ----------
    fock_space: fermionic.FermionSpace
        The fermionic Fock space

    Returns
    -------
    NDArray[Shape[``fock_space.n_single_particle_states``, ``fock_space.n_single_particle_states``, ``fock_space.dim``, ``fock_space.dim``], int]
        The second order skeleton tensor
    """
    t = np.zeros((fock_space.n_single_particle_states,
                  fock_space.n_single_particle_states,
                  fock_space.dim,
                  fock_space.dim),
                 dtype=int)
    for j in range(fock_space.n_single_particle_states):
        is_occupied = _is_occupied(fock_space.basis, j)
        for i in range(j):
            non_zero = np.copy(is_occupied)
            non_zero[is_occupied] &= _is_empty(fock_space[is_occupied], i)
            occupation_indices = fock_space.inverse[_flips(fock_space[non_zero], [i, j])]
            slice = np.power(-1, _hamming_weight(_select_orbitals(fock_space[non_zero], i, j)))
            t[i, j, occupation_indices, non_zero] = slice
            t[j, i, non_zero, occupation_indices] = slice
        t[j, j] = np.diag(is_occupied)
    return t
            
def fourth_order_tensor(fock_space: fermionic.FermionSpace
                       ) -> np.ndarray[int]:
    r"""
    Generates a tensor given by the following expression
    $$
    u_{i,j,k,l,\alpha,\beta}\coloneqq \left(c^\dagger_ia^\dagger_ja_ka_l\right)_{\alpha,\beta}
    $$
    where $c^\dagger_i$ and $c_i$ are the fermionic creation and anhilation
    operators acting on the $i$th orbital, and $\alpha$ and $\beta$ are the rows
    and columns of the matrix representation of the operator $c^\dagger_ic_j$ in
    the Fock basis.

    Parameters
    ----------
    fock_space: FermionSpace
        The fermionic Fock space

    Returns
    -------
    NDArray[Shape[``fock_space.n_single_particle_states``, ``fock_space.n_single_particle_states``, ``fock_space.n_single_particle_states``, ``fock_space.n_single_particle_states``, ``fock_space.dim``, ``fock_space.dim``], int]
        The fourth order skeleton tensor
    """
    u = np.zeros((fock_space.n_single_particle_states,
                  fock_space.n_single_particle_states,
                  fock_space.n_single_particle_states,
                  fock_space.n_single_particle_states,
                  fock_space.dim,
                  fock_space.dim),
                 dtype=int)
    for l in range(fock_space.n_single_particle_states):
        is_occupied = _is_occupied(fock_space.basis, l)
        for k in range(l):
            non_zero_k = np.copy(is_occupied)
            non_zero_k[is_occupied] &= _is_occupied(fock_space[is_occupied], k)
            for j in range(fock_space.n_single_particle_states):
                if j in [k, l]:
                    non_zero_j = non_zero_k
                else:
                    non_zero_j = np.copy(non_zero_k)
                    non_zero_j[non_zero_k] &= _is_empty(fock_space[non_zero_k], j)
                for i in range(j):
                    if i in [k, l]:
                        non_zero = non_zero_j
                    else:
                        non_zero = np.copy(non_zero_j)
                        non_zero[non_zero_j] &= _is_empty(fock_space[non_zero_j], i)
                    occupation_indices = fock_space.inverse[_flips(fock_space[non_zero], [i, j, k, l])]
                    slice = np.power(-1, _hamming_weight(_select_orbitals(fock_space[non_zero], i, j))
                                        +_hamming_weight(_select_orbitals(fock_space[non_zero], k, l))
                                        +(i>k)
                                        +(i>l)
                                        +(j>k)
                                        +(j>l)
                                    )
                    u[i, j, k, l, occupation_indices, non_zero] = slice
                    u[i, j, l, k, occupation_indices, non_zero] = -slice
                    u[j, i, k, l, occupation_indices, non_zero] = -slice
                    u[j, i, l, k, occupation_indices, non_zero] = slice

                    u[k, l, i, j, non_zero, occupation_indices] = slice
                    u[l, k, i, j, non_zero, occupation_indices] = -slice
                    u[k, l, j, i, non_zero, occupation_indices] = -slice
                    u[l, k, j, i, non_zero, occupation_indices] = slice
    return u