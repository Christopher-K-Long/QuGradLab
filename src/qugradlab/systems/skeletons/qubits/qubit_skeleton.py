"""
Defines functions for constructing the skeleton out of Pauli operators.
"""

import functools

import numpy as np

from .... import hilbert_spaces

PAULI = np.array([[[ 0,  1 ],  # Pauli-X
                   [ 1,  0 ]],
                  [[ 0,  1j],  # Pauli-Y
                   [-1j, 0 ]],
                  [[-1,  0 ],  # Pauli-Z
                   [ 0,  1 ]]])

LADDER_BASIS = np.array([[[ 0,  1],
                          [ 0,  0]],
                         [[ 0,  0],
                          [ 1,  0]],
                         [[-1, 0],
                          [ 0,  1]]])

def second_order_tensor(hilbert_space: hilbert_spaces.QubitSpace,
                        xyz_basis: bool = True):
    r"""
    Generates a tensor given by the following expression
    $$
    t_{i,j,\alpha,\beta}\coloneqq \left(\sigma^{(i)}_j\right)_{\alpha,\beta}
    $$
    where $\sigma^{(i)}_j$ is the $j$th Pauli operator acting on the $i$th
    qubit, and $\alpha$ and $\beta$ are the rows and columns of the matrix
    representation of the operator $\sigma^{(i)}_j$ in the computational basis.

    Parameters
    ----------
    hilbert_space: QubitSpace
        The hilbert space of the qubits
    xyz_basis: bool
        Uses the Pauli basis if ``True`` and the
        $\sigma_{\pm}\coloneqq(\sigma_1\pm i\sigma_2)/2$ and Pauli-z basis if
        ``False``, by default ``True``

    Returns
    -------
    NDArray[Shape[``hilbert_space.qubits``, 3, ``hilbert_space.dim``, ``hilbert_space.dim``], int]
        The second order skeleton tensor
    """
    basis = PAULI if xyz_basis else LADDER_BASIS
    qubits = hilbert_space.qubits
    t = np.zeros((qubits, 3, len(hilbert_space), len(hilbert_space)),
                 dtype=complex)
    for i in range(qubits):
        for j, operator in enumerate(basis):
            t[i, j] = functools.reduce(np.kron, [np.identity(2**(qubits-1-i)),
                                                 operator,
                                                 np.identity(2**i)])
    return t

def fourth_order_tensor(hilbert_space: hilbert_spaces.QubitSpace,
                        xyz_basis: bool = True):
    r"""
    Generates a tensor given by the following expression
    $$
    u_{i,j,k,l,\alpha,\beta}\coloneqq \left(\sigma^{(i)}_k\sigma^{(j)}_l\right)_{\alpha,\beta}
    $$
    where $\sigma^{(i)}_j$ is the $j$th Pauli operator acting on the $i$th
    qubit, and $\alpha$ and $\beta$ are the rows and columns of the matrix
    representation of the operator $\sigma^{(i)}_k\sigma^{(j)}_l$ in the
    computational basis.

    Parameters
    ----------
    hilbert_space: QubitSpace
        The hilbert space of the qubits
    xyz_basis: bool
        Uses the Pauli basis if ``True`` and the
        $\sigma_{\pm}\coloneqq(\sigma_1\pm i\sigma_2)/2$ and Pauli-z basis if
        ``False``, by default ``True``

    Returns
    -------
    NDArray[Shape[``hilbert_space.qubits``, ``hilbert_space.qubits``, 3, 3, ``hilbert_space.dim``, ``hilbert_space.dim``], int]
        The fourth order skeleton tensor
    """
    basis = PAULI if xyz_basis else LADDER_BASIS
    qubits = hilbert_space.qubits
    u = np.zeros((qubits,
                  qubits,
                  3,
                  3,
                  hilbert_space.dim,
                  hilbert_space.dim),
                 dtype=complex)
    for i in range(qubits):
        for j in range(qubits):
            if i != j:
                for k, operator1 in enumerate(basis):
                    for l, operator2 in enumerate(basis):
                        indices = np.array([i, j])
                        operators = np.array([operator1, operator2])
                        order = np.argsort(indices)
                        indices = indices[order]
                        operators = operators[order]
                        u[i, j, k, l] = \
                            functools.reduce(np.kron,
                                [np.identity(2**(qubits-1-indices[1])),
                                 operators[1],
                                 np.identity(2**(indices[1]-indices[0]-1)),
                                 operators[0],
                                 np.identity(2**indices[0])])
    return u