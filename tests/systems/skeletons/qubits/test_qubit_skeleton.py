import numpy as np
from qugradlab.systems.skeletons.qubits import qubit_skeleton
from qugradlab.hilbert_spaces import QubitSpace
from functools import reduce

I = np.identity(2)

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

def kron(*operators):
    return reduce(np.kron, operators)

def second_order_tensor_check(xyz_basis: bool):
    basis = PAULI if xyz_basis else LADDER_BASIS
    hilbert_space = QubitSpace(3)
    t = qubit_skeleton.second_order_tensor(hilbert_space, xyz_basis)
    assert np.array_equal(t[0, 0], kron(I, I, basis[0]))
    assert np.array_equal(t[0, 1], kron(I, I, basis[1]))
    assert np.array_equal(t[0, 2], kron(I, I, basis[2]))

    assert np.array_equal(t[1, 0], kron(I, basis[0], I))
    assert np.array_equal(t[1, 1], kron(I, basis[1], I))
    assert np.array_equal(t[1, 2], kron(I, basis[2], I))

    assert np.array_equal(t[2, 0], kron(basis[0], I, I))
    assert np.array_equal(t[2, 1], kron(basis[1], I, I))
    assert np.array_equal(t[2, 2], kron(basis[2], I, I))
            
def fourth_order_tensor_check(xyz_basis: bool):
    basis = PAULI if xyz_basis else LADDER_BASIS
    hilbert_space = QubitSpace(3)
    t = qubit_skeleton.fourth_order_tensor(hilbert_space, xyz_basis)
    assert np.array_equal(t[0, 0], np.zeros((3, 3, 8, 8), dtype=complex))
    assert np.array_equal(t[1, 1], np.zeros((3, 3, 8, 8), dtype=complex))
    assert np.array_equal(t[2, 2], np.zeros((3, 3, 8, 8), dtype=complex))

    assert np.array_equal(t[0, 1, 0, 0], kron(I, basis[0], basis[0]))
    assert np.array_equal(t[1, 0, 0, 0], kron(I, basis[0], basis[0]))
    assert np.array_equal(t[0, 2, 0, 0], kron(basis[0], I, basis[0]))
    assert np.array_equal(t[2, 0, 0, 0], kron(basis[0], I, basis[0]))
    assert np.array_equal(t[1, 2, 0, 0], kron(basis[0], basis[0], I))
    assert np.array_equal(t[2, 1, 0, 0], kron(basis[0], basis[0], I))

    assert np.array_equal(t[0, 1, 0, 1], kron(I, basis[1], basis[0]))
    assert np.array_equal(t[1, 0, 0, 1], kron(I, basis[0], basis[1]))
    assert np.array_equal(t[0, 2, 0, 1], kron(basis[1], I, basis[0]))
    assert np.array_equal(t[2, 0, 0, 1], kron(basis[0], I, basis[1]))
    assert np.array_equal(t[1, 2, 0, 1], kron(basis[1], basis[0], I))
    assert np.array_equal(t[2, 1, 0, 1], kron(basis[0], basis[1], I))

    assert np.array_equal(t[0, 1, 0, 2], kron(I, basis[2], basis[0]))
    assert np.array_equal(t[1, 0, 0, 2], kron(I, basis[0], basis[2]))
    assert np.array_equal(t[0, 2, 0, 2], kron(basis[2], I, basis[0]))
    assert np.array_equal(t[2, 0, 0, 2], kron(basis[0], I, basis[2]))
    assert np.array_equal(t[1, 2, 0, 2], kron(basis[2], basis[0], I))
    assert np.array_equal(t[2, 1, 0, 2], kron(basis[0], basis[2], I))

    assert np.array_equal(t[0, 1, 1, 0], kron(I, basis[0], basis[1]))
    assert np.array_equal(t[1, 0, 1, 0], kron(I, basis[1], basis[0]))
    assert np.array_equal(t[0, 2, 1, 0], kron(basis[0], I, basis[1]))
    assert np.array_equal(t[2, 0, 1, 0], kron(basis[1], I, basis[0]))
    assert np.array_equal(t[1, 2, 1, 0], kron(basis[0], basis[1], I))
    assert np.array_equal(t[2, 1, 1, 0], kron(basis[1], basis[0], I))

    assert np.array_equal(t[0, 1, 1, 1], kron(I, basis[1], basis[1]))
    assert np.array_equal(t[1, 0, 1, 1], kron(I, basis[1], basis[1]))
    assert np.array_equal(t[0, 2, 1, 1], kron(basis[1], I, basis[1]))
    assert np.array_equal(t[2, 0, 1, 1], kron(basis[1], I, basis[1]))
    assert np.array_equal(t[1, 2, 1, 1], kron(basis[1], basis[1], I))
    assert np.array_equal(t[2, 1, 1, 1], kron(basis[1], basis[1], I))

    assert np.array_equal(t[0, 1, 1, 2], kron(I, basis[2], basis[1]))
    assert np.array_equal(t[1, 0, 1, 2], kron(I, basis[1], basis[2]))
    assert np.array_equal(t[0, 2, 1, 2], kron(basis[2], I, basis[1]))
    assert np.array_equal(t[2, 0, 1, 2], kron(basis[1], I, basis[2]))
    assert np.array_equal(t[1, 2, 1, 2], kron(basis[2], basis[1], I))
    assert np.array_equal(t[2, 1, 1, 2], kron(basis[1], basis[2], I))

    assert np.array_equal(t[0, 1, 2, 0], kron(I, basis[0], basis[2]))
    assert np.array_equal(t[1, 0, 2, 0], kron(I, basis[2], basis[0]))
    assert np.array_equal(t[0, 2, 2, 0], kron(basis[0], I, basis[2]))
    assert np.array_equal(t[2, 0, 2, 0], kron(basis[2], I, basis[0]))
    assert np.array_equal(t[1, 2, 2, 0], kron(basis[0], basis[2], I))
    assert np.array_equal(t[2, 1, 2, 0], kron(basis[2], basis[0], I))

    assert np.array_equal(t[0, 1, 2, 1], kron(I, basis[1], basis[2]))
    assert np.array_equal(t[1, 0, 2, 1], kron(I, basis[2], basis[1]))
    assert np.array_equal(t[0, 2, 2, 1], kron(basis[1], I, basis[2]))
    assert np.array_equal(t[2, 0, 2, 1], kron(basis[2], I, basis[1]))
    assert np.array_equal(t[1, 2, 2, 1], kron(basis[1], basis[2], I))
    assert np.array_equal(t[2, 1, 2, 1], kron(basis[2], basis[1], I))

    assert np.array_equal(t[0, 1, 2, 2], kron(I, basis[2], basis[2]))
    assert np.array_equal(t[1, 0, 2, 2], kron(I, basis[2], basis[2]))
    assert np.array_equal(t[0, 2, 2, 2], kron(basis[2], I, basis[2]))
    assert np.array_equal(t[2, 0, 2, 2], kron(basis[2], I, basis[2]))
    assert np.array_equal(t[1, 2, 2, 2], kron(basis[2], basis[2], I))
    assert np.array_equal(t[2, 1, 2, 2], kron(basis[2], basis[2], I))
 
def test_second_order_tensor():
    second_order_tensor_check(True)

def test_second_order_tensor_ladder_basis():
    second_order_tensor_check(False)

def test_fourth_order_tensor():
    fourth_order_tensor_check(True)

def test_fourth_order_tensor_ladder_basis():
    fourth_order_tensor_check(False)