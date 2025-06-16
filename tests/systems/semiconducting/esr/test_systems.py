from functools import reduce
from math import comb

import numpy as np
from scipy.sparse import csr_matrix

from qugradlab.systems.semiconducting.esr import SpinChain, ValleyChain, \
                                                 SpinChainAngledDrive
I = np.identity(2)
X = np.array([[ 0,  1],  # Pauli X
              [ 1,  0]])
Y = np.array([[ 0,  1j], # Pauli Y
              [-1j, 0]])
Z = np.array([[-1,  0],  # Pauli Z
              [ 0,  1]])
J = np.array([[0.25,  0,     0,    0   ],
              [0,    -0.25,  0.5,  0   ],
              [0,     0.5,  -0.25, 0   ],
              [0,     0,     0,    0.25]])

def kron(*operators):
    return reduce(np.kron, operators)

def test_spin_chain_initialisation():
    for ferromagnetic in [True, False]:
        for use_graph in [True, False]:
            device = SpinChain(3,
                               [0.1, 0.2, 0.3],
                               3,
                               2,
                               1,
                               ferromagnetic,
                               use_graph)
            sign = -1 if ferromagnetic else 1
            assert device.dim == 2**3
            assert np.allclose(device.H0, 0.5*np.diag([-0.6, -0.4, -0.2, 0, 0, 0.2, 0.4, 0.6]))
            assert np.array_equal(device.Hs, np.array([0.5*(kron(X, I, I)+kron(I, X, I)+kron(I, I, X)), kron(I, sign*J), kron(sign*J, I)]))
            assert device.using_graph == use_graph
            assert device.ferromagnetic == ferromagnetic

def test_spin_chain_ferromangetic_read_only():
    device = SpinChain(3, [0.1, 0.2, 0.3], 3, 2, 1)
    try:
        device.ferromagnetic = False
    except AttributeError:
        pass
    else:
        raise AssertionError("SpinChain.ferromagnetic should be read-only")


def test_spin_chain_angled_drive_initialisation():
    for ferromagnetic in [True, False]:
        for use_graph in [True, False]:
            device = SpinChainAngledDrive(3,
                                          [0.1, 0.2, 0.3],
                                          [[1, 0, 0],
                                           [0, 1, 0],
                                           [1/np.sqrt(3),
                                            1/np.sqrt(3),
                                            1/np.sqrt(3)]],
                                          3,
                                          2,
                                          1,
                                          ferromagnetic,
                                          use_graph)
            sign = -1 if ferromagnetic else 1
            assert device.dim == 2**3
            assert np.allclose(device.H0, 0.5*np.diag([-0.6, -0.4, -0.2, 0, 0, 0.2, 0.4, 0.6]))
            assert np.array_equal(device.Hs, np.array([kron(X, I, I)/np.sqrt(3)+kron(Y, I, I)/np.sqrt(3)+kron(Z, I, I)/np.sqrt(3)+kron(I, Y, I)+kron(I, I, X), kron(I, sign*J), kron(sign*J, I)]))
            assert device.using_graph == use_graph
            assert device.ferromagnetic == ferromagnetic

def test_spin_chain_angled_drive_ferromangetic_read_only():
    device = SpinChainAngledDrive(3, [0.1, 0.2, 0.3], 3, 2, 1)
    try:
        device.ferromagnetic = False
    except AttributeError:
        pass
    else:
        raise AssertionError("SpinChain.ferromagnetic should be read-only")

def test_spin_chain_angled_drive_drive_vectors_read_only():
    device = SpinChainAngledDrive(3,
                                  [0.1, 0.2, 0.3],
                                  [[1, 0, 0],
                                   [0, 1, 0],
                                   [1/np.sqrt(3),
                                    1/np.sqrt(3),
                                    1/np.sqrt(3)]],
                                  3,
                                  2,
                                  1,
                                  True,
                                  True)
    try:
        device.drive_vectors = np.identity(3)
    except AttributeError:
        pass
    else:
        raise AssertionError("SpinChain.drive_vectors should be read-only")
    try:
        device.drive_vectors[0] = np.array([1, 0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("SpinChain.drive_vectors should be read-only")

def test_valley_chain_initialisation():
    zeeman_splitting = 0.1
    valley_splitting = 10
    dots = 3
    electrons = 2
    u=20
    u_valley_flip = 15
    valley_spin_orbit_coupling = 0.1
    # test asignment of parameters
    for use_graph in [True, False]:
        device = ValleyChain(dots,
                             electrons,
                             [zeeman_splitting]*dots,
                             valley_splitting,
                             u,
                             u_valley_flip,
                             valley_spin_orbit_coupling,
                             3,
                             2,
                             1,
                             use_graph)
        assert device.dim == comb(4*dots, electrons)
        assert device.u == u
        assert device.u_valley_flip == u_valley_flip
        assert device.valley_spin_orbit_coupling == valley_spin_orbit_coupling
        assert device.using_graph == use_graph
    # test single dot single electron
    device = ValleyChain(1,
                         1,
                         [zeeman_splitting],
                         valley_splitting,
                         u,
                         u_valley_flip,
                         valley_spin_orbit_coupling,
                         3,
                         2,
                         1,
                         use_graph)
    assert np.allclose(device.H0, np.array([[zeeman_splitting/2+valley_splitting/2, 0, 0, valley_spin_orbit_coupling],
                                            [0, -zeeman_splitting/2+valley_splitting/2, valley_spin_orbit_coupling, 0],
                                            [0, valley_spin_orbit_coupling, zeeman_splitting/2-valley_splitting/2, 0],
                                            [valley_spin_orbit_coupling, 0, 0 ,-zeeman_splitting/2-valley_splitting/2]]))
    assert np.array_equal(device.Hs, 0.5*np.array([[[0, 1, 0, 0],
                                                    [1, 0, 0, 0],
                                                    [0, 0, 0, 1],
                                                    [0, 0, 1, 0]]]))
    # test single dot with 2 electrons
    device = ValleyChain(1,
                         2,
                         [zeeman_splitting],
                         valley_splitting,
                         u,
                         u_valley_flip,
                         valley_spin_orbit_coupling,
                         3,
                         2,
                         1,
                         use_graph)
    assert np.allclose(device.H0, np.array([[valley_splitting+u, valley_spin_orbit_coupling, -u_valley_flip, u_valley_flip, -valley_spin_orbit_coupling, 0],
                                            [valley_spin_orbit_coupling, zeeman_splitting+u, 0, 0, 0, -valley_spin_orbit_coupling],
                                            [-u_valley_flip, 0, u, 0, 0, -u_valley_flip],
                                            [u_valley_flip, 0, 0 , u, 0, u_valley_flip],
                                            [-valley_spin_orbit_coupling, 0, 0, 0, -zeeman_splitting+u, valley_spin_orbit_coupling],
                                            [0, -valley_spin_orbit_coupling, -u_valley_flip, u_valley_flip, valley_spin_orbit_coupling, -valley_splitting+u]]))
    assert np.array_equal(device.Hs, 0.5*np.array([[[0, 0, 0, 0, 0, 0],
                                                    [0, 0, 1, 1, 0, 0],
                                                    [0, 1, 0, 0, 1, 0],
                                                    [0, 1, 0, 0, 1, 0],
                                                    [0, 0, 1, 1, 0, 0],
                                                    [0, 0, 0, 0, 0, 0]]]))

    # test double dot with 2 electrons interdot hopping test
    device = ValleyChain(2,
                         2,
                         [zeeman_splitting]*2,
                         valley_splitting,
                         u,
                         u_valley_flip,
                         valley_spin_orbit_coupling,
                         3,
                         2,
                         1,
                         use_graph)
    row = [0,  0,  2,  2, 1,  1,  4,  4, 7,  7,  8,  8, 13, 13, 3,  3, 9,  9, 14, 14, 18, 18, 10, 10, 15, 15, 19, 19, 22, 22,  5,  5, 16, 16, 20, 20, 23, 23, 25, 25, 12, 12, 21, 21, 26, 26, 27, 27]
    col = [7, 10, 12, 16, 8, 15, 13, 22, 0, 14,  1, 19,  4, 26, 9, 21, 3, 25,  7, 10,  5, 27,  0, 14,  1, 19,  8, 15,  4, 26, 18, 23,  2, 20, 12, 16,  5, 27, 9,  21,  2, 20,  3, 25, 13, 22, 18, 23]
    data = [1] * len(row)
    # fermionic phase has already been checked in
    #   ``tests/systems/skeletons/fermionic/test_fermionic_fock_skeleton.py``
    #   so we can just check the absolute value of the Hamiltonian
    assert np.array_equal(np.abs(device.Hs[1]), csr_matrix((data, (row, col)), shape=(28, 28)).toarray())

def test_valley_chain_u_read_only():
    device = ValleyChain(1,
                         1,
                         [0.1],
                         10,
                         20,
                         15,
                         0.1,
                         3,
                         2,
                         1,
                         True)
    try:
        device.u = 2
    except AttributeError:
        pass
    else:
        raise AssertionError("ValleyChain.u should be read-only")

def test_valley_chain_u_valley_flip_read_only():
    device = ValleyChain(1,
                         1,
                         [0.1],
                         10,
                         20,
                         15,
                         0.1,
                         3,
                         2,
                         1,
                         True)
    try:
        device.u_valley_flip = 2
    except AttributeError:
        pass
    else:
        raise AssertionError("ValleyChain.u_valley_flip should be read-only")

def test_valley_chain_valley_spin_orbit_coupling_read_only():
    device = ValleyChain(1,
                         1,
                         [0.1],
                         10,
                         20,
                         15,
                         0.1,
                         3,
                         2,
                         1,
                         True)
    try:
        device.valley_spin_orbit_coupling = 2
    except AttributeError:
        pass
    else:
        raise AssertionError("ValleyChain.valley_spin_orbit_coupling should be read-only")
