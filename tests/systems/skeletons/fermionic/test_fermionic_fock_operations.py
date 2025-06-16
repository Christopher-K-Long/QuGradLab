import numpy as np
import qugradlab.systems.skeletons.fermionic._fermionic_fock_operations as fo

TEST_BITS = 4
fock_states = np.arange(2**TEST_BITS)

def test_is_empty_correct_parity():
    assert fo.is_empty(np.array([0]), 0)
    
def test_is_occupied_correct_parity():
    assert fo.is_occupied(np.array([1]), 0)
    
def test_is_empty_is_occupied_compliment():
    assert all([np.all(np.not_equal(fo.is_empty(fock_states, i), fo.is_occupied(fock_states, i))) for i in range(TEST_BITS)])

def test_is_empty():
    # Only need to test `is_empty` as all ready checked `is_occupied` is the compliment
    assert all([int(format(fs, 'b').zfill(TEST_BITS)[-(i+1)]) != fo.is_empty(fs, i) for fs in fock_states for i in range(TEST_BITS)])

def test_fill_orbitals():
    # Using `is_empty` and `is_occupied` as these have been tested above
    results = []
    for j in range(TEST_BITS):
        for i in range(j):
            fock_state = fo.fill_orbitals(i,j)
            for k in range(i):
                results.append(fo.is_empty(fock_state, k))
            for k in range(i, j):
                results.append(fo.is_occupied(fock_state, k))
            for k in range(j, TEST_BITS):
                results.append(fo.is_empty(fock_state, k))
    assert all(results)
    
def test_select_orbitals():
    results = []
    for j in range(TEST_BITS):
        for i in range(j):
            selected = fo.select_orbitals(fock_states, i, j)
            for k in range(i):
                results += list(fo.is_empty(selected, k))
            fs_and_s = ~(fock_states ^ selected) # XNOR (bitwise equivilence)
            for k in range(i, j):
                assert list(fo.is_occupied(fs_and_s, k))
            for k in range(j, TEST_BITS):
                results += list(fo.is_empty(selected, k))
    assert all(results)
    
def test_flip_single():
    results = []
    for i in range(TEST_BITS):
        changes = ~(fock_states ^ fo.flips(fock_states, [i])) # XNOR (bitwise equivilence)
        for j in range(i):
            results += list(fo.is_occupied(changes, j))
        results += list(fo.is_empty(changes, i))
        for j in range(i+1, TEST_BITS):
            results += list(fo.is_occupied(changes, j))
    assert all(results)
    
def test_flip_pair():
    results = []
    for i in range(TEST_BITS):
        for j in range(TEST_BITS):
            if i == j:
                continue
            changes = ~(fock_states ^ fo.flips(fock_states, [i, j])) # XNOR (bitwise equivilence)
            for k in range(TEST_BITS):
                if k == i or k == j:
                    results += list(fo.is_empty(changes, k))
                else:
                    results += list(fo.is_occupied(changes, k))  
    assert all(results)
        