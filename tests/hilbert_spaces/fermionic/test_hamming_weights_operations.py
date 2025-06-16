import numpy as np
import qugradlab.hilbert_spaces.fermionic._hamming_weight_operations as hw

TEST_BITS = 4
fock_states = np.arange(2**TEST_BITS)
hamming_weights = np.array([format(fs, 'b').zfill(TEST_BITS).count('1') for fs in fock_states])

def test_hamming_weight():
    assert (hw.hamming_weight(fock_states) == hamming_weights).all()
def test_next_constant_hamming_weight():
    next_fock_states = np.zeros(len(fock_states)-1)
    for i, fs in enumerate(fock_states[1:]):
        j = fs+1
        while hw.hamming_weight(j) != hamming_weights[i+1]:
            j += 1
        next_fock_states[i] = j
    assert (next_fock_states == hw.next_constant_hamming_weight(fock_states[1:])).all()
def test_initial_constant_hamming_weight():
    initials = []
    current = 0
    for i, weight in enumerate(hamming_weights):
        if weight == current:
            initials.append(i)
            current += 1
    initials = np.array(initials)
    initial_hamming_weights = hw.initial_constant_hamming_weight(np.arange(len(initials)))
    assert (initials == initial_hamming_weights).all()
def test_all_constant_hamming_weight():
    for weight in range(TEST_BITS+2): 
        current = hw.initial_constant_hamming_weight(weight)
        for state in hw.all_constant_hamming_weight(TEST_BITS, weight):
            assert current == state
            current = 2**TEST_BITS + 1 if current == 0 else hw.next_constant_hamming_weight(current)
        assert current >= 2**TEST_BITS

def test_sub_hamming_weight_is():
    x = np.add.outer(fock_states<<TEST_BITS, fock_states)
    for hamming_weights in range(TEST_BITS+1):
        result = hw.sub_hamming_weight_is(hamming_weights, TEST_BITS, 2, x)
        correct_weight = hw.hamming_weight(fock_states) == hamming_weights
        assert np.array_equal(result,
                              np.logical_and.outer(correct_weight,
                                                   correct_weight))

def test_any_sub_hamming_weight_is():
    x = np.add.outer(fock_states<<TEST_BITS, fock_states)
    for hamming_weights in range(TEST_BITS+1):
        result = hw.any_sub_hamming_weight_is(hamming_weights, TEST_BITS, 2, x)
        correct_weight = hw.hamming_weight(fock_states) == hamming_weights
        assert np.array_equal(result,
                              np.logical_or.outer(correct_weight,
                                                   correct_weight))

def test_only_excited_states_occupied():
    x = np.add.outer(fock_states<<TEST_BITS, fock_states).flatten()
    x = np.add.outer(x<<(2*TEST_BITS), x)
    result = hw.only_excited_states_occupied(TEST_BITS, 2*TEST_BITS, 2, x)
    expected = np.logical_and.outer([True]*len(fock_states), [True]+[False]*(len(fock_states)-1)).flatten()
    expected = np.logical_and.outer(expected, expected)
    assert np.array_equal(result, expected)

def test_only_low_lieing_states_occupied():
    x = np.add.outer(fock_states<<TEST_BITS, fock_states).flatten()
    x = np.add.outer(x<<(2*TEST_BITS), x)
    result = hw.only_low_lieing_states_occupied(TEST_BITS, 2*TEST_BITS, 2, x)
    expected = np.logical_and.outer([True]+[False]*(len(fock_states)-1), [True]*len(fock_states)).flatten()
    expected = np.logical_and.outer(expected, expected)
    assert np.array_equal(result, expected)