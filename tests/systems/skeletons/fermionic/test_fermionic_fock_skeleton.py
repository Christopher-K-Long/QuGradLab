import numpy as np
import qugradlab.systems.skeletons.fermionic.fermionic_fock_skeleton as fs
import qugradlab.hilbert_spaces.fermionic._hamming_weight_operations as hw
from qugradlab.hilbert_spaces.fermionic import FermionSpace, \
                                               FixedParticleFermionSpace
from qugradlab.systems.skeletons.fermionic._fermionic_fock_operations import \
    is_empty, is_occupied, select_orbitals, flips

SITES = 4

def second_order_tensor_check(fock_states):
    t = fs.second_order_tensor(fock_states)
    assert t.shape == (SITES, SITES, len(fock_states), len(fock_states))
    for x, el in np.ndenumerate(t):
        if x[0] == x[1]: # on-site
            assert el == (x[2] == x[3] and is_occupied(fock_states[x[2]], x[0]))
        else: # hopping
            y = is_occupied(fock_states[x[3]], x[1])\
                and is_empty(fock_states[x[3]], x[0])\
                and is_empty(fock_states[x[2]], x[1])\
                and is_occupied(fock_states[x[2]], x[0])\
                and fock_states[x[2]] == flips(fock_states[x[3]], [x[0], x[1]])
            lower = min(x[0], x[1])
            upper = max(x[0], x[1])
            state = fock_states[x[3]] if lower == x[0] else fock_states[x[2]]
            phase = (-1)**hw.hamming_weight(select_orbitals(state, lower, upper))
            assert el == y*phase

def create_or_annihilate(state, phase, orbital):
        phase *= (-1)**hw.hamming_weight(select_orbitals(state, 0, orbital))
        state = flips(state, [orbital])
        return state, phase

def fourth_order_tensor_check(fock_states):
    t = fs.fourth_order_tensor(fock_states)
    assert t.shape == (SITES, SITES, SITES, SITES, len(fock_states), len(fock_states))
    for x, el in np.ndenumerate(t):
        if x[0] == x[1] or x[2] == x[3]:
            assert el == 0
        else:
            y = 0
            if is_occupied(fock_states[x[5]], x[2]) and is_occupied(fock_states[x[5]], x[3]):
                phase = 1
                state, phase = create_or_annihilate(fock_states[x[5]], phase, x[3])
                state, phase = create_or_annihilate(state, phase, x[2])
                if is_empty(state, x[0]) and is_empty(state, x[1]):
                    state, phase = create_or_annihilate(state, phase, x[1])
                    state, phase = create_or_annihilate(state, phase, x[0])
                    if state == fock_states[x[4]]:
                        y = phase
            assert el == y

def test_second_order_tensor_fixed_particle_number():
    for particle_number in range(SITES+1):
        fock_states = FixedParticleFermionSpace(SITES, particle_number)
        second_order_tensor_check(fock_states)

def test_fourth_order_tensor_fixed_particle_number():
    for particle_number in range(SITES+1):
        fock_states = FixedParticleFermionSpace(SITES, particle_number)
        fourth_order_tensor_check(fock_states)
        
def test_second_order_tensor():
    fock_states = FermionSpace(SITES)
    second_order_tensor_check(fock_states)
                     
def test_fourth_order_tensor():
    fock_states = FermionSpace(SITES)
    fourth_order_tensor_check(fock_states)