import numpy as np
from qugrad import HilbertSpace

from qugradlab.hilbert_spaces.fermionic import FermionSpace, FermionQuditSpace, \
                                               FixedParticleFermionSpace

def test_fermion_space_initialisation():
    assert FermionSpace(4).n_single_particle_states == 4

def test_fermion_space_n_single_particle_states_read_only():
    fermion_space = FermionSpace(4)
    assert fermion_space.n_single_particle_states == 4
    try:
        fermion_space.n_single_particle_states = 5
    except AttributeError:
        pass 
    else:
        raise AssertionError("FermionSpace.n_single_particle_states should be read-only")

def test_fermion_space_labels():
    fermion_space = FermionSpace(3)
    labels = fermion_space.labels()
    assert labels == ['|000⟩', '|100⟩', '|010⟩', '|110⟩',
                      '|001⟩', '|101⟩', '|011⟩', '|111⟩']
    assert fermion_space.labels(0) == '|000⟩'
    assert fermion_space.labels([0, 1, 2]) == ['|000⟩', '|100⟩', '|010⟩']

def test_fixed_particle_fermion_space_initialisation():
    fixed_space = FixedParticleFermionSpace(4, 2)
    assert fixed_space.n_single_particle_states == 4
    assert fixed_space.n_particles == 2
    assert fixed_space.dim == 6 
    labels = fixed_space.labels()
    assert labels == ['|1100⟩', '|1010⟩', '|0110⟩', '|1001⟩', '|0101⟩', '|0011⟩']

def test_fixed_particle_fermion_space_n_particles_read_only():
    fixed_space = FixedParticleFermionSpace(4, 2)
    assert fixed_space.n_particles == 2
    try:
        fixed_space.n_particles = 3
    except AttributeError:
        pass 
    else:
        raise AssertionError("FixedParticleFermionSpace.n_particles should be read-only")

def test_fermion_qudit_space_initialisation():
    fermion_qudit_space = FermionQuditSpace(2, 3, 2)
    assert fermion_qudit_space.n_single_particle_states == 6
    assert fermion_qudit_space.n_particles == 2
    assert fermion_qudit_space.sites == 2
    assert fermion_qudit_space.levels_per_site == 3
    assert fermion_qudit_space.dim == 15

def test_fermion_qudit_space_sites_read_only():
    fermion_qudit_space = FermionQuditSpace(2, 3, 2)
    assert fermion_qudit_space.sites == 2
    try:
        fermion_qudit_space.sites = 3
    except AttributeError:
        pass 
    else:
        raise AssertionError("FermionQuditSpace.sites should be read-only")

def test_fermion_qudit_space_levels_per_site_read_only():
    fermion_qudit_space = FermionQuditSpace(2, 3, 2)
    assert fermion_qudit_space.levels_per_site == 3
    try:
        fermion_qudit_space.levels_per_site = 4
    except AttributeError:
        pass 
    else:
        raise AssertionError("FermionQuditSpace.levels_per_site should be read-only")

def test_fermion_qudit_single_occupation_states():
    fermion_qudit_space = FermionQuditSpace(2, 3, 2)
    single_occupation_states = fermion_qudit_space.single_occupation_states()
    assert all(single_occupation_states == [False, False, False, True,  True,
                                            True,  True,  True,  True,  False,
                                            True,  True,  True,  False, False])

def test_fermion_qudit_space_computational_projector():
    fermion_qudit_space = FermionQuditSpace(2, 3, 2)
    projector = fermion_qudit_space.computational_projector()
    assert all(projector == [False, False, False, True,  True,
                             False, True,  True,  False, False,
                             False, False, False, False, False])

def test_fermion_qudit_space_computational_subspace():
    fermion_qudit_space = FermionQuditSpace(2, 3, 2)
    subspace = fermion_qudit_space.computational_subspace()
    assert subspace == HilbertSpace([int('001001', 2),
                                     int('001010', 2),
                                     int('010001', 2),
                                     int('010010', 2)])

def test_fermion_qudit_space_project_operator():
    np.random.seed(0)
    fermion_qudit_space = FermionQuditSpace(2, 3, 2)
    operator = np.random.rand(15, 15)
    projected_operator = fermion_qudit_space.project_operator(operator)
    expected = operator[fermion_qudit_space.computational_projector()]
    expected = expected[:, fermion_qudit_space.computational_projector()]
    assert np.array_equal(expected, projected_operator)

def test_fermion_qudit_space_dialate_operator():
    np.random.seed(0)
    fermion_space = FermionQuditSpace(2, 3, 2)
    operator = np.random.rand(4, 4)
    dialated_operator = fermion_space.dialate_operator(operator)
    assert np.array_equal(fermion_space.project_operator(dialated_operator),
                          operator)

def test_fermion_qudit_space_n_occupation_states():
    fermion_space = FermionQuditSpace(2, 3, 2)
    assert all(fermion_space.n_occupation_states(0) == [False]*15)
    assert all(fermion_space.n_occupation_states(1) == [False, False, False, True,  True,
                                                        True,  True,  True,  True,  False,
                                                        True,  True,  True,  False, False])
    assert all(fermion_space.n_occupation_states(2) == [True,  True,  True,  False, False,
                                                        False, False, False, False, True,
                                                        False, False, False, True,  True])
    for n in range(3, 16):
        assert all(fermion_space.n_occupation_states(n) == [False]*15)