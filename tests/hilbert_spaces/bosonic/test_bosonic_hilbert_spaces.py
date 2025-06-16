import numpy as np

from qugrad import HilbertSpace

from qugradlab.hilbert_spaces.bosonic import BosonSpace, \
                                             BosonQuditSpace

def test_bosonic_space_initialisation():
    boson_space = BosonSpace(3, 2)
    assert boson_space.n_single_particle_states == 3
    assert boson_space.truncation_level == 2

def test_bosonic_space_n_single_particle_states_read_only():
    boson_space = BosonSpace(3, 2)
    try:
        boson_space.n_single_particle_states = 4
    except AttributeError:
        pass
    else:
        raise AssertionError("BosonSpace.n_single_particle_states should be read-only")

def test_bosonic_space_truncation_level():
    boson_space = BosonSpace(3, 2)
    try:
        boson_space.truncation_level = 4
    except AttributeError:
        pass
    else:
        raise AssertionError("BosonSpace.truncation_level should be read-only")

def test_bosonic_space_labels():
    boson_space = BosonSpace(2, 3)
    labels = boson_space.labels()
    assert labels == ['|0, 0⟩', '|1, 0⟩', '|2, 0⟩',
                      '|0, 1⟩', '|1, 1⟩', '|2, 1⟩',
                      '|0, 2⟩', '|1, 2⟩', '|2, 2⟩']
    assert boson_space.labels(0) == '|0, 0⟩'
    assert boson_space.labels([0, 1, 2]) == ['|0, 0⟩', '|1, 0⟩', '|2, 0⟩']

def test_boson_qudit_space_computational_projector():
    boson_space = BosonQuditSpace(2, 3)
    assert all(boson_space.computational_projector() == [True,  True,  False,
                                                         True,  True,  False,
                                                         False, False, False])

def test_boson_qudit_space_computational_subspace():
    boson_space = BosonQuditSpace(2, 3)
    subspace = boson_space.computational_subspace()
    assert subspace == HilbertSpace([0, 1, 3, 4])

def test_boson_qudit_space_project_operator():
    np.random.seed(0)
    boson_space = BosonQuditSpace(2, 3)
    operator = np.random.rand(9, 9)
    projected_operator = boson_space.project_operator(operator)
    expected = operator[boson_space.computational_projector()]
    expected = expected[:, boson_space.computational_projector()]
    assert np.array_equal(expected,
                          projected_operator)

def test_boson_qudit_space_dialate_operator():
    np.random.seed(0)
    boson_space = BosonQuditSpace(2, 3)
    operator = np.random.rand(4, 4)
    dialated_operator = boson_space.dialate_operator(operator)
    assert dialated_operator.shape == (9, 9)
    assert np.array_equal(boson_space.project_operator(dialated_operator),
                          operator)