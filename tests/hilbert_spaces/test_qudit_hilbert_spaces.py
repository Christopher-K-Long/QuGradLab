import numpy as np

from qugradlab.hilbert_spaces import QuditSpace, QubitSpace

def test_qudit_space():
    try:
        QuditSpace([0, 1])
    except TypeError:
        pass
    else:
        raise AssertionError("QuditSpace should raise TypeError as it is an abstract class.")
    QuditSpace.computational_projector
    QuditSpace.dialate_operator
    QuditSpace.project_operator
    QuditSpace.computational_subspace

def test_qubit_space_initialisation():
    qubit_space = QubitSpace(2)
    assert qubit_space.qubits == 2

def test_qubit_space_qubits_read_only():
    qubit_space = QubitSpace(3)
    try:
        qubit_space.qubits = 4
    except AttributeError:
        pass
    else:
        raise AssertionError("QubitSpace.qubits should be read-only.")

def test_qubit_space_computational_projector():
    qubit_space = QubitSpace(2)
    assert np.array_equal(qubit_space.computational_projector(),
                          np.ones((4,), dtype=bool))

def test_qubit_space_computation_subspace():
    assert QubitSpace(2).computational_subspace() == \
           QubitSpace(2).get_subspace([True]*4)

def test_qubit_space_project_operator():
    np.random.seed(0)
    qudit_space = QubitSpace(2)
    operator = np.random.rand(4, 4)
    projected_operator = qudit_space.project_operator(operator)
    assert np.array_equal(projected_operator, operator)

def test_qubit_space_dialate_operator():
    np.random.seed(0)
    qudit_space = QubitSpace(2)
    operator = np.random.rand(4, 4)
    dialateed_operator = qudit_space.dialate_operator(operator)
    assert np.array_equal(dialateed_operator, operator)

def test_qubit_space_labels():
    qudit_space = QubitSpace(3)
    labels = qudit_space.labels()
    assert labels == ['|000⟩', '|100⟩', '|010⟩', '|110⟩',
                      '|001⟩', '|101⟩', '|011⟩', '|111⟩']
    assert qudit_space.labels(0) == '|000⟩'
    assert qudit_space.labels([0, 1, 2]) == ['|000⟩', '|100⟩', '|010⟩']