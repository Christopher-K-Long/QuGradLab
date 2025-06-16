import numpy as np

from qugradlab.systems.skeletons.qubits import QubitSystem, \
                                               qubit_skeleton
from qugradlab.hilbert_spaces import QubitSpace
from qugradlab.systems.skeletons._skeletal_system import contract_skeletons, \
                                                         get_Hs

QUBITS = 4
NCTRL = 3

def test_fermionic_system_initialisation():
    hilbert_space = QubitSpace(QUBITS)
    single_qubit_drift_coefficients = np.random.rand(QUBITS, 3)
    two_qubit_drift_coefficients = np.random.rand(QUBITS, QUBITS, 3, 3)
    single_qubit_ctrl_coefficients = np.random.rand(NCTRL, QUBITS, 3)
    two_qubit_ctrl_coefficients = np.random.rand(NCTRL, QUBITS, QUBITS, 3, 3)
    use_graph = True
    system = QubitSystem(hilbert_space,
                         single_qubit_drift_coefficients,
                         two_qubit_drift_coefficients,
                         single_qubit_ctrl_coefficients,
                         two_qubit_ctrl_coefficients,
                         use_graph)
    t = qubit_skeleton.second_order_tensor(hilbert_space)
    u = qubit_skeleton.fourth_order_tensor(hilbert_space)
    H0 = contract_skeletons([single_qubit_drift_coefficients,
                             two_qubit_drift_coefficients],
                            [t, u])
    Hs = get_Hs([single_qubit_ctrl_coefficients,two_qubit_ctrl_coefficients],
                [t, u])
    
    assert np.array_equal(system.H0, H0)
    assert np.array_equal(system.Hs, Hs)
    assert system.hilbert_space == hilbert_space
    assert system.using_graph == use_graph

    use_graph = False
    system = QubitSystem(hilbert_space,
                         single_qubit_drift_coefficients,
                         two_qubit_drift_coefficients,
                         single_qubit_ctrl_coefficients,
                         two_qubit_ctrl_coefficients,
                         use_graph)
    assert system.using_graph == use_graph