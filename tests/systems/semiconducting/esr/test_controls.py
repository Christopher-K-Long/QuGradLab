import numpy as np

from qugradlab.systems.semiconducting.esr._controls import Controls

def test_controls_initialisation():
    zeeman_splittings = np.array([1, 2, 3])
    max_drive_strength = 1.0
    J_min = 0.1
    J_max = 10.0

    controls = Controls(zeeman_splittings, max_drive_strength, J_min, J_max)

    assert np.array_equal(controls.zeeman_splittings, zeeman_splittings)
    assert controls.max_drive_strength == max_drive_strength
    assert controls.J_min == J_min
    assert controls.J_max == J_max

def test_pre_processing():
    zeeman_splittings = np.array([1, 2])
    max_drive_strength = 10
    J_min = 0
    J_max = 10.0

    controls = Controls(zeeman_splittings, max_drive_strength, J_min, J_max)

    drive_ctrl_amp = np.array([[0.5, -0.5], [1, 0.54], [-1, 0.2]])
    J_ctrl_amp = np.array([[1], [0.3], [0]])

    initial_state = np.array([0, 1, 0, 0])

    dt = 0.1

    output = controls._pre_processing(drive_ctrl_amp,
                                      zeeman_splittings,
                                      J_ctrl_amp,
                                      initial_state,
                                      dt)
    
    assert np.array_equal(output[0], np.concatenate([10*drive_ctrl_amp, 5*(J_ctrl_amp+1)], axis=-1))
    assert np.array_equal(output[1], initial_state)
    assert output[2] == dt
    assert np.array_equal(output[3], np.concatenate([zeeman_splittings, [0]*J_ctrl_amp.shape[-1]]))
    assert isinstance(output[4], list)
    assert np.array_equal(output[4], [drive_ctrl_amp.shape[-1]] + [1]*J_ctrl_amp.shape[-1])