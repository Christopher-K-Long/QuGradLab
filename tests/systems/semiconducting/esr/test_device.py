import numpy as np

from qugradlab.systems.semiconducting.esr._device import Device

def test_device_initialisation():
    zeeman_splittings = np.array([1, 2, 3])
    max_drive_strength = 1.0
    J_min = 0.1
    J_max = 10.0

    device = Device(zeeman_splittings, max_drive_strength, J_min, J_max)

    assert np.array_equal(device.zeeman_splittings, zeeman_splittings)
    assert device.max_drive_strength == max_drive_strength
    assert device.J_min == J_min
    assert device.J_max == J_max

def test_zeeman_splittings_read_only():
    zeeman_splittings = np.array([1, 2, 3])
    max_drive_strength = 1.0
    J_min = 0.1
    J_max = 10.0

    device = Device(zeeman_splittings, max_drive_strength, J_min, J_max)

    try:
        device.zeeman_splittings = np.array([0])
    except AttributeError:
        pass
    else:
        raise AssertionError("Device.zeeman_splittings should be read-only")
    
    try:
        device.zeeman_splittings[0] = 0
    except ValueError:
        pass
    else:
        raise AssertionError("Device.zeeman_splittings should be read-only")

def test_max_drive_strength_read_only():
    zeeman_splittings = np.array([1, 2, 3])
    max_drive_strength = 1.0
    J_min = 0.1
    J_max = 10.0

    device = Device(zeeman_splittings, max_drive_strength, J_min, J_max)

    try:
        device.max_drive_strength = 0
    except AttributeError:
        pass
    else:
        raise AssertionError("Device.max_drive_strength should be read-only")

def test_max_J_min_read_only():
    zeeman_splittings = np.array([1, 2, 3])
    max_drive_strength = 1.0
    J_min = 0.1
    J_max = 10.0

    device = Device(zeeman_splittings, max_drive_strength, J_min, J_max)

    try:
        device.J_min = 0
    except AttributeError:
        pass
    else:
        raise AssertionError("Device.J_min should be read-only")

def test_max_J_max_read_only():
    zeeman_splittings = np.array([1, 2, 3])
    max_drive_strength = 1.0
    J_min = 0.1
    J_max = 10.0

    device = Device(zeeman_splittings, max_drive_strength, J_min, J_max)

    try:
        device.J_max = 0
    except AttributeError:
        pass
    else:
        raise AssertionError("Device.J_max should be read-only")