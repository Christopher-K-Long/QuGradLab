import numpy as np

from qugradlab.pulses.sampling import get_sample_points, SampleTimes, \
                                      sample_from_piecewise_linear

def test_get_sample_points():
    ts, dt = get_sample_points(1.4, 11)
    assert np.array_equal(ts, np.linspace(0, 1.4, 11))
    assert np.isclose(dt, 0.14)

def test_sample_from_piecewise_linear():
    signal = 2*np.array([0, 1, 2, 3, 4], np.float64)
    signal = np.expand_dims(signal, axis=-1)
    fractional_indices = np.array([0, 0.2, 0.34, 0.5, 1, 3, 2.4, 4])
    samples = sample_from_piecewise_linear(signal, fractional_indices)
    assert np.allclose(samples, np.expand_dims(2*fractional_indices, axis=-1))

def test_sample_times_initialisation():
    T = 10
    dt = 0.1
    number_sample_points = 101
    t = np.linspace(0, T, number_sample_points)
    sample_times = SampleTimes(T=T, dt=dt)
    assert np.array_equal(sample_times.t, t)
    assert sample_times.dt == dt
    assert sample_times.number_sample_points == number_sample_points
    assert sample_times.T == T

    sample_times = SampleTimes(dt=dt, number_sample_points=number_sample_points)
    assert np.array_equal(sample_times.t, t)
    assert sample_times.dt == dt
    assert sample_times.number_sample_points == number_sample_points
    assert sample_times.T == T

    sample_times = SampleTimes(T=T, number_sample_points=number_sample_points)
    assert np.array_equal(sample_times.t, t)
    assert sample_times.dt == dt
    assert sample_times.number_sample_points == number_sample_points
    assert sample_times.T == T

def test_sample_times_t_read_only():
    sample_times = SampleTimes(T=10, dt=0.1)
    
    try:
        sample_times.t = np.array([1, 2, 3])
    except AttributeError:
        pass
    else:
        raise AssertionError("SampleTimes.t should be read-only")
    try:
        sample_times.t[0] = 1
    except ValueError:
        pass
    else:
        raise AssertionError("SampleTimes.t should be read-only")
    
    sample_times.dt = 0.2
    
    try:
        sample_times.t = np.array([1, 2, 3])
    except AttributeError:
        pass
    else:
        raise AssertionError("SampleTimes.t should be read-only")
    try:
        sample_times.t[0] = 1
    except ValueError:
        pass
    else:
        raise AssertionError("SampleTimes.t should be read-only")
    
    sample_times.T = 5

    try:
        sample_times.t = np.array([1, 2, 3])
    except AttributeError:
        pass
    else:
        raise AssertionError("SampleTimes.t should be read-only")
    try:
        sample_times.t[0] = 1
    except ValueError:
        pass
    else:
        raise AssertionError("SampleTimes.t should be read-only")

    sample_times.number_sample_points = 50

    try:
        sample_times.t = np.array([1, 2, 3])
    except AttributeError:
        pass
    else:
        raise AssertionError("SampleTimes.t should be read-only")
    try:
        sample_times.t[0] = 1
    except ValueError:
        pass
    else:
        raise AssertionError("SampleTimes.t should be read-only")

def test_sample_times_setters():
    sample_times = SampleTimes(T=10, dt=0.1)
    sample_times.T = 20
    assert np.array_equal(sample_times.t, np.linspace(0, 20, 101))
    assert sample_times.dt == sample_times.t[1]
    assert sample_times.number_sample_points == 101
    assert sample_times.T == sample_times.t[-1]

    sample_times.dt = 0.5
    assert np.array_equal(sample_times.t, np.linspace(0, 100*0.5, 101))
    assert sample_times.dt == sample_times.t[1]
    assert sample_times.number_sample_points == 101
    assert sample_times.T == sample_times.t[-1]

    sample_times.number_sample_points = 50
    assert np.array_equal(sample_times.t, np.linspace(0, 100*0.5, 50))
    assert sample_times.dt == sample_times.t[1]
    assert sample_times.number_sample_points == 50
    assert sample_times.T == sample_times.t[-1]