import numpy as np

from qugradlab.pulses.invertible_functions.scaling import  linear_rescaling

def test_linear_rescaling():
    np.random.seed(0)
    x = 2*np.random.rand(3, 5) - 1 # Random values in [-1, 1]
    min_x = -0.5
    max_x = 1.2
    assert np.allclose(linear_rescaling(x, min_x, max_x), 0.5*(x+1) * (max_x-min_x) + min_x)
    assert np.allclose(linear_rescaling.inverse(linear_rescaling(x, min_x, max_x), min_x, max_x), x)

    scaling = linear_rescaling.specify_parameters(min=min_x, max=max_x)
    assert np.allclose(scaling(x), 0.5*(x+1) * (max_x-min_x) + min_x)
    assert np.allclose(scaling.inverse(scaling(x)), x)