from qugradlab.pulses.invertible_functions import InvertibleFunction

def test_invertible_function_initialisation():
    def sample_function(x):
        return x + 1

    invertible_function = InvertibleFunction(sample_function)
    
    assert invertible_function._func == sample_function
    assert invertible_function._inverse is None
    try:
        invertible_function.inverse
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Expected NotImplementedError when accessing inverse before setting it.")

def test_invertible_function_set_inverse():
    def sample_function(x):
        return x + 1

    def inverse_function(x):
        return x - 1

    invertible_function = InvertibleFunction(sample_function)
    invertible_function.set_inverse(inverse_function)

    assert invertible_function.inverse._func == inverse_function
    assert invertible_function.inverse.inverse == invertible_function
    assert invertible_function(invertible_function.inverse(5)) == 5
    assert invertible_function.inverse(invertible_function(5)) == 5

def test_specify_parameters():
    def sample_function(x, a):
        return x + a

    invertible_function = InvertibleFunction(sample_function)
    
    def inverse_function(x, a):
        return x - a

    invertible_function.set_inverse(inverse_function)

    specified_invertible_function = invertible_function.specify_parameters(a=2)

    assert specified_invertible_function(3) == 5
    assert specified_invertible_function.inverse(5) == 3


def test_compose():
    def inner_function(x):
        return x * 2

    def outer_function(x):
        return x + 3

    inner_invertible_function = InvertibleFunction(inner_function)
    outer_invertible_function = InvertibleFunction(outer_function)

    composed_function = \
        outer_invertible_function.compose(inner_invertible_function)

    assert composed_function(5) == 13
    assert composed_function._inverse is None
    try:
        composed_function.inverse
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Expected NotImplementedError when accessing inverse before setting it.")

    inner_invertible_function.set_inverse(lambda x: x / 2)
    outer_invertible_function.set_inverse(lambda x: x - 3)

    composed_function = \
        outer_invertible_function.compose(inner_invertible_function)

    assert composed_function(5) == 13  # (5 * 2) + 3
    assert composed_function.inverse(13) == 5  # (13 - 3) / 2