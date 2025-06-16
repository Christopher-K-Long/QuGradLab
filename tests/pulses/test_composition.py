import numpy as np
import tensorflow as tf

from qugradlab.pulses.composition import pad, concatenate_functions

def test_pad():
    x = tf.constant([[1, 2], [3, 4]])
    assert np.all(pad(x) == tf.constant([[0, 0], [1, 2], [3, 4], [0, 0]]))
    assert np.all(pad(x, 1) == tf.constant([[1, 1], [1, 2], [3, 4], [1, 1]]))
    assert np.all(pad(x, left=False) == tf.constant([[1, 2], [3, 4], [0, 0]]))
    assert np.all(pad(x, right=False) == tf.constant([[0, 0], [1, 2], [3, 4]]))
    assert np.all(pad(x, left=False, right=False) == tf.constant([[1, 2], [3, 4]]))

def test_concatenate_functions():
    def f1(x, y):
        return [x + y]

    def f2(x, y):
        return [x - y]

    concatenated_function = concatenate_functions([f1, f2])
    result = concatenated_function([[2, 1], [3, 2]])
    assert np.array_equal(result, [3, 1])