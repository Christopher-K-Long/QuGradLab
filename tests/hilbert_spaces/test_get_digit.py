import numpy as np

from qugradlab.hilbert_spaces._get_digit import get_digit

def test_get_digit():
    assert get_digit(4132, 10, 0) == 2
    assert get_digit(4132, 10, 1) == 3
    assert get_digit(4132, 10, 2) == 1
    assert get_digit(4132, 10, 3) == 4

    assert get_digit(int('1001001', 2), 2, 0) == 1
    assert get_digit(int('1001001', 2), 2, 1) == 0
    assert get_digit(int('1001001', 2), 2, 2) == 0
    assert get_digit(int('1001001', 2), 2, 3) == 1
    assert get_digit(int('1001001', 2), 2, 4) == 0
    assert get_digit(int('1001001', 2), 2, 5) == 0
    assert get_digit(int('1001001', 2), 2, 6) == 1

    assert get_digit(int('1021201', 3), 3, 0) == 1
    assert get_digit(int('1021201', 3), 3, 1) == 0
    assert get_digit(int('1021201', 3), 3, 2) == 2
    assert get_digit(int('1021201', 3), 3, 3) == 1
    assert get_digit(int('1021201', 3), 3, 4) == 2
    assert get_digit(int('1021201', 3), 3, 5) == 0
    assert get_digit(int('1021201', 3), 3, 6) == 1


    digits = get_digit(np.array([12, 3]), np.array([10, 2]), np.array([0, 1]))
    assert np.array_equal(digits, np.array([[[2, 1], [0, 0]],
                                            [[3, 0], [1, 1]]]))