import numpy as np

from qugrad import HilbertSpace

from qugradlab.systems.skeletons import _skeletal_system

def test_partial_flatten():
    array = np.array([[[1], [2]], [[3], [4]]])
    assert np.array_equal(_skeletal_system.partial_flatten(array, 0), [array])
    assert np.array_equal(_skeletal_system.partial_flatten(array, 1), array)
    assert np.array_equal(_skeletal_system.partial_flatten(array, 2), np.array([[1], [2], [3], [4]]))
    assert np.array_equal(_skeletal_system.partial_flatten(array, 3), np.array([1, 2, 3, 4]))
    assert np.array_equal(_skeletal_system.partial_flatten(array, 4), np.array([1, 2, 3, 4]))
    assert np.array_equal(_skeletal_system.partial_flatten(array, -1), array)
    assert np.array_equal(_skeletal_system.partial_flatten(array, -2), np.array([[1, 2], [3, 4]]))
    assert np.array_equal(_skeletal_system.partial_flatten(array, -3), np.array([1, 2, 3, 4]))
    assert np.array_equal(_skeletal_system.partial_flatten(array, -4), np.array([1, 2, 3, 4]))

def test_partial_flatten_empty():
    array = np.empty((10, 5, 0, 1))
    assert np.array_equal(_skeletal_system.partial_flatten(array, 0), [array])
    assert np.array_equal(_skeletal_system.partial_flatten(array, 1), array)
    assert np.array_equal(_skeletal_system.partial_flatten(array, 2), np.empty((50, 0, 1)))
    assert np.array_equal(_skeletal_system.partial_flatten(array, 3), np.empty((0, 1)))
    assert np.array_equal(_skeletal_system.partial_flatten(array, 4), np.empty((0,)))
    assert np.array_equal(_skeletal_system.partial_flatten(array, -1), array)
    assert np.array_equal(_skeletal_system.partial_flatten(array, -2), np.empty((10, 5, 0)))
    assert np.array_equal(_skeletal_system.partial_flatten(array, -3), np.empty((10, 0)))
    assert np.array_equal(_skeletal_system.partial_flatten(array, -4), np.empty((0,)))

def test_contract_general():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    result = _skeletal_system.contract_general(a, b, 0)
    expected_result = np.einsum("ij,ij->", a, b)
    assert np.array_equal(result, expected_result)
    
    result = _skeletal_system.contract_general(a, b, 1)
    expected_result = np.array([[19, 22], [43, 50]])
    assert np.array_equal(result, expected_result)

    result = _skeletal_system.contract_general(a, b, 2)
    expected_result = np.einsum("ij,kl->ijkl", a, b)
    assert np.array_equal(result, expected_result)

    a = np.array([[[1], [2]], [[3], [4]]])
    b = np.array([[[5], [6]], [[7], [8]]])

    result = _skeletal_system.contract_general(a, b, 0)
    expected_result = np.einsum("ijk,ijk->", a, b)
    assert np.array_equal(result, expected_result)

    a = np.array([[[1], [2]], [[3], [4]]])
    b = np.array([[[5, 6]], [[7, 8]]])
    
    result = _skeletal_system.contract_general(a, b, 1)
    expected_result = np.einsum("...ij,ijk->...k", a, b)
    assert np.array_equal(result, expected_result)

    a = np.array([[[1], [2]], [[3], [4]]])
    b = np.array([[[5, 6], [7, 8]]])

    result = _skeletal_system.contract_general(a, b, 2)
    expected_result = np.einsum("...i,ijk->...jk", a, b)
    assert np.array_equal(result, expected_result)

    result = _skeletal_system.contract_general(a, b, 3)
    expected_result = np.einsum("...,ijk->...ijk", a, b)
    assert np.array_equal(result, expected_result)

def test_contract_skeleton():
    coefficients = np.array([1, 2, 3])
    skeleton = np.array([[[4, 5],
                          [6, 7]],
                         [[8, 9],
                          [10, 11]],
                         [[12, 13],
                          [14, 15]]])

    result = _skeletal_system.contract_skeleton(coefficients, skeleton)
    expected_result = _skeletal_system.contract_general(coefficients,
                                                        skeleton,
                                                        2)
    assert np.array_equal(result, expected_result)

def test_contract_skeletons():
    skeleton_1 = np.array([[[[[0, 1],
                              [2, 3]],
                             [[4, 5],
                              [6, 7]]],
                            [[[8, 9],
                              [10, 11]],
                             [[12, 13],
                              [14, 15]]]]])
    skeleton_2 = np.array([[[4, 5],
                            [6, 7]],
                           [[8, 9],
                            [10, 11]]])
    skeletons = [skeleton_1, skeleton_2]
    coefficients_1 = np.array([[[[1, 2],
                                 [3, 4]]]])
    coefficients_2 = np.array([[5, 6]])
    coefficients = [coefficients_1, coefficients_2]
    result = _skeletal_system.contract_skeletons(coefficients, skeletons)
    expected_result = _skeletal_system.contract_skeleton(coefficients_1,
                                                         skeleton_1) \
                     + _skeletal_system.contract_skeleton(coefficients_2,
                                                          skeleton_2)
    assert np.array_equal(result, expected_result)

def test_flatten_skeleton():
    skeleton = np.array([[[[[0, 1],
                            [2, 3]],
                           [[4, 5],
                            [6, 7]]],
                          [[[8, 9],
                            [10, 11]],
                           [[12, 13],
                            [14, 15]]]]])
    result = _skeletal_system.flatten_skeleton(skeleton)
    expected_result = _skeletal_system.partial_flatten(skeleton, 3)
    assert np.array_equal(result, expected_result)

def test_get_Hs():
    skeleton_1 = np.array([[[[[0, 1],
                              [2, 3]],
                             [[4, 5],
                              [6, 7]]],
                            [[[8, 9],
                              [10, 11]],
                             [[12, 13],
                              [14, 15]]]]])
    skeleton_2 = np.array([[[4, 5],
                            [6, 7]],
                           [[8, 9],
                            [10, 11]]])
    skeletons = [skeleton_1, skeleton_2]
    coefficients_1 = np.array([[[[1, 2],
                                 [3, 4]]],
                               [[[1, 2],
                                 [3, 4]]]])
    coefficients_2 = np.array([[5, 6]])
    coefficients = [coefficients_1, coefficients_2]
    result = _skeletal_system.get_Hs(coefficients, skeletons)

    contraction_1 = _skeletal_system.contract_skeleton(coefficients_1,
                                                       skeleton_1)
    contraction_2 = _skeletal_system.contract_skeleton(coefficients_2,
                                                       skeleton_2)
    expected_result = np.concatenate([contraction_1,
                                      _skeletal_system.flatten_skeleton(contraction_2)])
    assert np.array_equal(result, expected_result)

def test_initialise_skeletal_system():
    skeleton_1 = np.array([[[[[0, 1],
                              [2, 3]],
                             [[4, 5],
                              [6, 7]]],
                            [[[8, 9],
                              [10, 11]],
                             [[12, 13],
                              [14, 15]]]]])
    skeleton_2 = np.array([[[4, 5],
                            [6, 7]],
                           [[8, 9],
                            [10, 11]]])
    drift_skeletons = [skeleton_1, skeleton_2]
    coefficients_1 = np.array([[[[1, 2],
                                 [3, 4]]]])
    coefficients_2 = np.array([[5, 6]])
    drift_coefficients = [coefficients_1, coefficients_2]

    skeleton_1 = np.array([[[[[0, 1],
                              [2, 3]],
                             [[4, 5],
                              [6, 7]]],
                            [[[8, 9],
                              [10, 11]],
                             [[12, 13],
                              [14, 15]]]]])
    skeleton_2 = np.array([[[4, 5],
                            [6, 7]],
                           [[8, 9],
                            [10, 11]]])
    ctrl_skeletons = [skeleton_1, skeleton_2]
    coefficients_1 = np.array([[[[1, 2],
                                 [3, 4]]],
                               [[[1, 2],
                                 [3, 4]]]])
    coefficients_2 = np.array([[5, 6]])
    ctrl_coefficients = [coefficients_1, coefficients_2]

    hilbert_space = HilbertSpace([0, 1])
    use_graph = True

    skeletal_system = _skeletal_system.SkeletalSystem(drift_coefficients,
                                                      drift_skeletons,
                                                      ctrl_coefficients,
                                                      ctrl_skeletons,
                                                      hilbert_space,
                                                      use_graph)
    assert np.array_equal(skeletal_system.H0, _skeletal_system.contract_skeletons(drift_coefficients,
                                                                                  drift_skeletons))
    assert np.array_equal(skeletal_system.Hs, _skeletal_system.get_Hs(ctrl_coefficients,
                                                                      ctrl_skeletons))
    assert skeletal_system.hilbert_space == hilbert_space
    assert skeletal_system.using_graph == use_graph

    use_graph = False

    skeletal_system = _skeletal_system.SkeletalSystem(drift_coefficients,
                                                      drift_skeletons,
                                                      ctrl_coefficients,
                                                      ctrl_skeletons,
                                                      hilbert_space,
                                                      use_graph)

    assert skeletal_system.using_graph == use_graph