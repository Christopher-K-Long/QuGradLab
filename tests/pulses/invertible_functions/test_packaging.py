import numpy as np

from qugradlab.pulses.invertible_functions.packaging import pack, unpack, \
                                                            package_complex, \
                                                            unpackage_complex, \
                                                            ViewPop

def test_view_pop():
    x = [1, 2, 3, 4, 5, 6]
    v = ViewPop(x)
    assert np.array_equal(v(2), [1, 2])
    assert np.array_equal(v(2), [3, 4])
    assert np.array_equal(v(1), [5])
    assert np.array_equal(v(2), [6])
    assert np.array_equal(v(3), [])

def test_pack():
    x = [np.array([1, 2, 3]),
         np.array([[1, 0], [0, 1]]),
         np.array([5, 4])]
    packed = pack(x)
    assert np.array_equal(packed, np.array([1, 2, 3, 1, 0, 0, 1, 5, 4]))    
    assert pack.inverse == unpack
    assert len(unpack(packed, [[3], [2, 2], [2]])) == len(x)
    assert all([np.array_equal(unpack(packed, [[3], [2, 2], [2]])[i], x[i]) for i in range(len(x))])
    packer = pack.specify_parameters(shapes=[[3], [2, 2], [2]])
    assert all([np.array_equal(packer.inverse(packer(x))[i], x[i]) for i in range(len(x))])
    unpacker = unpack.specify_parameters(shapes=[[3], [2, 2], [2]])
    assert all([np.array_equal(unpacker(unpacker.inverse(x))[i], x[i]) for i in range(len(x))])

def test_package_complex():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    packed = package_complex(x)
    assert np.array_equal(packed, np.array([1 + 2j, 3 + 4j, 5 + 6j]))
    assert package_complex.inverse == unpackage_complex
    assert np.array_equal(unpackage_complex(packed), x)