"""
A base classes for quantum systems constructed from a contraction of
coefficients with a set of operators.
"""

import numpy as np
from qugrad import QuantumSystem, HilbertSpace

def partial_flatten(array: np.ndarray, n: int) -> np.ndarray:
    """
    Flattens the first `n` dimensions of an `array`. If `n` is negative, it
    flattens the last ``-n`` dimensions of the array.

    Parameters
    ----------
    array : np.ndarray
        The array to be partially flattened.
    n : int
        The number of dimensions to flatten. If `n` is negative, it flattens
        the last ``-n`` dimensions of the ``array``.

    Returns
    -------
    np.ndarray
        The partially flattened array.
    """
    if array.size == 0:
        if n >= 0:
            return np.empty((np.prod(array.shape[:n], dtype=int),)+array.shape[n:], dtype=array.dtype)
        return np.empty((array.shape[:n])+(np.prod(array.shape[n:], dtype=int),), dtype=array.dtype)
    if n >= 0:
        return array.reshape((-1,)+array.shape[n:])
    return array.reshape((array.shape[:n])+(-1,))

def contract_general(a: np.ndarray,
                     b: np.ndarray,
                     remaining_ndim_b: int
                    ) -> np.ndarray:
    """
    Contracts `a` and `b` such that only the last `remaining_ndim_b` axes of `b`
    and any excess axes of `a` are left uncontracted.

    Parameters
    ----------
    a : np.ndarray
        The first array to be contracted.
    b : np.ndarray
        The second array to be contracted.
    remaining_ndim_b : int
        The number of axes of `b` (from the end) that should remain
        uncontracted.

    Returns
    -------
    np.ndarray
        The result of the contraction.
    """
    ndims_to_contract = b.ndim-remaining_ndim_b
    b = partial_flatten(b, ndims_to_contract)
    if ndims_to_contract == 0:
        a = np.expand_dims(a, axis=-1)
    else:
        a = partial_flatten(a, -ndims_to_contract)
    if remaining_ndim_b == 0:
        return np.einsum("...i,i->...", a, b)
    a_pre_shape = a.shape[:-1]
    b_post_shape = b.shape[1:]
    shape = a_pre_shape + b_post_shape
    b = partial_flatten(b, -remaining_ndim_b)
    output = np.einsum("...i,ij->...j", a, b)
    return output.reshape(shape)

def contract_skeleton(coefficients: np.ndarray,
                      skeleton:np.ndarray
                     ) -> np.ndarray:
    """
    Contracts coefficients with an operator skeleton.

    Parameters
    ----------
    coefficients : np.ndarray
        The coefficients to be contracted with the skeleton.
    skeleton : np.ndarray
        The operator skeleton to be contracted with the coefficients.

    Returns
    -------
    np.ndarray
        The result of the contraction.
    """
    return contract_general(coefficients, skeleton, 2)

def contract_skeletons(coefficients: list[np.ndarray],
                       skeletons: list[np.ndarray]
                      ) -> np.ndarray:
    """
    Contracts a list of coefficients with a list of operator skeletons and then
    sums the results.

    Parameters
    ----------
    coefficients : list[np.ndarray]
        The coefficients to be contracted with the skeletons.
    skeletons : list[np.ndarray]
        The operator skeletons to be contracted with the coefficients.

    Returns
    -------
    np.ndarray
        The result of the contractions.
    """
    return sum(map(contract_skeleton, coefficients, skeletons))

def flatten_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """
    Flattens a skeleton into an array of operators.

    Parameters
    ----------
    skeleton : np.ndarray
        The operator skeleton to be flattened.

    Returns
    -------
    np.ndarray
        The flattened skeleton.
    """
    return partial_flatten(skeleton, skeleton.ndim - 2)

def get_Hs(coefficients: list[np.ndarray],
           skeletons: list[np.ndarray]
          ) -> np.ndarray:
    """
    
    """
    Hs = map(contract_skeleton, coefficients, skeletons)
    return np.concatenate(list(map(flatten_skeleton, Hs)))

class SkeletalSystem(QuantumSystem):
    """
    A quantum system with a Hamiltonian built upon an operator skeleton.
    """    
    def __init__(self,
                 drift_coefficients: list[np.ndarray],
                 drift_skeletons: list[np.ndarray],
                 ctrl_coefficients: list[np.ndarray],
                 ctrl_skeletons: list[np.ndarray],
                 hilbert_space: HilbertSpace,
                 use_graph: bool = True):
        """

        Parameters
        ----------
        drift_coefficients : list[np.ndarray]
            The coefficients of the drift Hamiltonian
        drift_skeletons : list[np.ndarray]
            The operator skeletons for the drift Hamiltonian
        coefficients : list[np.ndarray]
            The coefficients of the control Hamiltonians
        skeletons : list[np.ndarray]
            The operator skeletons for the control Hamiltonians
        hilbert_space : HilbertSpace
            The Hilbert space of the system
        use_graph : bool
            Whether to use `TensorFlow <https://www.tensorflow.org>`__ graphs
            during computation, by default ``True``
        """
        Hs = get_Hs(ctrl_coefficients, ctrl_skeletons)
        del ctrl_skeletons
        H0 = contract_skeletons(drift_coefficients, drift_skeletons)
        del drift_skeletons
        super().__init__(H0, Hs, hilbert_space, use_graph)