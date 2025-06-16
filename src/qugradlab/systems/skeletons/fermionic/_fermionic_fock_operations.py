"""
Defines functions for bitwise operations on Fermionic Fock states encoded as
bit-strings represented by integers.
"""

from typing import Iterable, Union

import numpy as np

def is_empty(fock_states: Union[int, np.ndarray[int]],
             i: int
            ) -> Union[bool, np.ndarray[bool]]:
    """
    Deterines whether a fermionic orbital ``i`` is empty.

    Parameters
    ----------
    fock_states : int | np.NDArray[Shape[s := Any_Shape], int]
        The Fock state(s) for which the occupation is to be determined
    i : int
        The orbtial in the Fock state to check the occupation of.

    Returns
    -------
    bool | NDArray[Shape[s], bool]
        ``True`` if the ``i`` th orbital is empty.

    Note
    ----
    This function is the compliment of :func:`is_occupied()`.
    """
    return fock_states & (1 << i) == 0

def is_occupied(fock_states: Union[int, np.ndarray[int]],
                i: int
               ) -> Union[bool, np.ndarray[bool]]:
    """
    Deterines whether a fermionic orbital ``i`` is occupied.

    Parameters
    ----------
    fock_states : int | NDArray[Shape[s := Any_Shape], int]
        The Fock state(s) for which the occupation is to be determined
    i : int
        The orbtial in the Fock state to check the occupation of.

    Returns
    -------
    bool | NDArray[Shape[s], bool]
        ``True`` if the ``i`` th orbital is occupied.

    Note
    ----
    This function is the compliment of :func:`is_empty()`.
    """
    return np.logical_not(is_empty(fock_states, i))

def fill_orbitals(i: int, j: int) -> int:
    """
    Returns the bit-string corresponding to the fermionic Fock state with
    orbitals in the range ``i:j`` occupied only.

    Parameters
    ----------
    i : int
        The lower orbital index
    j : int
        The upper orbital index

    Returns
    -------
    int
        The bit-string corresponding to the Fock state with orbitals in the
        range ``i:j`` occupied only.
    """
    return ((1<<j-i)-1)<<i

def select_orbitals(fock_states: Union[int, np.ndarray[int]],
                    i: int,
                    j: int
                   ) -> Union[int, np.ndarray[int]]:
    """
    Empties all fermionic orbitals not in the range ``i:j``.

    Parameters
    ----------
    fock_states : int | NDArray[Shape[s := Any_Shape], int]
        The Fock state(s) to mask
    i : int
        The lower orbital index
    j : int
        The upper orbital index

    Returns
    -------
     int | NDArray[Shape[s], int]
        The masked Fock state(s)
    """
    return fock_states & fill_orbitals(i,j)

def flips(fock_states: Union[int, np.ndarray[int]],
          indices: Iterable[int]
         ) -> Union[int, np.ndarray[int]]:
    """
    Flips the bits at the positions listed in `indices` for each fermionic Fock
    state provided.

    Parameters
    ----------
    fock_states : int | NDArray[Shape[s := Any_Shape], int]
        The Fock state(s) to flip the occupations for
    indices : list of int
        The positions at which to flip the occupations

    Returns
    -------
    int | NDArray[Shape[s], int]
        The Fock state(s) with the flipped occupations
    """
    output = np.copy(fock_states)
    for i in indices:
        output ^= 1<< i
    return output