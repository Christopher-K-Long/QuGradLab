"""
Defines functions for handelling ordered lists of integers of fixed hamming
weights.
"""

from typing import Iterator, Union

import numpy as np

def next_constant_hamming_weight(x: int) -> int:
    """
    Returns the smallest integer ``n > x`` with the same hamming weight as `x`.

    Parameters
    ----------
    x : int
        The previous integer with the same hamming weight

    Returns
    -------
    int
        The next integer with the same hamming weight

    Warning
    -------
    This function will give unexpected results and potentially raise errors if
    ``x`` is not a positive integer.

    Note
    ----
    The algorithm can be found in:

    Schroeppel, Richard C.; Orman, Hilarie K. (1972-02-29). "compilation".
    HAKMEM. By Beeler, Michael; Gosper, Ralph William; Schroeppel, Richard C.
    (report). Artificial Intelligence Laboratory, Massachusetts Institute of
    Technology, Cambridge, Massachusetts, USA. hdl:1721.1/6086.
    MIT AI Memo 239. p. 81 ITEM 175 (Gosper)
    """
    c = x & -x
    r = x + c
    return (((r^x) >> 2) // c) | r

def initial_constant_hamming_weight(weight: int) -> int:
    """
    Calculates the smallest positive integer with a given hamming weight.

    Parameters
    ----------
    weight : int
        The desired hamming weight

    Returns
    -------
    x : int
        The smallest positive integer with hamming weight ``weight``
    """
    return (1<<weight)-1

def all_constant_hamming_weight(n: int, weight: int) -> Iterator[int]:
    """
    An iterator yielding monotonically increasing sequential positive integers
    of hamming weight ``weight`` with bit-strings of length ``<= n`` starting
    with the smallest positive integer with hamming weight ``weight``.

    Parameters
    ----------
    n : int
        The maximum bit string length
    weight : int
        The hamming weight of the integers

    Yields
    ------
    int
        Monotonically increasing sequential positive integers
        of hamming weight ``weight`` with bit-strings of length ``<=n`` starting
        with the smallest positive integer with hamming weight ``weight``.
    """
    if n >= weight:
        yield (x := initial_constant_hamming_weight(weight))
        if weight != 0:
            max = initial_constant_hamming_weight(n)
            x = next_constant_hamming_weight(x)
            while x < max:
                yield x
                x = next_constant_hamming_weight(x)

def hamming_weight(x: Union[int, np.ndarray[int]],
                  ) -> Union[int, np.ndarray[int]]:
    """
    Calculates the hamming weight of binary representation of each of the
    provided integers.

    Parameters
    ----------
    x : int | NDArray[Shape[s := Any_Shape], int]
        The integers to calculate the hamming weights of

    Returns
    -------
    NDArray[Shape[s], int]
        The hamming weights of the integers

    Warning
    -------
    This function only supports integers with 64 bits or less.

    Notes
    -----
    Algorithm from
    
    Warren Jr., Henry S. (2013) [2002]. Hacker's Delight (2 ed.).
    Addison Wesley - Pearson Education, Inc. pp. 81–96.
    ISBN 978-0-321-84268-8. 0-321-84268-5
    """
    t = x.dtype.type
    bit_mask = t(-1)
    s55 = t(0x5555555555555555 & bit_mask)
    s33 = t(0x3333333333333333 & bit_mask)
    s0F = t(0x0F0F0F0F0F0F0F0F & bit_mask)
    s01 = t(0x0101010101010101 & bit_mask)
    
    x = x - ((x >> 1) & s55)
    x = (x & s33) + ((x >> 2) & s33)
    x = (x + (x >> 4)) & s0F
    return (x * s01) >> (8 * (x.itemsize - 1))

def sub_hamming_weight_is(sub_hamming_weight: int,
                          sub_cardinality: int,
                          number_of_sub_strings: int,
                          bit_string: Union[int, np.ndarray[int]]
                         ) -> Union[bool, np.ndarray[bool]]:
    """
    Determines if the hamming weight of each sub-bitstring is equal to
    `sub_hamming_weight`.

    Parameters
    ----------
    sub_hamming_weight : int
        The hamming weight to check for in each sub-bitstring
    sub_cardinality : int
        The length of each sub-bitstring
    number_of_sub_strings : int
        The number of sub-bitstrings in the bit-string
    bit_string : int | NDArray[Shape[s := Any_Shape], int]
        The bit-string(s) to check the sub hamming weights of

    Returns
    -------
    int | NDArray[Shape["s"], bool]
        ``True`` if the hamming weight of each sub-bitstring is equal to
        ``sub_hamming_weight``

    Warning
    -------
    This function only supports integers with 64 bits or less.
    """
    result = np.ones_like(bit_string, dtype=bool)
    for n in range(number_of_sub_strings):
        
        result &= (hamming_weight(bit_string >> n  *sub_cardinality)
                   == sub_hamming_weight * (number_of_sub_strings - n))
        if not result.any():
            return result
    return result

def any_sub_hamming_weight_is(sub_hamming_weight: int,
                              sub_cardinality: int,
                              number_of_sub_strings: int,
                              bit_string: Union[int, np.ndarray[int]]
                             ) -> Union[bool, np.ndarray[bool]]:
    """
    Determines if the hamming weight of any sub-bitstring is equal to
    `sub_hamming_weight`.

    Parameters
    ----------
    sub_hamming_weight : int
        The hamming weight to check for in each sub-bitstring
    sub_cardinality : int
        The length of each sub-bitstring
    number_of_sub_strings : int
        The number of sub-bitstrings in the bit-string
    bit_string : int | NDArray[Shape[s := Any_Shape], int]
        The bit-string(s) to check the sub hamming weights of

    Returns
    -------
    int | NDArray[Shape["s"], bool]
        ``True`` if the hamming weight of any sub-bitstring is equal to
        ``sub_hamming_weight``

    Warning
    -------
    This function only supports integers with 64 bits or less.
    """
    result = np.zeros_like(bit_string, dtype=bool)
    for n in range(number_of_sub_strings):
        result |= (hamming_weight((bit_string >> n * sub_cardinality)
                                  &((1 << sub_cardinality) - 1))
                   == sub_hamming_weight)
    return result

def only_excited_states_occupied(number_of_high_levels: int,
                                 sub_cardinality: int,
                                 number_of_sub_strings: int,
                                 bit_string: Union[int, np.ndarray[int]]
                                ) -> Union[bool, np.ndarray[bool]]:
    """
    Checks if only the each sub-bitstring has only zeros in the first
    ``sub_cardinality-number_of_high_levels`` bits.
    
    Parameters
    ----------
    number_of_high_levels : int
        The number of high bits at the end of each sub-bitstring that can be
        non-zero
    sub_cardinality : int
        The length of each sub-bitstring
    number_of_sub_strings : int
        The number of sub-bitstrings in the bit-string
    bit_string : int | NDArray[Shape[s := Any_Shape], int]
        The bit-string(s) to check

    Returns
    -------
    bool | NDArray[Shape[s], bool]
        ``True`` if the first ``sub_cardinality-number_of_high_levels`` bits of
        each sub-bitstring are all zero.
    """
    result = np.ones_like(bit_string, dtype=bool)
    low_lieing_states = sub_cardinality-number_of_high_levels
    for n in range(number_of_sub_strings):
        result &= (bit_string >> n * sub_cardinality
                   == (bit_string >> (n * sub_cardinality + low_lieing_states))
                      << low_lieing_states)
        if not result.any():
            return result
    return result

def only_low_lieing_states_occupied(number_of_low_levels: int,
                                    sub_cardinality: int,
                                    number_of_sub_strings: int,
                                    bit_string: Union[int, np.ndarray[int]]
                                   ) -> Union[bool, np.ndarray[bool]]:
    """
    Checks if only the each sub-bitstring has only zeros in the last
    ``sub_cardinality-number_of_low_levels`` bits.
    
    Parameters
    ----------
    number_of_low_levels : int
        The number of high bits at the end of each sub-bitstring that can be
        non-zero
    sub_cardinality : int
        The length of each sub-bitstring
    number_of_sub_strings : int
        The number of sub-bitstrings in the bit-string
    bit_string : int | NDArray[Shape[s := Any_Shape], int]
        The bit-string(s) to check

    Returns
    -------
    bool | NDArray[Shape[s], bool]
        ``True`` if the last ``sub_cardinality-number_of_low_levels`` bits of
        each sub-bitstring are all zero.
    """
    result = np.ones_like(bit_string, dtype=bool)
    excited_states = sub_cardinality-number_of_low_levels
    for n in range(number_of_sub_strings):
        result &= (bit_string >> (n * sub_cardinality + number_of_low_levels)
                   == (bit_string >> ((n + 1) * sub_cardinality))
                      << excited_states)
        if not result.any():
            return result
    return result