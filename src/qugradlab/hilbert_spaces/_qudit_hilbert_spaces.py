from typing import Optional, Iterable, Union
from abc import ABC, abstractmethod
from ._get_digit import get_digit

import numpy as np

from qugrad import HilbertSpace


class QuditSpace(HilbertSpace, ABC):
    """
    An abstract class defining the Hilbert space for a collection of qudits.
    """
    @abstractmethod
    def computational_projector(self) -> np.ndarray[bool]:
        """
        Important
        ---------
        This is an abstract method.


        Generates a boolean filter for the computation basis states in
        :attr:`basis`.

        Returns
        -------
        NDArray[Shape[:attr:`dim`], bool]
            A boolean filter for the computation basis states in
            :attr:`basis`.
        """
        ...
    def computational_subspace(self) -> HilbertSpace:
        """
        Initialises a :class:`qugrad.HilbertSpace` corresponding the computation
        subspace.

        Returns
        -------
        qugrad.HilbertSpace
            The computational subspace
        """
        return self.get_subspace(self.computational_projector())
    def dialate_operator(self, operator: np.ndarray) -> np.ndarray:
        r"""
        Dialates an operator $\hat O$ that acts on the computational subspace to
        an operator $\hat O\oplus 0$ that acts on the whole Hilbert space.

        Parameters
        ----------
        operator : NDArray[Shape[``computational_subspace().dim``, ``computational_subspace().dim``] complex]
            The operator that acts on the computational subspace

        Returns
        -------
        NDArray[Shape[:attr:`dim`, :attr:`dim`] complex]
            The dialated operator that acts on the whole Hilbert space
        """
        proj = np.identity(self.dim)[self.computational_projector()]
        return proj.T@operator@proj
    def project_operator(self, operator: np.ndarray) -> np.ndarray:
        """
        Projects an operator that acts on the Hilbert space to an
        operator that acts only on the computational subspace.

        Parameters
        ----------
        operator : NDArray[Shape[:attr:`dim`, :attr:`dim`] complex]
            The operator that acts on the whole Hilbert space

        Returns
        -------
        NDArray[Shape[``computational_subspace().dim``, ``computational_subspace().dim``] complex]
            The projected operator that acts on the computational subspace.
        """
        proj = self.computational_projector()
        return operator[proj][:, proj]

class QubitSpace(QuditSpace):
    """
    A computational Hilbert space.
    """

    _qubits: int
    "The number of qubits"
    
    def __init__(self, qubits: int):
        """
        Initialises a :class:`QubitSpace`

        Parameters
        ----------
        qubits : int
            The number of qubits
        """
        self._qubits = qubits
        super().__init__(np.arange(2**qubits))
    @property
    def qubits(self) -> int:
        "The number of qubits"
        return self._qubits
    def computational_projector(self):
        """
        Generates a boolean filter for the computation basis states in
        :attr:`basis`.

        Returns
        -------
        NDArray[Shape[:attr:`dim`], bool]
            A boolean filter for the computation basis states in
            :attr:`basis`.
        """
        return np.ones(len(self), dtype=bool)
    @staticmethod
    def _labels(digits: Iterable[str]) -> str:
        """
        Generates a strings that represent the state specified by the `digits`.

        Parameters
        ----------
        digits : Iterable[str]
            The digits representing the state

        Returns
        -------
        str
            The label for the specified state.
        """
        return f"|{''.join(digits)}⟩"
    def labels(self,
               states: Optional[Union[int, list[int]]] = None
              ) -> Union[str, list[str]]:
        """
        Generates a string (list of strings) that represent the state(s).

        Parameters
        ----------
        states : int | list[int], optional
            The state(s) to label. If ``None`` then the labels for all states in
            :attr:`basis` are returned. By default ``None``.

        Returns
        -------
        str | list[str]
            The label(s) for the specified states.
        """
        if states is None: states = self.basis
        digits = get_digit(states,
                           2,
                           np.arange(self._qubits)
                          ).astype(str)
        if isinstance(states, int):
            return self._labels(digits)
        return [self._labels(d) for d in digits]