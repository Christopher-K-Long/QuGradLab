from typing import Optional, Iterable, Union

import numpy as np

from qugrad import HilbertSpace

from .. import QuditSpace
from .._get_digit import get_digit

class BosonSpace(HilbertSpace):
    """
    Represents a truncated Fock space of boson modes.
    """
    
    _n_single_particle_states: int
    "The number of single particle states"
    

    _truncation_level: int
    "The maximum number of bosons per single particle state"
    
    def __init__(self, n_single_particle_states: int, truncation_level: int):
        """
        Initialises a :class:`BosonSpace`

        Parameters
        ----------
        n_single_particle_states: int
            The number of single particle states
        truncation_level: int
            The maximum number of bosons per single particle state
        """
        self._n_single_particle_states = n_single_particle_states
        self._truncation_level = truncation_level
        super().__init__(np.arange(truncation_level**n_single_particle_states))
    @property
    def n_single_particle_states(self) -> int:
        "The number of single particle states"
        return self._n_single_particle_states
    @property
    def truncation_level(self) -> int:
        "The maximum number of bosons per single particle state"
        return self._truncation_level
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
        
        return f"|{', '.join(digits)}⟩"
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
                           self._truncation_level,
                           np.arange(self._n_single_particle_states)
                          ).astype(str)
        if isinstance(states, int):
            return self._labels(digits)
        return [self._labels(d) for d in digits]

class BosonQuditSpace(BosonSpace, QuditSpace):
    """
    A :class:`BosonSpace` with a computational structure. The computational
    subspace consists of the states which have unoccupied and singly occupied
    single particle states.
    """
    def computational_projector(self)-> np.ndarray[bool]:
        """
        Generates a boolean filter for the computation basis states in
        :attr:`basis`. The computational subspace consists of the states which
        have unoccupied and singly occupied single particle states.

        Returns
        -------
        NDArray[Shape[:attr:`dim`], bool]
            A boolean filter for the computation basis states in :attr:`basis`.
        """
        projector = get_digit(self.basis,
                              self._truncation_level,
                              np.arange(self._n_single_particle_states)) <= 1
        return np.all(projector, axis=-1)