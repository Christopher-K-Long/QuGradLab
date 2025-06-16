from typing import Optional, Iterable, Union

import numpy as np

from qugrad import HilbertSpace

from ._hamming_weight_operations import only_low_lieing_states_occupied,\
                                        sub_hamming_weight_is, \
                                        any_sub_hamming_weight_is, \
                                        all_constant_hamming_weight
from .. import QuditSpace
from .._get_digit import get_digit
    

class FermionSpace(HilbertSpace):
    """
    Represents a Fermionic Fock space.
    """
    
    _n_single_particle_states: int
    "The number of single particle states a fermion can take on"
    
    def __init__(self, n_single_particle_states: int):
        """
        Initialises a :class:`FermionSpace`

        Parameters
        ----------
        n_single_particle_states: int
            The number of single particle states a fermion can take on
        """
        self._n_single_particle_states = n_single_particle_states
        super().__init__(np.arange(2**n_single_particle_states))
    @property
    def n_single_particle_states(self) -> int:
        "The number of single particle states a fermion can take on"
        return self._n_single_particle_states
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
                           np.arange(self._n_single_particle_states)
                          ).astype(str)
        if isinstance(states, int):
            return self._labels(digits)
        return [self._labels(d) for d in digits]

class FixedParticleFermionSpace(FermionSpace):
    """
    Represents a Fermionic Fock space constrained to have a fixed particle
    number.
    """

    _n_particles: int
    "The number of particles"
    
    def __init__(self, n_single_particle_states: int, n_particles: int):
        """
        Initialises a :class:`FixedParticleFermionSpace`

        Parameters
        ----------
        n_single_particle_states: int
            The number of single particle states a fermion can take on
        n_particles : int
            The number of particles
        """
        self._n_single_particle_states = n_single_particle_states
        self._n_particles = n_particles
        basis = all_constant_hamming_weight(n_single_particle_states,
                                            n_particles)
        HilbertSpace.__init__(self, list(basis))
    @property
    def n_particles(self) -> int:
        "The number of particles"
        return self._n_particles

class FermionQuditSpace(FixedParticleFermionSpace, QuditSpace):
    """
    A :class:`FixedParticleFermionSpace` with a computational structure. The
    Hilbert space is split into the tensor product of sites (qudits), with each
    site hosting a specified number of levels. The computational subspace
    consists of the single occupation states that only have particles occupying
    the lowest two levels.
    """

    _sites: int
    "The number of sites (qudits)"

    _levels_per_site: int
    "The number of states per site (qudit)"
    
    def __init__(self,
                 sites: int,
                 levels_per_site: int,
                 n_particles: int):
        """
        Initialises a :class:`FermionQuditSpace`.

        Parameters
        ----------
        sites : int
            The number of sites (qudits)
        levels_per_site : int
            The number of states per site (qudit)
        n_particles : int
            The number of particles
        """
        self._sites = sites
        self._levels_per_site = levels_per_site
        FixedParticleFermionSpace.__init__(self,
                                           sites * levels_per_site,
                                           n_particles)
    @property
    def sites(self) -> int:
        "The number of sites (qudits)"
        return self._sites
    @property
    def levels_per_site(self) -> int:
        "The number of states per site (qudit)"
        return self._levels_per_site
    def computational_projector(self) -> np.ndarray[bool]:
        """
        Generates a boolean filter for the computation basis states in
        :attr:`basis`. The computational subspace consists of the single
        occupation states that only have particles occupying the lowest two
        levels.

        Returns
        -------
        NDArray[Shape[:attr:`dim`], bool]
            A boolean filter for the computation basis states in :attr:`basis`.
        """
        return only_low_lieing_states_occupied(2,
                                               self.levels_per_site,
                                               self._sites,
                                               self.basis) \
               & self.single_occupation_states()
    def single_occupation_states(self) -> np.ndarray[bool]:
        """
        Returns a boolean array indicating whether each of the :attr:`basis`
        states is a single occupation state (each site has exactly one
        particle).

        Returns
        -------
        NDArray[Shape[:attr:`dim`], bool]
            A boolean array indicating whether each of the :attr:`basis`
            states is a single occupation state.
        """
        return sub_hamming_weight_is(1,
                                     self.levels_per_site,
                                     self._sites,
                                     self.basis)
    def n_occupation_states(self, occupation: int) -> np.ndarray[bool]:
        """
        Returns a boolean array indicating whether each of the :attr:`basis`
        states has at most the specified occupation.

        Parameters
        ----------
        occupation : int
            The occupation to check the :attr:`basis` states for.

        Returns
        -------
        NDArray[Shape[:attr:`dim`], bool]
            A boolean array indicating whether each of the :attr:`basis`
            states has at most the specified occupation.

        Note
        ----
        :meth:`single_occupation_states` is only equivalent to
        ``n_occupation_states(1)`` when :attr:`n_particles` is greater than or
        equal to :attr:`sites`.
        """
        projector = any_sub_hamming_weight_is(occupation,
                                              self.levels_per_site,
                                              self._sites,
                                              self.basis)
        for n in range(occupation+1,self.n_particles+1):
            projector &= np.logical_not(
                any_sub_hamming_weight_is(n,
                                          self.levels_per_site,
                                          self._sites,
                                          self.basis))
        return projector