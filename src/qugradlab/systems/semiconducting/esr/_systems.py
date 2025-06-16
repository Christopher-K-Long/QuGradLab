"""
A collection of :class:`qugrad.QuantumSystem` s for electron spin resonance (ESR)
devices.
"""

from functools import reduce

import numpy as np
import tensorflow as tf

from scipy.linalg import block_diag

from ._controls import Controls
from ._device import Device
from ....hilbert_spaces import QubitSpace
from ....hilbert_spaces.fermionic import FermionQuditSpace
from ....pulses.composition import concatenate_functions
from ....pulses.invertible_functions.scaling import linear_rescaling
from ...skeletons.qubits import QubitSystem
from ...skeletons.fermionic import FermionicSystem

PAULI_X = np.array([[0, 1],
                    [1, 0]])
"""Pauli-x operator"""

PAULI_Z = np.array([[1, 0],
                    [0, -1]])
"""Pauli-z operator"""

class SpinChain(Controls, QubitSystem):
    r"""
    A :class:`qugrad.QuantumSystem` for a spin chain with electron spin
    resonance (ESR) controls. The Hamiltonian is given by
    $$
    H(t) = \sum_{i=0}^{\texttt{spins}-1} \frac{1}{2} B_i Z_i
    + g(t) \sum_{i=0}^{\texttt{spins}-1} \frac{1}{2} X_i
    + \frac{1}{4} \sum_{i=0}^{\texttt{spins}-2} J_i(t)
    \vec\sigma_i \cdot \vec\sigma_{i+1},
    $$
    where
    $\vec\sigma_i\equiv\begin{pmatrix}X_i & Y_i & Z_i\end{pmatrix}^\intercal$
    is the vector of the Pauli-x, -y, and -z operators acting on the $i$th spin,
    $B_i$ are the Zeeman splittings, $J_i(t)$ is the exchange coupling, and
    $$
    g(t) = \sum_{j=0}^{\texttt{spins}-1}\real\left(a_j(t)e^{i\omega_j t}\right),
    $$
    is the Rabi drive with frequency components $\omega_j$ and amplitudes
    $a_j(t)$.

    See Also
    --------
    * :class:`SpinChainAngledDrive`
    * :class:`ValleyChain`


    -
    """

    _ferromagnetic: bool
    """Whether the exchange coupling is ferromagnetic or antiferromagnetic"""
    
    def __init__(self,
                 spins: int,
                 zeeman_splittings: np.ndarray[float],
                 max_drive_strength: float,
                 J_max: float,
                 J_min: float = 0,
                 feromagnetic: bool = True,
                 use_graph: bool = True):
        r"""
        Initialises a spin chain with ESR controls. The Hamiltonian is
        given by:
        $$
        H(t) = \sum_{i=0}^{\texttt{spins}-1} \frac{1}{2} B_i Z_i
        + g(t) \sum_{i=0}^{\texttt{spins}-1} \frac{1}{2} X_i
        + \frac{1}{4} \sum_{i=0}^{\texttt{spins}-2} J_i(t)
        \vec\sigma_i \cdot \vec\sigma_{i+1},
        $$
        where
        $\vec\sigma_i\equiv\begin{pmatrix}X_i&Y_i&Z_i\end{pmatrix}^\intercal$
        is the vector of the Pauli-x, -y, and -z operators acting on the $i$th
        spin, $B_i$ corresponds to `zeeman_splittings`, $J_i(t)$ is the exchange
        coupling, and
        $$
        g(t) =
        \sum_{j=0}^{\texttt{spins}-1}\real\left(a_j(t)e^{i\omega_j t}\right),
        $$
        is the Rabi drive with frequency components $\omega_j$ and amplitudes
        $a_j(t)$.
        
        Parameters
        ----------
        spins : int
            The number of spins in the chain
        zeeman_splittings : NDArray[Shape[spins], float]
            The Zeeman splitting of each of the spins
        max_drive_strength : float
            The maximum drive strength that can be applied at a specific
            frequency and quadrature. That is if their are ``n_drive_ctrl``
            frequencies and both quadratures are used then the maximum amplitude
            of the drive that can be applied to the device is::

                np.sqrt(2) * n_drive_ctrl * max_drive_strength
        J_max : float
            The minimum value of the exchange coupling $J$
        J_min : float
            The maximum value of the exchange coupling $J$, by default 0
        feromagnetic : bool
            If ``True``, the exchange coupling is ferromagnetic. If ``False``,
            the exchange coupling is antiferromagnetic. By default, ``True``.
        use_graph : bool
            Whether to use `TensorFlow <https://www.tensorflow.org>`__ graphs
            during computation, by default ``True``
        """
        Controls.__init__(self,
                          zeeman_splittings,
                          max_drive_strength,
                          J_min,
                          J_max)
        single_qubit_drift_coefficients = \
            np.multiply.outer(zeeman_splittings, np.array([0, 0, 0.5]))
            
        single_qubit_ctrl_coefficients = np.array([[[0.5, 0, 0]]*spins])
        
        forward_connectivity = np.einsum("ij,jk->ijk",
                                         np.eye(spins, spins, 0),
                                         np.eye(spins, spins, 1)
                                        )[:-1]
        backward_connectivity = np.einsum("ij,jk->ijk",
                                          np.eye(spins, spins, 1),
                                          np.eye(spins, spins, -1)
                                         )[:-1]
        connectivity = forward_connectivity + backward_connectivity
        J = 0.5**3*np.multiply.outer(connectivity, np.identity(3))
        # one factor of 0.5 is for double counting, the other two are the
        #   factors of two differences between spin operators and Pauli
        #   operators
        if feromagnetic: J *= -1
        self._ferromagnetic = feromagnetic
        QubitSystem.__init__(self,
                             QubitSpace(spins),
                             single_qubit_drift_coefficients,
                             np.zeros((spins, spins, 3, 3)),
                             single_qubit_ctrl_coefficients,
                             J,
                             use_graph)
    @property
    def ferromagnetic(self) -> bool:
        """
        Whether the exchange coupling is ferromagnetic or antiferromagnetic
        """
        return self._ferromagnetic

class SpinChainAngledDrive(Controls, QubitSystem):
    r"""
    A :class:`qugrad.QuantumSystem` for a spin chain with electron spin
    resonance (ESR) controls. The Hamiltonian is given by
    $$
    H(t) = \sum_{i=0}^{\texttt{spins}-1} \frac{1}{2} B_i Z_i
    + g(t) \sum_{i=0}^{\texttt{spins}-1} \frac{1}{2}
    \vec v_i \cdot \vec\sigma_i
    + \frac{1}{4} \sum_{i=0}^{\texttt{spins}-2} J_i(t)
    \vec\sigma_i \cdot \vec\sigma_{i+1},
    $$
    where
    $\vec\sigma_i\equiv\begin{pmatrix}X_i & Y_i & Z_i\end{pmatrix}^\intercal$
    is the vector of the Pauli-x, -y, and -z operators acting on the $i$th spin,
    $B_i$ are the Zeeman splittings, $J_i(t)$ is the exchange coupling,
    $\vec v_i$ are the drive vectors, and
    $$
    g(t) = \sum_{j=0}^{\texttt{spins}-1}\real\left(a_j(t)e^{i\omega_j t}\right),
    $$
    is the Rabi drive with frequency components $\omega_j$ and amplitudes
    $a_j(t)$.

    See Also
    --------
    * :class:`SpinChain`
    * :class:`ValleyChain`


    -
    """

    _ferromagnetic: bool
    """Whether the exchange coupling is ferromagnetic or antiferromagnetic"""

    _drive_vectors: np.ndarray[float]
    """The drive vectors for each of the spins"""
    
    def __init__(self,
                 spins: int,
                 zeeman_splittings: np.ndarray[float],
                 drive_vectors: np.ndarray[float],
                 max_drive_strength: float,
                 J_max: float,
                 J_min: float = 0,
                 feromagnetic: bool = True,
                 use_graph: bool = True):
        r"""
        Initialises a spin chain with ESR controls. The Hamiltonian is
        given by:
        $$
        H(t) = \sum_{i=0}^{\texttt{spins}-1} \frac{1}{2} B_i Z_i
        + g(t) \sum_{i=0}^{\texttt{spins}-1} \frac{1}{2}
        \vec v_i \cdot \vec\sigma_i
        + \frac{1}{4} \sum_{i=0}^{\texttt{spins}-2} J_i(t)
        \vec\sigma_i \cdot \vec\sigma_{i+1},
        $$
        where
        $\vec\sigma_i\equiv\begin{pmatrix}X_i&Y_i&Z_i\end{pmatrix}^\intercal$
        is the vector of the Pauli-x, -y, and -z operators acting on the $i$th
        spin, $B_i$ corresponds to `zeeman_splittings`, $J_i(t)$ is the exchange
        coupling, $\vec v_i$ corresponds to `drive_vectors`, and
        $$
        g(t) =
        \sum_{j=0}^{\texttt{spins}-1}\real\left(a_j(t)e^{i\omega_j t}\right),
        $$
        is the Rabi drive with frequency components $\omega_j$ and amplitudes
        $a_j(t)$.
        
        Parameters
        ----------
        spins : int
            The number of spins in the chain
        zeeman_splittings : NDArray[Shape[spins], float]
            The Zeeman splitting of each of the spins
        drive_vectors : NDArray[Shape[spins, 3], float]
            The drive vectors for each of the spins
        max_drive_strength : float
            The maximum drive strength that can be applied at a specific
            frequency and quadrature. That is if their are ``n_drive_ctrl``
            frequencies and both quadratures are used then the maximum amplitude
            of the drive that can be applied to the device is::

                np.sqrt(2) * n_drive_ctrl * max_drive_strength
        J_max : float
            The minimum value of the exchange coupling $J$
        J_min : float
            The maximum value of the exchange coupling $J$, by default 0
        feromagnetic : bool
            If ``True``, the exchange coupling is ferromagnetic. If ``False``,
            the exchange coupling is antiferromagnetic. By default, ``True``.
        use_graph : bool
            Whether to use `TensorFlow <https://www.tensorflow.org>`__ graphs
            during computation, by default ``True``
        """
        Controls.__init__(self,
                          zeeman_splittings,
                          max_drive_strength,
                          J_min,
                          J_max)
        self._drive_vectors = np.array(drive_vectors)
        self._drive_vectors.flags.writeable = False
        single_qubit_drift_coefficients = \
            np.multiply.outer(zeeman_splittings, np.array([0, 0, 0.5]))
        
        forward_connectivity = np.einsum("ij,jk->ijk",
                                         np.eye(spins, spins, 0),
                                         np.eye(spins, spins, 1)
                                        )[:-1]
        backward_connectivity = np.einsum("ij,jk->ijk",
                                          np.eye(spins, spins, 1),
                                          np.eye(spins, spins, -1)
                                         )[:-1]
        connectivity = forward_connectivity + backward_connectivity
        J = 0.5**3*np.multiply.outer(connectivity, np.identity(3))
        # one factor of 0.5 is for double counting, the other two are the
        #   factors of two differences between spin operators and Pauli
        #   operators
        if feromagnetic: J *= -1
        self._ferromagnetic = feromagnetic
        QubitSystem.__init__(self,
                             QubitSpace(spins),
                             single_qubit_drift_coefficients,
                             np.zeros((spins, spins, 3, 3)),
                             self._drive_vectors,
                             J,
                             use_graph)
    @property
    def ferromagnetic(self) -> bool:
        """
        Whether the exchange coupling is ferromagnetic or antiferromagnetic
        """
        return self._ferromagnetic
    @property
    def drive_vectors(self) -> np.ndarray[float]:
        """The drive vectors for each of the spins"""
        return self._drive_vectors

class ValleyChain(Controls, FermionicSystem):
    r"""
    A :class:`qugrad.QuantumSystem` for a linear array of silicon electron
    quantum dots including the valley degree of freedom and electron spin
    resonance (ESR) controls. The Hamiltonian is given by
    $$
    H(t) = \sum_{\substack{{i,j}\\{\alpha,\beta}}}i_{\alpha}^\dagger j_{\beta}
    \tilde t^{ij}_{\alpha\beta}
    +U_0\sum_{i,\left<\alpha,\beta\right>}
    i_{\alpha}^\dagger i_{\beta}^\dagger i_{\alpha} i_{\beta}
    +U_1\sum_{i,\alpha,\beta}
    i_{\alpha}^\dagger i_{\beta}^\dagger i_{\alpha} i_{(\bar\beta_v,\beta_s)},
    $$
    where $i_{\alpha}^\dagger$ and $i_{\alpha}$ are the creation and annihilation
    operators for the $i$th dot and $\alpha$ indexes the spin and valley index
    degrees of freedom, $\left<\alpha,\beta\right>$ represents unique unordered
    pairs of $\alpha$ and $\beta$, $(\bar\beta_v,\beta_s)$ has the same spin
    index as $\beta$ but the oposite valley index, U_0$ is the on-site Coulomb
    no-valley-flip interaction, $U_1$ is the on-site Coulomb valley-flip
    interaction, the interdot hoppings have the form
    $\tilde t^{ij}_{\alpha\beta}=h^{ij}(t)\delta_{\alpha\beta}$, and the
    intradot hoppings take the form
    $$
    \tilde t^{ii}_{\alpha\beta}=\begin{bmatrix}
    \frac{1}{2}\left(V+Z_i\right)&g(t)^*&&\nu_{\textrm{SO}}^*\\
    g(t)&\frac{1}{2}\left(V-Z_i\right)&\nu_{\textrm{SO}}&\\
    &\nu_{\textrm{SO}}^*&\frac{1}{2}\left(-V+Z_i\right)&g(t)^*\\
    \nu_{\textrm{SO}}&&g(t)&\frac{1}{2}\left(-V-Z_i\right)
    \end{bmatrix}\begin{matrix}\left(1,\uparrow\right)\\\left(1,\downarrow\right)\\\left(0,\uparrow\right)\\\left(0,\downarrow\right)\end{matrix}
    $$
    where $V$ is the valley splitting, $Z_i$ is the Zeeman splitting on the
    $i$th dot, $\nu_{\textrm{SO}}$ is the valley-spin-orbit coupling, and
    $$
    g(t) = \sum_{j=0}^{\texttt{dots}-1}\real\left(a_j(t)e^{i\omega_j t}\right),
    $$
    is the Rabi drive with frequency components $\omega_j$ and amplitudes
    $a_j(t)$.

    See Also
    --------
    * :class:`SpinChain`
    * :class:`SpinChainAngledDrive`


    -
    """

    _u: float
    """The on-site Coulomb no-valley-flip interaction"""
    
    _u_valley_flip: float
    """The on-site Coulomb valley-flip interaction"""
    
    _valley_spin_orbit_coupling: float
    """The valley-spin-orbit coupling"""

    def __init__(self,
                 dots: int,
                 electrons: int,
                 zeeman_splittings: np.ndarray[complex],
                 valley_splitting: float,
                 u: float,
                 u_valley_flip: float,
                 valley_spin_orbit_coupling: float,
                 max_drive_strength: float,
                 J_max: float,
                 J_min: float = 0,
                 use_graph: bool = True):
        r"""
        Initialises a linear array of silicon electron quantum dots including
        the valley degree of freedom and electron spin resonance (ESR) controls.
        The Hamiltonian is given by
        $$
        H(t) = \sum_{\substack{{i,j}\\{\alpha,\beta}}}
        i_{\alpha}^\dagger j_{\beta}
        \tilde t^{ij}_{\alpha\beta}
        +U_0\sum_{i,\left<\alpha,\beta\right>}
        i_{\alpha}^\dagger i_{\beta}^\dagger i_{\alpha} i_{\beta}
        +U_1\sum_{i,\alpha,\beta}
        i_{\alpha}^\dagger i_{\beta}^\dagger
        i_{\alpha} i_{(\bar\beta_v,\beta_s)},
        $$
        where $i_{\alpha}^\dagger$ and $i_{\alpha}$ are the creation and
        annihilation operators for the $i$th dot and $\alpha$ indexes the spin
        and valley index degrees of freedom, $\left<\alpha,\beta\right>$
        represents unique unordered pairs of $\alpha$ and $\beta$,
        $(\bar\beta_v,\beta_s)$ has the same spin index as $\beta$ but the
        oposite valley index, U_0$ is the on-site Coulomb no-valley-flip
        interaction and corresponds to `u`, $U_1$ is the on-site Coulomb
        valley-flip interaction and corresponds to `u_valley_flip`, the interdot
        hoppings have the form
        $\tilde t^{ij}_{\alpha\beta}=h^{ij}(t)\delta_{\alpha\beta}$, and the
        intradot hoppings take the form
        $$
        \tilde t^{ii}_{\alpha\beta}=\begin{bmatrix}
        \frac{1}{2}\left(V+Z_i\right)&g(t)^*&&\nu_{\textrm{SO}}^*\\
        g(t)&\frac{1}{2}\left(V-Z_i\right)&\nu_{\textrm{SO}}&\\
        &\nu_{\textrm{SO}}^*&\frac{1}{2}\left(-V+Z_i\right)&g(t)^*\\
        \nu_{\textrm{SO}}&&g(t)&\frac{1}{2}\left(-V-Z_i\right)
        \end{bmatrix}\begin{matrix}
        \left(1,\uparrow\right)\\
        \left(1,\downarrow\right)\\
        \left(0,\uparrow\right)\\
        \left(0,\downarrow\right)
        \end{matrix}
        $$
        where $V$ corresponds to `valley_splitting`, $Z_i$ is the Zeeman
        splitting on the $i$th dot and corresponds `zeeman_splittings`,
        $\nu_{\textrm{SO}}$ corresponds to `valley_spin_orbit_coupling`, and
        $$
        g(t) = \sum_{j=0}^{\texttt{dots}-1}\real\left(a_j(t)e^{i\omega_j t}\right),
        $$
        is the Rabi drive with frequency components $\omega_j$ and amplitudes
        $a_j(t)$.

        Parameters
        ----------
        dots : int
            The number of dots in the array
        electrons : int
            The number of electrons in the 
        zeeman_splittings : NDArray[Shape[spins], float]
            The Zeeman splitting of each of the spins
        valley_splitting : float
            The valley splitting
        u : float
            The on-site Coulomb no-valley-flip interaction
        u_valley_flip : float
            The on-site Coulomb valley-flip interaction
        valley_spin_orbit_coupling : float
            The valley-spin-orbit coupling
        max_drive_strength : float
            The maximum drive strength that can be applied at a specific
            frequency and quadrature. That is if their are ``n_drive_ctrl``
            frequencies and both quadratures are used then the maximum amplitude
            of the drive that can be applied to the device is::

                np.sqrt(2) * n_drive_ctrl * max_drive_strength
        J_max : float
            The minimum value of the exchange coupling $J$
        J_min : float
            The maximum value of the exchange coupling $J$, by default 0
        use_graph : bool
            Whether to use `TensorFlow <https://www.tensorflow.org>`__ graphs
            during computation, by default ``True``
        """
        # Defining the control amplitude functions
        rescale_rabi_drive = \
            linear_rescaling.specify_parameters(min=-max_drive_strength,
                                                max=max_drive_strength)
        rescale_J = linear_rescaling.specify_parameters(min=J_min, max=J_max)
        hopping_amplitude = lambda x: tf.sqrt(rescale_J(x)*self.u)/2
        self._rescale_and_concatenate = \
            concatenate_functions([rescale_rabi_drive, hopping_amplitude])
        # Initialising the device parameters
        Device.__init__(self,
                        zeeman_splittings,
                        max_drive_strength,
                        J_min,
                        J_max)
        self._u = u
        self._u_valley_flip = u_valley_flip
        self._valley_spin_orbit_coupling = valley_spin_orbit_coupling

        # Generating the drift hopping Hamiltonian
        drift_hopping_blocks = []
        for zeeman_splitting in zeeman_splittings:
            drift_hopping_blocks.append(
                valley_spin_orbit_coupling  * np.kron(PAULI_X, PAULI_X)
                + valley_splitting/2 * np.kron(PAULI_Z, np.eye(2))
                + zeeman_splitting/2 * np.kron(np.eye(2), PAULI_Z))
        drift_hoppings = block_diag(*drift_hopping_blocks)

        # Generating the Rabi drive Hamiltonian
        esr_drives = [0.5*np.kron(np.eye(dots*2), PAULI_X)]

        # Generating the interdot hopping Hamiltonian
        forward_connectivity = np.einsum("ij,jk->ijk",
                                         np.eye(dots, dots, 0),
                                         np.eye(dots, dots, 1)
                                        )[:-1]
        backward_connectivity = np.einsum("ij,jk->ijk",
                                          np.eye(dots, dots, 1),
                                          np.eye(dots, dots, -1)
                                         )[:-1]
        copuling_matrix = np.identity(4)
        # When hopping in the oposite direction the oposite rotation is picked
        #   up to ensure hermiticity
        reverse_coupling_matrix = copuling_matrix.T.conj()
        interdot_hoppings = \
            np.kron(forward_connectivity,
                    np.expand_dims(copuling_matrix, axis=0)) \
            + np.kron(backward_connectivity,
                      np.expand_dims(reverse_coupling_matrix, axis=0))
                           
        ctrl_hoppings = np.concatenate([esr_drives, interdot_hoppings], axis=0)

        # Generating the Coulomb interaction Hamiltonian
        # No spin flip
        spin_coupling = np.einsum("ij,kl->ikjl", np.identity(2), np.identity(2))
        # Valley couplings
        hartree_coupling = \
            np.einsum("ij,kl->ikjl", np.identity(2), np.identity(2))
        single_flip_valley_coupling = \
           np.einsum("ij,kl,jk->ijkl", np.identity(2), PAULI_X, np.identity(2))\
          +np.einsum("ij,kl,jk->ijlk", np.identity(2), PAULI_X, np.identity(2))\
          +np.einsum("ij,kl,jk->iljk", np.identity(2), PAULI_X, np.identity(2))\
          +np.einsum("ij,kl,jk->lijk", np.identity(2), PAULI_X, np.identity(2))
        valley_couplings = -u * hartree_coupling \
                           -u_valley_flip * single_flip_valley_coupling
        # On-site Coulomb only
        dot_couplings = np.einsum("ij,kl,jk->ijkl",
                                  np.identity(dots),
                                  np.identity(dots),
                                  np.identity(dots))
        coulomb_integrals = 0.5*reduce(np.kron, [dot_couplings,
                                                 valley_couplings,
                                                 spin_coupling])
        # The 0.5 accounts for double counting
        FermionicSystem.__init__(self,
                                 FermionQuditSpace(dots, 4, electrons),
                                 drift_hoppings,
                                 coulomb_integrals,
                                 ctrl_hoppings,
                                 use_graph)

    @property
    def u(self) -> float:
        """The on-site Coulomb no-valley-flip interaction"""
        return self._u
    @property
    def u_valley_flip(self) -> float:
        """The on-site Coulomb valley-flip interaction"""
        return self._u_valley_flip
    @property
    def valley_spin_orbit_coupling(self) -> float:
        """The valley-spin-orbit coupling"""
        return self._valley_spin_orbit_coupling