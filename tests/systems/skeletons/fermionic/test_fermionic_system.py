import numpy as np

from qugradlab.systems.skeletons.fermionic import FermionicSystem, \
                                                  fermionic_fock_skeleton
from qugradlab.hilbert_spaces.fermionic import FermionSpace, \
                                               FixedParticleFermionSpace
from qugradlab.systems.skeletons._skeletal_system import contract_skeletons, \
                                                         get_Hs

N_SINGLE_PARTICLE_STATES = 4
NCTRL = 3

def fermionic_system_initialisation_check(hilbert_space):
    drift_hoppings = np.random.rand(N_SINGLE_PARTICLE_STATES,
                                    N_SINGLE_PARTICLE_STATES)
    coulomb_integrals = np.random.rand(N_SINGLE_PARTICLE_STATES,
                                       N_SINGLE_PARTICLE_STATES,
                                       N_SINGLE_PARTICLE_STATES,
                                       N_SINGLE_PARTICLE_STATES)
    ctrl_hoppings = np.random.rand(NCTRL,
                                   N_SINGLE_PARTICLE_STATES,
                                   N_SINGLE_PARTICLE_STATES)
    use_graph = True
    system = FermionicSystem(hilbert_space,
                             drift_hoppings,
                             coulomb_integrals,
                             ctrl_hoppings,
                             use_graph)
    t = fermionic_fock_skeleton.second_order_tensor(hilbert_space)
    u = fermionic_fock_skeleton.fourth_order_tensor(hilbert_space)
    H0 = contract_skeletons([drift_hoppings, coulomb_integrals], [t, u])
    Hs = get_Hs([ctrl_hoppings], [t])
    
    assert np.array_equal(system.H0, H0)
    assert np.array_equal(system.Hs, Hs)
    assert system.hilbert_space == hilbert_space
    assert system.using_graph == use_graph

    use_graph = False
    system = FermionicSystem(hilbert_space,
                             drift_hoppings,
                             coulomb_integrals,
                             ctrl_hoppings,
                             use_graph)
    assert system.using_graph == use_graph

def test_fermionic_system_initialisation():
    hilbert_space = FermionSpace(N_SINGLE_PARTICLE_STATES)
    fermionic_system_initialisation_check(hilbert_space)

def test_fermionic_system_initialisation_fixed_particle_number():
    for particle_number in range(N_SINGLE_PARTICLE_STATES+1):
        hilbert_space = FixedParticleFermionSpace(N_SINGLE_PARTICLE_STATES,
                                                  particle_number)
        fermionic_system_initialisation_check(hilbert_space)