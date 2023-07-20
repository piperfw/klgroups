#!/usr/bin/env python

import sys
# Assumes script placed in a run from oqupy/examples
# Note .git version required for working two-time correlation calculations (below)
sys.path.insert(0,'..')
import oqupy
import numpy as np
import matplotlib.pyplot as plt
from oqupy import contractions


# computational parameters
tempo_parameters = oqupy.TempoParameters(dt=0.2, dkmax=20, epsrel=10**(-6))
start_time = 0.0
end_time = 10.0

# Molecule 1 parameters
alpha1 = 0.1 # Bath coupling 
nuc1 = 0.15 # Bath cut-off
T1 = 0.01 # Bath temperature
omega1 = 0.05 # System Frequency
# Molecule 2 (current use same bath, different detuning)
omega2 = -0.05 
# Photon
Omega = 1 # LM coupling
omegac = 0.0 
kappa = 0.2 # dissipation
# Incoherent drive/loss of molecules (currently the same)
Gamma_down = 0.2 #
Gamma_up = 0.8 * Gamma_down

sigma_z = oqupy.operators.sigma("z")
sigma_plus = oqupy.operators.sigma("+")
sigma_minus = oqupy.operators.sigma("-")

# Molecular Hamiltonians 
def H_MF_1(t, a): # time (unused), field expectation  
    return 0.5 * omega1 * sigma_z +\
        0.5 * Omega * (a * sigma_plus + np.conj(a) * sigma_minus)
def H_MF_2(t, a):
    return 0.5 * omega2 * sigma_z +\
        0.5 * Omega * (a * sigma_plus + np.conj(a) * sigma_minus)

fractions = [0.5, 0.5] # fractions of type-1 and type-2 molecules
# Field equation of motion
def field_eom(t, states, field): # time, list of (current) molecule density matrices, current field expectation
    sx_exp_list = [np.matmul(sigma_minus, state).trace() for state in states]
    sx_exp_weighted_sum = sum([fraction*sx_exp for fraction, sx_exp in zip(fractions, sx_exp_list)])
    return -(1j*omegac+kappa)*field - 0.5j*Omega*sx_exp_weighted_sum

# System objects for each molecule type
molecule1 = oqupy.TimeDependentSystemWithField(H_MF_1, 
                                               gammas=[lambda t: Gamma_up, lambda t: Gamma_down],
                                               lindblad_operators=[lambda t: sigma_plus, lambda t: sigma_minus],
                                               )
molecule2 = oqupy.TimeDependentSystemWithField(H_MF_2,
                                               gammas=[lambda t: Gamma_up, lambda t: Gamma_down],
                                               lindblad_operators=[lambda t: sigma_plus, lambda t: sigma_minus],
                                               )
# MeanFieldSystem containing both molecules
mean_field_system = oqupy.MeanFieldSystem([molecule1, molecule2], field_eom=field_eom)

# Correlations used to construct bath objects
correlations1 = oqupy.PowerLawSD(alpha=alpha1,
                                zeta=1, # Ohmic
                                cutoff=nuc1,
                                cutoff_type='gaussian', # exp(-(nu/nuc)**2)
                                temperature=T1)
bath1 = oqupy.Bath(0.5 * sigma_z, correlations1)
# process tensor (currently used for both molecules)
process_tensor = oqupy.pt_tempo_compute(bath=bath1,
                                        start_time=start_time,
                                        end_time=end_time,
                                        parameters=tempo_parameters)
# could construct a different bath -> process tensor for molecule2


# Initial states
initial_field = np.sqrt(0.05)
initial_state_1 = np.array([[0,0],[0,1]]) # spin down
initial_state_2 = np.array([[0,0],[0,1]])
initial_state_list = [initial_state_1, initial_state_2] # must be in a list

# DYNAMICS CALCULATION
# Compute dynamics using same process tensor for each molecule
mean_field_dynamics_process = \
        contractions.compute_dynamics_with_field(mean_field_system,
                initial_field=initial_field, 
                initial_state_list=initial_state_list, 
                start_time=start_time,
                process_tensor_list = [process_tensor, process_tensor])

fig, axes = plt.subplots(2, figsize=(6,6), sharex=True)
times, fields = mean_field_dynamics_process.field_expectations()
axes[0].plot(times, np.abs(fields)**2)
axes[0].set_ylabel('n/N')
for i, molecule_dynamics in enumerate(mean_field_dynamics_process.system_dynamics):
    times, sm = molecule_dynamics.expectations(sigma_minus, real=True)
    axes[1].plot(times, sm.real, label=f'Molecule {i}')
axes[1].legend()
axes[1].set_ylabel('Re<sigma^->')
axes[1].set_xlabel('t')
fig.savefig('dynamics.pdf', bbox_inches='tight')

# TWO-TIME CORRELATOR CALCULATION - working with .git version only
# Control objects for each molecule - by default have no action i.e. act identically 
control_list = [oqupy.Control(molecule1.dimension), oqupy.Control(molecule2.dimension)]
# N.B. apply_time must be a float otherwise (if int) interpreted as timestep 
# apply raising operator to both molecules at t=0
apply_time = 0.0
control_list[0].add_single(apply_time, oqupy.operators.left_super(sigma_plus))
control_list[1].add_single(apply_time, oqupy.operators.left_super(sigma_plus))
# repeat calculation with this control
mean_field_dynamics_process = \
        contractions.compute_dynamics_with_field(mean_field_system,
                initial_field=initial_field, 
                initial_state_list=initial_state_list, 
                start_time=start_time,
                process_tensor_list = [process_tensor, process_tensor],
                control_list=control_list # NEW
                )
fig, axes = plt.subplots(2, figsize=(6,6), sharex=True)
times, fields = mean_field_dynamics_process.field_expectations()
axes[0].plot(times, np.abs(fields)**2)
axes[0].set_ylabel('n/N')
for i, molecule_dynamics in enumerate(mean_field_dynamics_process.system_dynamics):
    times, smsp = molecule_dynamics.expectations(sigma_minus)
    axes[1].plot(times, smsp.real, label=f'Molecule {i}')
axes[1].legend()
axes[1].set_ylabel('Re<sigma^-(t)sigma^+(0)>')
axes[1].set_xlabel('t')
fig.savefig('corr.pdf', bbox_inches='tight')
