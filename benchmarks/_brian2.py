#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import warnings
warnings.filterwarnings('ignore', message='_get_vc_env is private')

import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def spikes_to_current(spike_times, k=1.0, t_max=100.0, dt=0.1):
	"""
		Convert spike times into a discrete current trace.
	"""
	times = np.arange(0, t_max + dt, dt)
	currents = np.zeros_like(times)
	for s in spike_times:
		idx = int(round(s / dt))
		if 0 <= idx < len(currents):
			currents[idx] += k
	return times, currents

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def simulate_LIF_model_brian(
		spike_times,
		synapse_strength,
		t_max,
		dt,
		v_rest,
		v_reset,
		firing_threshold,
		membrane_resistance,
		membrane_time_scale,
		abs_refractory_period,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
		Simulate a LIF neuron receiving discrete one-step current inputs from spikes.
	"""
	b2.set_device('cpp_standalone', build_on_run=False)
	# Reset scope
	b2.start_scope()
	# Build discrete current trace from spikes
	times, currents = spikes_to_current(spike_times, k=synapse_strength, t_max=t_max, dt=dt)
	input_current = b2.TimedArray(currents * b2.nA, dt=dt * b2.ms)
	# Build model
	eqs = """
		dv/dt = (-(v - v_rest) + membrane_resistance * input_current(t)) / membrane_time_scale : volt (unless refractory)
	"""
	neuron = b2.NeuronGroup(
		1, 
		model=eqs, 
		reset='v=v_reset', 
		threshold='v>firing_threshold',
		refractory=abs_refractory_period, 
		method='linear'
	)
	# Initialization
	neuron.v = v_rest
	# Monitors
	state_monitor = b2.StateMonitor(neuron, 'v', record=True)
	spike_monitor = b2.SpikeMonitor(neuron)
	# Run simulation
	b2.run(t_max * b2.ms)
	b2.device.build(directory='_lif_cpp', compile=True, run=True, debug=False)
	# Get outputs
	times = state_monitor.t / b2.ms
	spikes = np.array([s / b2.ms for s in spike_monitor.t])
	potentials = state_monitor.v[0] / b2.mV
	# Clear files
	b2.device.delete()
	b2.device.reinit()
	return times, potentials, spikes

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def simulate_AdEx_model_brian(
		spike_times,
		synapse_strength,
		t_max,
		dt,
		tau_m,
		R,
		v_rest,
		v_reset,
		v_rheobase,
		a,
		b,
		firing_threshold,
		delta_T,
		tau_w,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
		Simulate a AdEx neuron receiving discrete one-step current inputs from spikes.
	"""
	b2.set_device('cpp_standalone', build_on_run=False)
	# Reset scope
	b2.start_scope()
	# Build discrete current trace from spikes
	times, currents = spikes_to_current(spike_times, k=synapse_strength, t_max=t_max, dt=dt)
	input_current = b2.TimedArray(currents * b2.nA, dt=dt * b2.ms)
	# Build model
	eqs = """
		dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * input_current(t) - R * w)/(tau_m) : volt
		dw/dt = (a*(v-v_rest)-w)/tau_w : amp
	"""
	neuron = b2.NeuronGroup(
		1,
		model=eqs, 
		reset='v=v_reset;w+=b', 
		threshold='v>firing_threshold',
		method='euler'
	)
	# Initialization
	neuron.v = v_rest
	neuron.w = 0.0 * b2.pA
	# Monitors
	state_monitor = b2.StateMonitor(neuron, ['v', 'w'], record=True)
	spike_monitor = b2.SpikeMonitor(neuron)
	# Run simulation
	b2.run(t_max * b2.ms)
	b2.device.build(directory='_adex_cpp', compile=True, run=True, debug=False)
	# Get outputs
	times = state_monitor.t / b2.ms
	spikes = np.array([s / b2.ms for s in spike_monitor.t])
	potentials = state_monitor.v[0] / b2.mV
	adaptations = state_monitor.w[0] / b2.pA
	# Clear files
	b2.device.delete()
	b2.device.reinit()
	return times, potentials, spikes

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################