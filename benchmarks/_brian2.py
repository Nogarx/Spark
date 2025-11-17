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
		method='exact'
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
	b2.device.delete(force=True)
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
		method='rk2'
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
	b2.device.delete(force=True)
	b2.device.reinit()
	return times, potentials, spikes

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def brian_adex_performance_non_interactive(
        units,
        ker_density,
        synapse_strength,
        t_max,
        k_time,
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
    ):
	# Build currents
    currents = []
    iters = int(t_max / k_time)
    k_steps = int(k_time / dt)
    for _ in range(iters):
        idx = np.random.rand(units) < 0.05
        it_current = np.sum(np.random.rand(units,units)[:,idx], axis=1)
        for _ in range(k_steps):
            currents.append(it_current)
    currents = np.array(currents)

    b2.set_device('cpp_standalone', build_on_run=False)
    # Reset scope
    b2.start_scope()
    input_current = b2.TimedArray(currents * b2.nA, dt=dt)

    # Neurons
    eqs_in = """
        dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * input_current(t,i) - R * w)/(tau_m) : volt
        dw/dt = (a*(v-v_rest)-w)/tau_w : amp
    """
    eqs = """
        dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T) - R * w)/(tau_m) : volt
        dw/dt = (a*(v-v_rest)-w)/tau_w : amp
    """
    n_A1 = b2.NeuronGroup(
        units,
        model=eqs_in, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_A1.v = v_rest
    n_B1 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_B1.v = v_rest
    n_B2 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_B2.v = v_rest
    n_B3 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_B3.v = v_rest
    n_C1 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_C1.v = v_rest
    n_C2 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_C2.v = v_rest
    n_C3 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_C3.v = v_rest
    n_D1 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_D1.v = v_rest

    # Synapses
    ker_A1_B1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_A1_B1 = b2.Synapses(n_A1, n_B1, model='w_syn : volt', on_pre='v += w_syn')
    s_A1_B1_src, s_A1_B1_tgt = ker_A1_B1.nonzero()
    s_A1_B1.connect(i=s_A1_B1_src, j=s_A1_B1_tgt)
    s_A1_B1.w_syn = ker_A1_B1[s_A1_B1_src, s_A1_B1_tgt] * b2.mV

    ker_B1_B2 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_B1_B2 = b2.Synapses(n_B1, n_B2, model='w_syn : volt', on_pre='v += w_syn')
    s_B1_B2_src, s_B1_B2_tgt = ker_B1_B2.nonzero()
    s_B1_B2.connect(i=s_B1_B2_src, j=s_B1_B2_tgt)
    s_B1_B2.w_syn = ker_B1_B2[s_B1_B2_src, s_B1_B2_tgt] * b2.mV

    ker_B2_B3 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_B2_B3 = b2.Synapses(n_B2, n_B3, model='w_syn : volt', on_pre='v += w_syn')
    s_B2_B3_src, s_B2_B3_tgt = ker_B2_B3.nonzero()
    s_B2_B3.connect(i=s_B2_B3_src, j=s_B2_B3_tgt)
    s_B2_B3.w_syn = ker_B2_B3[s_B2_B3_src, s_B2_B3_tgt] * b2.mV

    ker_B3_D1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_B3_D1 = b2.Synapses(n_B3, n_D1, model='w_syn : volt', on_pre='v += w_syn')
    s_B3_D1_src, s_B3_D1_tgt = ker_B3_D1.nonzero()
    s_B3_D1.connect(i=s_B3_D1_src, j=s_B3_D1_tgt)
    s_B3_D1.w_syn = ker_B3_D1[s_B3_D1_src, s_B3_D1_tgt] * b2.mV

    ker_A1_C1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_A1_C1 = b2.Synapses(n_A1, n_C1, model='w_syn : volt', on_pre='v += w_syn')
    s_A1_C1_src, s_A1_C1_tgt = ker_A1_C1.nonzero()
    s_A1_C1.connect(i=s_A1_C1_src, j=s_A1_C1_tgt)
    s_A1_C1.w_syn = ker_A1_C1[s_A1_C1_src, s_A1_C1_tgt] * b2.mV

    ker_C1_C2 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_C1_C2 = b2.Synapses(n_C1, n_C2, model='w_syn : volt', on_pre='v += w_syn')
    s_C1_C2_src, s_C1_C2_tgt = ker_C1_C2.nonzero()
    s_C1_C2.connect(i=s_C1_C2_src, j=s_C1_C2_tgt)
    s_C1_C2.w_syn = ker_C1_C2[s_C1_C2_src, s_C1_C2_tgt] * b2.mV

    ker_C2_C3 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_C2_C3 = b2.Synapses(n_C2, n_C3, model='w_syn : volt', on_pre='v += w_syn')
    s_C2_C3_src, s_C2_C3_tgt = ker_C2_C3.nonzero()
    s_C2_C3.connect(i=s_C2_C3_src, j=s_C2_C3_tgt)
    s_C2_C3.w_syn = ker_C2_C3[s_C2_C3_src, s_C2_C3_tgt] * b2.mV

    ker_C3_D1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_C3_D1 = b2.Synapses(n_C3, n_D1, model='w_syn : volt', on_pre='v += w_syn')
    s_C3_D1_src, s_C3_D1_tgt = ker_C3_D1.nonzero()
    s_C3_D1.connect(i=s_C3_D1_src, j=s_C3_D1_tgt)
    s_C3_D1.w_syn = ker_C3_D1[s_C3_D1_src, s_C3_D1_tgt] * b2.mV

    b2.run(t_max)
    b2.device.build(directory='_adex_cpp', compile=True, run=False, debug=False)

    times = []
    for _ in range(10):
        start = time.time()
        b2.device.run()
        end = time.time()
        times.append(end - start)

    # Clear files
    b2.device.delete(force=True)
    b2.device.reinit()

    return times


#-----------------------------------------------------------------------------------------------------------------------------------------------#

def brian_adex_performance_interactive_mock(
        units,
        ker_density,
        synapse_strength,
        t_max,
        k_time,
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
    ):
    b2.set_device('cpp_standalone', build_on_run=False)
    # Reset scope
    b2.start_scope()
    # Build discrete current trace from spikes
    iters = int(t_max / k_time)
    k_steps = int(k_time / dt)
    input_current = b2.TimedArray(np.zeros((k_steps, units)) * b2.nA, dt=dt)

    # Neurons
    eqs_in = """
        dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * input_current(t,i) - R * w)/(tau_m) : volt
        dw/dt = (a*(v-v_rest)-w)/tau_w : amp
    """
    eqs = """
        dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T) - R * w)/(tau_m) : volt
        dw/dt = (a*(v-v_rest)-w)/tau_w : amp
    """
    n_A1 = b2.NeuronGroup(
        units,
        model=eqs_in, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_A1.v = v_rest
    n_B1 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_B1.v = v_rest
    n_B2 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_B2.v = v_rest
    n_B3 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_B3.v = v_rest
    n_C1 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_C1.v = v_rest
    n_C2 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_C2.v = v_rest
    n_C3 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_C3.v = v_rest
    n_D1 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_D1.v = v_rest

    # Synapses
    ker_A1_B1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_A1_B1 = b2.Synapses(n_A1, n_B1, model='w_syn : volt', on_pre='v += w_syn')
    s_A1_B1_src, s_A1_B1_tgt = ker_A1_B1.nonzero()
    s_A1_B1.connect(i=s_A1_B1_src, j=s_A1_B1_tgt)
    s_A1_B1.w_syn = ker_A1_B1[s_A1_B1_src, s_A1_B1_tgt] * b2.mV

    ker_B1_B2 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_B1_B2 = b2.Synapses(n_B1, n_B2, model='w_syn : volt', on_pre='v += w_syn')
    s_B1_B2_src, s_B1_B2_tgt = ker_B1_B2.nonzero()
    s_B1_B2.connect(i=s_B1_B2_src, j=s_B1_B2_tgt)
    s_B1_B2.w_syn = ker_B1_B2[s_B1_B2_src, s_B1_B2_tgt] * b2.mV

    ker_B2_B3 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_B2_B3 = b2.Synapses(n_B2, n_B3, model='w_syn : volt', on_pre='v += w_syn')
    s_B2_B3_src, s_B2_B3_tgt = ker_B2_B3.nonzero()
    s_B2_B3.connect(i=s_B2_B3_src, j=s_B2_B3_tgt)
    s_B2_B3.w_syn = ker_B2_B3[s_B2_B3_src, s_B2_B3_tgt] * b2.mV

    ker_B3_D1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_B3_D1 = b2.Synapses(n_B3, n_D1, model='w_syn : volt', on_pre='v += w_syn')
    s_B3_D1_src, s_B3_D1_tgt = ker_B3_D1.nonzero()
    s_B3_D1.connect(i=s_B3_D1_src, j=s_B3_D1_tgt)
    s_B3_D1.w_syn = ker_B3_D1[s_B3_D1_src, s_B3_D1_tgt] * b2.mV

    ker_A1_C1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_A1_C1 = b2.Synapses(n_A1, n_C1, model='w_syn : volt', on_pre='v += w_syn')
    s_A1_C1_src, s_A1_C1_tgt = ker_A1_C1.nonzero()
    s_A1_C1.connect(i=s_A1_C1_src, j=s_A1_C1_tgt)
    s_A1_C1.w_syn = ker_A1_C1[s_A1_C1_src, s_A1_C1_tgt] * b2.mV

    ker_C1_C2 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_C1_C2 = b2.Synapses(n_C1, n_C2, model='w_syn : volt', on_pre='v += w_syn')
    s_C1_C2_src, s_C1_C2_tgt = ker_C1_C2.nonzero()
    s_C1_C2.connect(i=s_C1_C2_src, j=s_C1_C2_tgt)
    s_C1_C2.w_syn = ker_C1_C2[s_C1_C2_src, s_C1_C2_tgt] * b2.mV

    ker_C2_C3 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_C2_C3 = b2.Synapses(n_C2, n_C3, model='w_syn : volt', on_pre='v += w_syn')
    s_C2_C3_src, s_C2_C3_tgt = ker_C2_C3.nonzero()
    s_C2_C3.connect(i=s_C2_C3_src, j=s_C2_C3_tgt)
    s_C2_C3.w_syn = ker_C2_C3[s_C2_C3_src, s_C2_C3_tgt] * b2.mV

    ker_C3_D1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_C3_D1 = b2.Synapses(n_C3, n_D1, model='w_syn : volt', on_pre='v += w_syn')
    s_C3_D1_src, s_C3_D1_tgt = ker_C3_D1.nonzero()
    s_C3_D1.connect(i=s_C3_D1_src, j=s_C3_D1_tgt)
    s_C3_D1.w_syn = ker_C3_D1[s_C3_D1_src, s_C3_D1_tgt] * b2.mV

    for _ in range(iters):
        idx = np.random.rand(units) < 0.05
        it_current = np.sum(np.random.rand(units,units)[:,idx], axis=1).reshape(1,-1)
        input_current = b2.TimedArray(np.repeat(it_current, k_steps, axis=0) * b2.nA, dt=dt)
        b2.run(k_time)

    b2.device.build(directory='_adex_cpp', compile=True, run=False, debug=False)

    times = []
    for _ in range(10):
        start = time.time()
        b2.device.run()
        end = time.time()
        times.append(end - start)

    # Clear files
    b2.device.delete(force=True)
    b2.device.reinit()

    return times

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def brian_adex_performance_interactive_numpy_mock(
        units,
        ker_density,
        synapse_strength,
        t_max,
        k_time,
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
    ):
    b2.prefs.codegen.target = 'numpy'
    b2.set_device('runtime')
    # Reset scope
    b2.start_scope()
    # Build discrete current trace from spikes
    iters = int(t_max / k_time)
    k_steps = int(k_time / dt)
    input_current = b2.TimedArray(np.zeros((k_steps, units)) * b2.nA, dt=dt)

    # Neurons
    eqs_in = """
        dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * input_current(t,i) - R * w)/(tau_m) : volt
        dw/dt = (a*(v-v_rest)-w)/tau_w : amp
    """
    eqs = """
        dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T) - R * w)/(tau_m) : volt
        dw/dt = (a*(v-v_rest)-w)/tau_w : amp
    """
    n_A1 = b2.NeuronGroup(
        units,
        model=eqs_in, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_A1.v = v_rest
    n_B1 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_B1.v = v_rest
    n_B2 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_B2.v = v_rest
    n_B3 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_B3.v = v_rest
    n_C1 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_C1.v = v_rest
    n_C2 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_C2.v = v_rest
    n_C3 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_C3.v = v_rest
    n_D1 = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler'
    )
    n_D1.v = v_rest

    # Synapses
    ker_A1_B1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_A1_B1 = b2.Synapses(n_A1, n_B1, model='w_syn : volt', on_pre='v += w_syn')
    s_A1_B1_src, s_A1_B1_tgt = ker_A1_B1.nonzero()
    s_A1_B1.connect(i=s_A1_B1_src, j=s_A1_B1_tgt)
    s_A1_B1.w_syn = ker_A1_B1[s_A1_B1_src, s_A1_B1_tgt] * b2.mV

    ker_B1_B2 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_B1_B2 = b2.Synapses(n_B1, n_B2, model='w_syn : volt', on_pre='v += w_syn')
    s_B1_B2_src, s_B1_B2_tgt = ker_B1_B2.nonzero()
    s_B1_B2.connect(i=s_B1_B2_src, j=s_B1_B2_tgt)
    s_B1_B2.w_syn = ker_B1_B2[s_B1_B2_src, s_B1_B2_tgt] * b2.mV

    ker_B2_B3 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_B2_B3 = b2.Synapses(n_B2, n_B3, model='w_syn : volt', on_pre='v += w_syn')
    s_B2_B3_src, s_B2_B3_tgt = ker_B2_B3.nonzero()
    s_B2_B3.connect(i=s_B2_B3_src, j=s_B2_B3_tgt)
    s_B2_B3.w_syn = ker_B2_B3[s_B2_B3_src, s_B2_B3_tgt] * b2.mV

    ker_B3_D1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_B3_D1 = b2.Synapses(n_B3, n_D1, model='w_syn : volt', on_pre='v += w_syn')
    s_B3_D1_src, s_B3_D1_tgt = ker_B3_D1.nonzero()
    s_B3_D1.connect(i=s_B3_D1_src, j=s_B3_D1_tgt)
    s_B3_D1.w_syn = ker_B3_D1[s_B3_D1_src, s_B3_D1_tgt] * b2.mV

    ker_A1_C1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_A1_C1 = b2.Synapses(n_A1, n_C1, model='w_syn : volt', on_pre='v += w_syn')
    s_A1_C1_src, s_A1_C1_tgt = ker_A1_C1.nonzero()
    s_A1_C1.connect(i=s_A1_C1_src, j=s_A1_C1_tgt)
    s_A1_C1.w_syn = ker_A1_C1[s_A1_C1_src, s_A1_C1_tgt] * b2.mV

    ker_C1_C2 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_C1_C2 = b2.Synapses(n_C1, n_C2, model='w_syn : volt', on_pre='v += w_syn')
    s_C1_C2_src, s_C1_C2_tgt = ker_C1_C2.nonzero()
    s_C1_C2.connect(i=s_C1_C2_src, j=s_C1_C2_tgt)
    s_C1_C2.w_syn = ker_C1_C2[s_C1_C2_src, s_C1_C2_tgt] * b2.mV

    ker_C2_C3 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_C2_C3 = b2.Synapses(n_C2, n_C3, model='w_syn : volt', on_pre='v += w_syn')
    s_C2_C3_src, s_C2_C3_tgt = ker_C2_C3.nonzero()
    s_C2_C3.connect(i=s_C2_C3_src, j=s_C2_C3_tgt)
    s_C2_C3.w_syn = ker_C2_C3[s_C2_C3_src, s_C2_C3_tgt] * b2.mV

    ker_C3_D1 = (synapse_strength * np.random.rand(units, units)) * (np.random.rand(units, units) < ker_density)
    s_C3_D1 = b2.Synapses(n_C3, n_D1, model='w_syn : volt', on_pre='v += w_syn')
    s_C3_D1_src, s_C3_D1_tgt = ker_C3_D1.nonzero()
    s_C3_D1.connect(i=s_C3_D1_src, j=s_C3_D1_tgt)
    s_C3_D1.w_syn = ker_C3_D1[s_C3_D1_src, s_C3_D1_tgt] * b2.mV



    times = []
    for _ in range(10):
        start = time.time()
        for _ in range(iters):
            idx = np.random.rand(units) < 0.05
            it_current = np.sum(np.random.rand(units,units)[:,idx], axis=1).reshape(1,-1)
            input_current = b2.TimedArray(np.repeat(it_current, k_steps, axis=0) * b2.nA, dt=dt)
            b2.run(k_time)
        end = time.time()
        times.append(end - start)

    # Clear files
    b2.device.delete(force=True)
    b2.device.reinit()

    return times

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################