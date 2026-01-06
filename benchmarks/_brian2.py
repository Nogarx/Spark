#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import warnings
warnings.filterwarnings('ignore', message='_get_vc_env is private')

import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

import gc
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm
from functools import partial

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def simulate_LIF_model_brian(
        currents,
        t_max,
        dt,
        v_rest,
        v_reset,
        firing_threshold,
        membrane_resistance,
        membrane_time_scale,
        abs_refractory_period,
        delete=True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Simulate a LIF neuron receiving discrete one-step current inputs from spikes.
    """
    b2.set_device('cpp_standalone', build_on_run=False)
    # Reset scope
    b2.start_scope()
    b2.defaultclock.dt = dt*b2.ms
    # Current trace
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
    if delete:
        b2.device.delete(force=True)
    b2.device.reinit()
    times = np.array(times).reshape(-1)
    potentials = np.array(potentials).reshape(-1)
    spikes = np.array(spikes).reshape(-1)
    return times, potentials, spikes

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def simulate_AdEx_model_brian(
        currents,
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
        delete=True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Simulate a AdEx neuron receiving discrete one-step current inputs from spikes.
    """
    b2.set_device('cpp_standalone', build_on_run=False)
    # Reset scope
    b2.start_scope()
    b2.defaultclock.dt = dt*b2.ms
    # Current trace
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
    if delete:
        b2.device.delete(force=True)
    b2.device.reinit()
    times = np.array(times).reshape(-1)
    potentials = np.array(potentials).reshape(-1)
    spikes = np.array(spikes).reshape(-1)
    return times, potentials, spikes

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def simulate_HH_model_brian(
        currents,
        t_max,
        dt,
        c_m,
        e_leak,
        e_k,
        e_na,
        g_leak,
        g_na,
        g_k,
        threshold,
        delete=True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Simulate a HH neuron receiving discrete one-step current inputs from spikes.
    """
    b2.set_device('cpp_standalone', build_on_run=False)
    # Reset scope
    b2.start_scope()
    b2.defaultclock.dt = dt*b2.ms
    # Current trace
    input_current = b2.TimedArray(currents * b2.uA, dt=dt * b2.ms)
    # Build model
    eqs = """
        dm/dt = alpha_m * (1-m) - beta_m * m : 1
        alpha_m = (0.1/mV) * (-v+25*mV) / (exp( (-v+25*mV)/(10*mV) ) - 1) / ms : Hz
        beta_m = 4 * exp(-v/(18*mV)) / ms : Hz

        dn/dt = alpha_n * (1-n) - beta_n * n : 1
        alpha_n = (0.01/mV) * (-v+10*mV) / (exp( (-v+10*mV)/(10*mV) ) - 1) / ms : Hz
        beta_n = 0.125 * exp(-v/(80*mV) ) / ms : Hz

        dh/dt = alpha_h * (1-h) - beta_h * h : 1
        alpha_h = 0.07 * exp( -v/(20*mV) ) / ms : Hz
        beta_h = 1 / (exp( (-v+30*mV)/(10*mV) ) + 1) / ms : Hz
            
        dv/dt = (-g_leak*(v-e_leak) - g_na*(m**3)*h*(v-e_na) - g_k*(n**4)*(v-e_k) + input_current(t))/c_m : volt
    """
    neuron = b2.NeuronGroup(
        1,
        model=eqs, 
        threshold='v > threshold',
        refractory='v > threshold',
        method='exponential_euler'
    )
    # Initialization
    neuron.v = e_leak
    neuron.h = 1
    neuron.m = 0
    neuron.n = 0.5
    # Monitors
    state_monitor = b2.StateMonitor(neuron, ['v', 'm', 'h', 'n'], record=True)
    spike_monitor = b2.SpikeMonitor(neuron)
    # Run simulation
    b2.run(t_max * b2.ms)
    b2.device.build(directory='_hh_cpp', compile=True, run=True, debug=False)
    # Get outputs
    times = state_monitor.t / b2.ms
    spikes = np.array([s / b2.ms for s in spike_monitor.t])
    potentials = state_monitor.v[0] / b2.mV - 70
    # Clear files
    if delete:
        b2.device.delete(force=True)
    b2.device.reinit()
    times = np.array(times).reshape(-1)
    potentials = np.array(potentials).reshape(-1)
    spikes = np.array(spikes).reshape(-1)
    return times, potentials, spikes

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def make_adex_group(units, model_params) -> b2.NeuronGroup:
    eqs = """
        dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T) - R * w)/(tau_m) : volt
        dw/dt = (a*(v-v_rest)-w)/tau_w : amp
    """
    group = b2.NeuronGroup(
        units,
        model=eqs, 
        reset='v=v_reset;w+=b', 
        threshold='v>firing_threshold',
        method='euler',
        namespace=model_params,
    )
    group.v = model_params['v_rest']
    return group

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def connect_neuron_groups(pre, post, units, kernel=None, ker_density=0.5, synapse_strength=1.0) -> b2.Synapses:
    if kernel is None:
        kernel = synapse_strength * np.random.rand(units, units)
        kernel *= (np.random.rand(units, units) < ker_density)
    synapses = b2.Synapses(pre, post, model='w_syn : volt', on_pre='v_post += w_syn')
    src, tgt = kernel.nonzero()
    synapses.connect(i=src, j=tgt)
    synapses.w_syn = kernel[src, tgt] * b2.mV
    return synapses

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def spikes_to_b2_gen(spikes, dt) -> b2.SpikeGeneratorGroup:
    times, indices = np.argwhere(spikes > 0).T
    times = times * dt
    generator =  b2.SpikeGeneratorGroup(spikes.shape[-1], indices, times)
    return generator

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def run_brian_model_cpp(
        build_func,
        sim_repetitions,
        units,
        ker_density,
        synapse_strength,
        freq_spike,
        t_max,
        dt,
        model_params,
        kernels = None,
        delete = True,
        directory='_adex_cpp',
    ):
    spikes = np.random.rand(int(t_max / dt), units) < float(freq_spike * dt)
    # Start scope
    b2.set_device('cpp_standalone', build_on_run=False)
    b2.start_scope()
    b2.defaultclock.dt = dt
    compile_start = time.time()

    # Build network model 
    network = build_func(
        spikes, 
        units, dt, 
        ker_density, 
        synapse_strength, 
        model_params,
        kernels=kernels,
    )
    network.run(t_max)

    # Compile model
    b2.device.build(directory=directory, compile=True, run=False, debug=False)
    compile_end = time.time()
    compile_time = compile_end - compile_start

    # Execute model
    sim_times = []
    for _ in range(sim_repetitions):
        start = time.time()
        b2.device.run(directory=directory, with_output=True)
        end = time.time()
        sim_times.append(end - start)

    # Clear files
    b2.device.reinit()
    if delete:
        b2.device.delete(force=True)

    return compile_time, sim_times

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def run_brian_model_cpp_step(
        build_func,
        sim_repetitions,
        units,
        ker_density,
        synapse_strength,
        freq_spike,
        t_max,
        sim_step,
        dt,
        model_params,
        kernels = None,
        delete = True,
        directory='_adex_cpp',
    ):
    spikes = np.random.rand(int(t_max / dt), units) < float(freq_spike * dt)
    iterations = int(t_max / sim_step)

    # Start scope
    b2.set_device('cpp_standalone', build_on_run=False)
    b2.start_scope()
    b2.defaultclock.dt = dt
    compile_start = time.time()

    # Build network model 
    network = build_func(
        spikes, 
        units, dt, 
        ker_density, 
        synapse_strength, 
        model_params,
        kernels=kernels,
    )
    network.run(sim_step)

    # Compile model
    b2.device.build(directory=directory, compile=True, run=False, debug=False)
    compile_end = time.time()
    compile_time = compile_end - compile_start

    # Execute model
    sim_times = []
    for _ in range(sim_repetitions):
        start = time.time()
        for _ in range(iterations):
            b2.device.run(directory=directory, with_output=True)
        end = time.time()
        sim_times.append(end - start)
        b2.device.reinit()

    # Clear files
    b2.device.reinit()
    if delete:
        b2.device.delete(force=True)


    return compile_time, sim_times

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def run_brian_model_numpy(
        build_func,
        sim_repetitions,
        units,
        ker_density,
        synapse_strength,
        freq_spike,
        t_max,
        sim_step,
        dt,
        model_params,
        kernels = None,
    ):
    spikes = np.random.rand(int(t_max / dt), units) < float(freq_spike * dt)
    iterations = int(t_max / sim_step)

    # Start scope
    b2.prefs.codegen.target = 'numpy'
    b2.set_device('runtime')
    b2.start_scope()
    b2.defaultclock.dt = dt
    compile_start = time.time()

    # Build network model 
    network = build_func(
        spikes, 
        units, dt, 
        ker_density, 
        synapse_strength, 
        model_params,
        kernels=kernels,
    )
    # Compile model
    # NOTE: Here compile time is just building the network object
    compile_end = time.time()
    compile_time = compile_end - compile_start

    # Execute model
    sim_times = []
    network.store()
    for _ in range(sim_repetitions):
        start = time.time()
        for _ in range(iterations):
            network.run(sim_step)
        end = time.time()
        sim_times.append(end - start)
        network.restore()

    del network
    gc.collect()
    b2.device.reinit()

    return compile_time, sim_times

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def adex_model_1(spikes, units, dt, ker_density, synapse_strength, model_params, kernels: dict = {}):
    # Network model 
    layers = ('A', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D')
    edges = [
        ('input', 'A'),
        ('A', 'B1'),
        ('B1', 'B2'),
        ('B2', 'B3'),
        ('B3', 'D'),
        ('A', 'C1'),
        ('C1', 'C2'),
        ('C2', 'C3'),
        ('C3', 'D'),
    ]
    # Build neuron groups
    groups = {
        **{'input': spikes_to_b2_gen(spikes, dt)},
        **{name: make_adex_group(units, model_params) for name in layers},
    }

    # Build synapses
    synapses = [
        connect_neuron_groups(
            pre=groups[pre], 
            post=groups[post],
            units=units,
            kernel=kernels.get((pre,post), default=None) if isinstance(kernels, dict) else None, 
            ker_density=ker_density, 
            synapse_strength=synapse_strength,
        )
        for pre, post in edges
    ]

    # Network
    output_monitor = b2.SpikeMonitor(groups['D'])
    groups_values = groups.values()
    network = b2.Network(
        output_monitor,
        *groups_values,
        *synapses
    )
    return network

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def adex_model_2(spikes, units, dt, ker_density, synapse_strength, model_params, kernels: dict = {}):
    # Network model 
    layers = ('A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D')
    edges = [
        ('input', 'A1'),
        ('input', 'B1'),
        ('input', 'C1'),
        ('A1', 'A2'),
        ('B1', 'B2'),
        ('C1', 'C2'),
        ('A2', 'D'),
        ('B2', 'D'),
        ('C2', 'D'),
    ]
    # Build neuron groups
    groups = {
        **{'input': spikes_to_b2_gen(spikes, dt)},
        **{name: make_adex_group(units, model_params) for name in layers},
    }

    # Build synapses
    synapses = [
        connect_neuron_groups(
            pre=groups[pre], 
            post=groups[post],
            units=units,
            kernel=kernels.get((pre,post), default=None) if isinstance(kernels, dict) else None, 
            ker_density=ker_density, 
            synapse_strength=synapse_strength,
        )
        for pre, post in edges
    ]

    # Network
    output_monitor = b2.SpikeMonitor(groups['D'])
    groups_values = groups.values()
    network = b2.Network(
        output_monitor,
        *groups_values,
        *synapses
    )
    return network

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def adex_model_3(spikes, units, dt, ker_density, synapse_strength, model_params, kernels: dict = {}):
    # Network model 
    layers = ('A', 'B', 'C', 'D', 'E', 'F', 'G')
    edges = [
        ('input', 'A'),
        ('A', 'B'),
        ('A', 'C'),
        ('A', 'D'),
        ('A', 'E'),
        ('A', 'F'),
        ('B', 'G'),
        ('C', 'G'),
        ('D', 'G'),
        ('E', 'G'),
        ('F', 'G'),
    ]
    # Build neuron groups
    groups = {
        **{'input': spikes_to_b2_gen(spikes, dt)},
        **{name: make_adex_group(units, model_params) for name in layers},
    }

    # Build synapses
    synapses = [
        connect_neuron_groups(
            pre=groups[pre], 
            post=groups[post],
            units=units,
            kernel=kernels.get((pre,post), default=None) if isinstance(kernels, dict) else None, 
            ker_density=ker_density, 
            synapse_strength=synapse_strength,
        )
        for pre, post in edges
    ]

    # Network
    output_monitor = b2.SpikeMonitor(groups['G'])
    groups_values = groups.values()
    network = b2.Network(
        output_monitor,
        *groups_values,
        *synapses
    )
    return network

#-----------------------------------------------------------------------------------------------------------------------------------------------#

brian2_adex_performance_model_1_cpp = partial(run_brian_model_cpp, build_func=adex_model_1)
brian2_adex_performance_model_1_cpp_step = partial(run_brian_model_cpp_step, build_func=adex_model_1)
brian2_adex_performance_model_1_numpy = partial(run_brian_model_numpy, build_func=adex_model_1)

brian2_adex_performance_model_2_cpp = partial(run_brian_model_cpp, build_func=adex_model_2)
brian2_adex_performance_model_2_cpp_step = partial(run_brian_model_cpp_step, build_func=adex_model_2)
brian2_adex_performance_model_2_numpy = partial(run_brian_model_numpy, build_func=adex_model_2)

brian2_adex_performance_model_3_cpp = partial(run_brian_model_cpp, build_func=adex_model_3)
brian2_adex_performance_model_3_cpp_step = partial(run_brian_model_cpp_step, build_func=adex_model_3)
brian2_adex_performance_model_3_numpy = partial(run_brian_model_numpy, build_func=adex_model_3)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
