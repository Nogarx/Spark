#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import spark
import time
from functools import partial
from _hh import HodgkinHuxleyNeuron, HodgkinHuxleySomaConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def spike_events_to_array(spikes, max_t, dt=0.1) -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0, max_t, int(max_t/dt)+1)
    array = np.zeros(times.shape)
    for s in spikes:
        array[int(s / dt)] = 1
    return times, array

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.jit
def run_model_one_steps(graph, state, **inputs)-> tuple[dict, nnx.GraphState | nnx.VariableState]:
    model = spark.merge(graph, state)
    outputs = model(**inputs)
    _, state = spark.split((model))
    return outputs, state

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@partial(jax.jit, static_argnames='k')
def run_model_k_steps(graph, state, k, **inputs)-> tuple[dict, nnx.GraphState | nnx.VariableState]:
    model = spark.merge(graph, state)
    for i in range(k):
        outputs = model(**inputs)
    _, state = spark.split((model))
    return outputs, state

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def simulate_model_spark(
    spike_times,
    build_func,
    offset,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Initialize model
    model = build_func(**kwargs)
    # Map spikes
    times, in_spikes = spike_events_to_array(spike_times, 100, dt=kwargs['dt'])
    # Run simulation
    graph, state = spark.split((model))
    spikes, potentials = [], []
    for s in in_spikes:
        in_spikes = spark.SpikeArray( jnp.array([s]) )
        outputs, state = run_model_one_steps(graph, state, in_spikes=in_spikes)
        spikes.append(np.array(outputs['out_spikes'].value))
        model = spark.merge(graph, state)
        potentials.append(np.array(model.soma.potential.value) + offset)
    spikes = [t*kwargs['dt'] for t, s in enumerate(spikes) if s == 1]
    return times, potentials, spikes


#-----------------------------------------------------------------------------------------------------------------------------------------------#

def build_LIF_model(
    dt,
    synapse_strength,
    potential_rest,
    potential_reset,
    potential_tau,
    resistance,
    threshold,
    cooldown,
) -> spark.nn.neurons.LIFNeuron:
    # NOTE: We need to substract 1 timestep from the cooldown to match the Brian2 dynamic 
    # Since we update the refractory the period at the end of the cycle, Brian2 does it at the begining. 
    lif_neuron = spark.nn.neurons.LIFNeuron(
        _s_units = (1,),
        _s_dt = dt,
        _s_dtype = jnp.float16,
        inhibitory_rate = 0.0,
        soma_config = spark.nn.somas.StrictRefractoryLeakySomaConfig(
            potential_rest = potential_rest,
            potential_reset = potential_reset,
            potential_tau = potential_tau,
            resistance = resistance,
            threshold = threshold,
            cooldown = cooldown - dt, 
        ),
        synapses_config = spark.nn.synapses.LinearSynapsesConfig(
            units = (1,),
            kernel_initializer = spark.nn.initializers.ConstantInitializerConfig(scale=synapse_strength),
        ),
        delays_config = None,
        learning_rule_config = None,
    )
    lif_neuron(in_spikes=spark.SpikeArray( jnp.zeros((1,)) ))
    return lif_neuron

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def build_AdEx_model(
	dt,
	synapse_strength,
	potential_rest,
	potential_reset,
	potential_tau,
	resistance, # MΩ -> GΩ
	threshold,
	rheobase_threshold,
	spike_slope,
	adaptation_tau,
	adaptation_delta,
	adaptation_subthreshold,	
) -> spark.nn.neurons.AdExNeuron:
	# NOTE: We need to substract 1 timestep from the cooldown to match the Brian2 dynamic 
	# Since we update the refractory the period at the end of the cycle, Brian2 does it at the begining. 
	adex_neuron = spark.nn.neurons.AdExNeuron(
		_s_units = (1,),
		_s_dt = dt,
		_s_dtype = jnp.float16,
		inhibitory_rate = 0.0,
		soma_config = spark.nn.somas.AdaptiveExponentialSomaConfig(
			potential_rest = potential_rest,
			potential_reset = potential_reset,
			potential_tau = potential_tau,
			resistance = resistance,
			threshold = threshold,
			rheobase_threshold = rheobase_threshold,
			spike_slope = spike_slope,
			adaptation_tau = adaptation_tau,
			adaptation_delta = adaptation_delta,
			adaptation_subthreshold = adaptation_subthreshold,
		),
		synapses_config = spark.nn.synapses.LinearSynapsesConfig(
			units = (1,),
			kernel_initializer = spark.nn.initializers.ConstantInitializerConfig(scale=synapse_strength),
		),
		delays_config = None,
		learning_rule_config = None,
	)
	adex_neuron(in_spikes=spark.SpikeArray( jnp.zeros((1,)) ))
	return adex_neuron

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def build_HH_model(
    dt,
    synapse_strength,
    c_m,
    e_leak,
    e_na,
    e_k,
    g_leak,
    g_na,
    g_k,
    threshold,
) -> HodgkinHuxleyNeuron:
    hh_neuron = HodgkinHuxleyNeuron(
        _s_units = (1,),
        _s_dt = dt,
        _s_dtype = jnp.float16,
        inhibitory_rate = 0.0,
        soma_config = HodgkinHuxleySomaConfig(
            c_m = c_m,
            e_leak = e_leak,
            e_na = e_na,
            e_k = e_k,
            g_leak = g_leak,
            g_na = g_na,
            g_k = g_k,
            threshold = threshold, 
        ),
        synapses_config = spark.nn.synapses.LinearSynapsesConfig(
            units = (1,),
            kernel_initializer = spark.nn.initializers.ConstantInitializerConfig(scale=synapse_strength),
        ),
        delays_config = None,
        learning_rule_config = None,
    )
    hh_neuron(in_spikes=spark.SpikeArray( jnp.zeros((1,)) ))
    return hh_neuron

#-----------------------------------------------------------------------------------------------------------------------------------------------#

simulate_LIF_model_spark = partial(simulate_model_spark, build_func=build_LIF_model)

simulate_AdEx_model_spark = partial(simulate_model_spark, build_func=build_AdEx_model)

simulate_HH_model_spark = partial(simulate_model_spark, build_func=build_HH_model)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def adex_brain_model_1_config(
	units,
	ker_density,
	dt,
	synapse_strength,
	potential_rest,
	potential_reset,
	potential_tau,
	resistance, # MΩ -> GΩ
	threshold,
	rheobase_threshold,
	spike_slope,
	adaptation_tau,
	adaptation_delta,
	adaptation_subthreshold,
) -> spark.nn.BrainConfig:

	adex_config = spark.nn.neurons.AdExNeuronConfig(
		_s_units = (units,),
		_s_dt = dt,
		_s_dtype = jnp.float16,
		inhibitory_rate = 0.0,
		soma_config = spark.nn.somas.AdaptiveExponentialSomaConfig(
			potential_rest = potential_rest,
			potential_reset = potential_reset,
			potential_tau = potential_tau,
			resistance = resistance,
			threshold = threshold,
			rheobase_threshold = rheobase_threshold,
			spike_slope = spike_slope,
			adaptation_tau = adaptation_tau,
			adaptation_delta = adaptation_delta,
			adaptation_subthreshold = adaptation_subthreshold,
		),
		synapses_config = spark.nn.synapses.LinearSynapsesConfig(
			units = (units,),
			kernel_initializer = spark.nn.initializers.SparseUniformInitializerConfig(
				density = ker_density,
				scale = synapse_strength
			),
		),
		delays_config = None,
		learning_rule_config = None,
	)

	def adex_specs(name, origin, port) -> spark.ModuleSpecs:
		return spark.ModuleSpecs(
			name = name, 
			module_cls = spark.nn.neurons.AdExNeuron, 
			inputs = {
				'in_spikes': [
					spark.PortMap(origin=o, port=p) for o, p in zip(origin, port)
				]
			},
			config = adex_config
		)
	
	input_map = {
		'spikes': spark.InputSpec(
			payload_type=spark.FloatArray, 
			shape=(units,), 
			dtype=jnp.float16,
		)
	}
	output_map = {
		'spikes': {
			'input': spark.PortMap(
				origin='n_D1',
				port='out_spikes'
			),
			'spec': spark.OutputSpec(
				payload_type=spark.SpikeArray,
				shape=(units,),
				dtype=jnp.float16
			)
		}
	}
	modules_map = {
		'n_A1': adex_specs('n_A1', ['__call__'], ['spikes']),
		'n_B1': adex_specs('n_B1', ['n_A1'], ['out_spikes']),
		'n_B2': adex_specs('n_B2', ['n_B1'], ['out_spikes']),
		'n_B3': adex_specs('n_B3', ['n_B2'], ['out_spikes']),
		'n_C1': adex_specs('n_C1', ['n_A1'], ['out_spikes']),
		'n_C2': adex_specs('n_C2', ['n_C1'], ['out_spikes']),
		'n_C3': adex_specs('n_C3', ['n_C2'], ['out_spikes']),
		'n_D1': adex_specs('n_D1', ['n_B3', 'n_C3'], ['out_spikes', 'out_spikes']),
	}

	return spark.nn.BrainConfig(input_map=input_map, output_map=output_map, modules_map=modules_map)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def adex_brain_model_2_config(
	units,
	ker_density,
	dt,
	synapse_strength,
	potential_rest,
	potential_reset,
	potential_tau,
	resistance, # MΩ -> GΩ
	threshold,
	rheobase_threshold,
	spike_slope,
	adaptation_tau,
	adaptation_delta,
	adaptation_subthreshold,
) -> spark.nn.BrainConfig:

	adex_config = spark.nn.neurons.AdExNeuronConfig(
		_s_units = (units,),
		_s_dt = dt,
		_s_dtype = jnp.float16,
		inhibitory_rate = 0.0,
		soma_config = spark.nn.somas.AdaptiveExponentialSomaConfig(
			potential_rest = potential_rest,
			potential_reset = potential_reset,
			potential_tau = potential_tau,
			resistance = resistance,
			threshold = threshold,
			rheobase_threshold = rheobase_threshold,
			spike_slope = spike_slope,
			adaptation_tau = adaptation_tau,
			adaptation_delta = adaptation_delta,
			adaptation_subthreshold = adaptation_subthreshold,
		),
		synapses_config = spark.nn.synapses.LinearSynapsesConfig(
			units = (units,),
			kernel_initializer = spark.nn.initializers.SparseUniformInitializerConfig(
				density = ker_density,
				scale = synapse_strength
			),
		),
		delays_config = None,
		learning_rule_config = None,
	)

	def adex_specs(name, origin, port) -> spark.ModuleSpecs:
		return spark.ModuleSpecs(
			name = name, 
			module_cls = spark.nn.neurons.AdExNeuron, 
			inputs = {
				'in_spikes': [
					spark.PortMap(origin=o, port=p) for o, p in zip(origin, port)
				]
			},
			config = adex_config
		)
	
	input_map = {
		'spikes': spark.InputSpec(
			payload_type=spark.FloatArray, 
			shape=(units,), 
			dtype=jnp.float16,
		)
	}
	output_map = {
		'spikes': {
			'input': spark.PortMap(
				origin='n_D1',
				port='out_spikes'
			),
			'spec': spark.OutputSpec(
				payload_type=spark.SpikeArray,
				shape=(units,),
				dtype=jnp.float16
			)
		}
	}
	modules_map = {
		'n_A1': adex_specs('n_A1', ['__call__'], ['spikes']),
		'n_A2': adex_specs('n_A2', ['n_A1'], ['out_spikes']),
		'n_B1': adex_specs('n_B1', ['__call__'], ['spikes']),
		'n_B2': adex_specs('n_B2', ['n_B1'], ['out_spikes']),
		'n_C1': adex_specs('n_C1', ['__call__'], ['spikes']),
		'n_C2': adex_specs('n_C2', ['n_C1'], ['out_spikes']),
		'n_D1': adex_specs('n_D1', ['n_A2', 'n_B2', 'n_C2'], ['out_spikes', 'out_spikes', 'out_spikes']),
	}

	return spark.nn.BrainConfig(input_map=input_map, output_map=output_map, modules_map=modules_map)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def adex_brain_model_3_config(
	units,
	ker_density,
	dt,
	synapse_strength,
	potential_rest,
	potential_reset,
	potential_tau,
	resistance, # MΩ -> GΩ
	threshold,
	rheobase_threshold,
	spike_slope,
	adaptation_tau,
	adaptation_delta,
	adaptation_subthreshold,
) -> spark.nn.BrainConfig:

	adex_config = spark.nn.neurons.AdExNeuronConfig(
		_s_units = (units,),
		_s_dt = dt,
		_s_dtype = jnp.float16,
		inhibitory_rate = 0.0,
		soma_config = spark.nn.somas.AdaptiveExponentialSomaConfig(
			potential_rest = potential_rest,
			potential_reset = potential_reset,
			potential_tau = potential_tau,
			resistance = resistance,
			threshold = threshold,
			rheobase_threshold = rheobase_threshold,
			spike_slope = spike_slope,
			adaptation_tau = adaptation_tau,
			adaptation_delta = adaptation_delta,
			adaptation_subthreshold = adaptation_subthreshold,
		),
		synapses_config = spark.nn.synapses.LinearSynapsesConfig(
			units = (units,),
			kernel_initializer = spark.nn.initializers.SparseUniformInitializerConfig(
				density = ker_density,
				scale = synapse_strength
			),
		),
		delays_config = None,
		learning_rule_config = None,
	)

	def adex_specs(name, origin, port) -> spark.ModuleSpecs:
		return spark.ModuleSpecs(
			name = name, 
			module_cls = spark.nn.neurons.AdExNeuron, 
			inputs = {
				'in_spikes': [
					spark.PortMap(origin=o, port=p) for o, p in zip(origin, port)
				]
			},
			config = adex_config
		)
	
	input_map = {
		'spikes': spark.InputSpec(
			payload_type=spark.FloatArray, 
			shape=(units,), 
			dtype=jnp.float16,
		)
	}
	output_map = {
		'spikes': {
			'input': spark.PortMap(
				origin='n_D1',
				port='out_spikes'
			),
			'spec': spark.OutputSpec(
				payload_type=spark.SpikeArray,
				shape=(units,),
				dtype=jnp.float16
			)
		}
	}
	modules_map = {
		'n_A1': adex_specs('n_A1', ['__call__'], ['spikes']),
		'n_B1': adex_specs('n_B1', ['n_A1'], ['out_spikes']),
		'n_C1': adex_specs('n_C1', ['n_A1'], ['out_spikes']),
		'n_D1': adex_specs('n_D1', ['n_A1'], ['out_spikes']),
		'n_E1': adex_specs('n_E1', ['n_A1'], ['out_spikes']),
		'n_F1': adex_specs('n_F1', ['n_A1'], ['out_spikes']),
		'n_G1': adex_specs(
			'n_G1', 
			['n_B1', 'n_C1', 'n_D1', 'n_E1', 'n_F1'], 
			['out_spikes', 'out_spikes', 'out_spikes', 'out_spikes', 'out_spikes']),
	}

	return spark.nn.BrainConfig(input_map=input_map, output_map=output_map, modules_map=modules_map)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def spark_adex_performance(
	build_func,
	sim_repetitions,
	t_steps,
	k_steps,
	units,
	ker_density,
	dt,
	synapse_strength,
	potential_rest,
	potential_reset,
	potential_tau,
	resistance,
	threshold,
	rheobase_threshold,
	spike_slope,
	adaptation_tau,
	adaptation_delta,
	adaptation_subthreshold,
):
	# Build brain
	brain_config = build_func(
		units = units,
		ker_density = ker_density,
		dt = dt,
		synapse_strength = synapse_strength,
		potential_rest = potential_rest,
		potential_reset = potential_reset,
		potential_tau = potential_tau,
		resistance = resistance,
		threshold = threshold,
		rheobase_threshold = rheobase_threshold,
		spike_slope = spike_slope,
		adaptation_tau = adaptation_tau,
		adaptation_delta = adaptation_delta,
		adaptation_subthreshold = adaptation_subthreshold,
	)
	brain = spark.nn.Brain(config=brain_config)
	brain(spikes=spark.SpikeArray( jnp.zeros((units,), dtype=jnp.float16)) )
	graph, state = spark.split((brain))
	# Compile iteration function
	run_model_k_steps(graph, state, k_steps, spikes=spark.SpikeArray( 
		jnp.zeros((units,), dtype=jnp.float16)
	))
	# Benchmark
	times = []
	iters = (t_steps // k_steps)
	for _ in range(sim_repetitions):
		# This effectively resets the brain
		_, state = spark.split((brain))
		start = time.time()
		# Simulate
		for i in range(iters):
			# Note: this is equivalent to passing a constant current to the input neurons for k steps.
			outputs, state = run_model_k_steps(graph, state, k_steps, spikes=spark.SpikeArray( 
				(jax.random.uniform(jax.random.key(i), (units,)) < 0.05).astype(jnp.float16) 
			))
		end = time.time()
		times.append(end-start)

	return times

#-----------------------------------------------------------------------------------------------------------------------------------------------#

spark_adex_performance_model_1 = partial(spark_adex_performance, build_func=adex_brain_model_1_config)

spark_adex_performance_model_2 = partial(spark_adex_performance, build_func=adex_brain_model_2_config)

spark_adex_performance_model_3 = partial(spark_adex_performance, build_func=adex_brain_model_3_config)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################