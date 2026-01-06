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
from _hh import HodgkinHuxleySoma, HodgkinHuxleySomaConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# TODO: Find a workaround to make scan iterate over dictionaries.

@partial(jax.jit, static_argnames=['steps', 'unroll'])
def run_model_fidelity(graph, state, currents, steps, unroll=10):
	def step_fn(carry_state, c):# -> tuple[GraphState | VariableState, Any]:
		# Unpack carry_state
		model_state = carry_state
		# Merge model
		model = spark.merge(graph, model_state)
		# Run model
		outputs = model(current=spark.CurrentArray(jnp.array([c])))
		# Get new state
		_, new_state = spark.split((model))
		return new_state, outputs
	# Run the scan loop
	final_state, final_outputs = jax.lax.scan(
		step_fn, 
		state, 
		xs=currents, 
		length=steps,
		unroll=unroll,
	)
	return final_outputs, final_state

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@partial(jax.jit, static_argnames=['steps', 'unroll'])
def run_model_performance(graph, state, spikes, steps, unroll=20):
	def step_fn(carry_state, s):# -> tuple[GraphState | VariableState, Any]:
		# Unpack carry_state
		model_state = carry_state
		# Merge model
		model = spark.merge(graph, model_state)
		# Run model
		outputs = model(spikes=spark.SpikeArray(jnp.array(s)))
		# Get new state
		_, new_state = spark.split((model))
		return new_state, outputs
	# Run the scan loop
	final_state, final_outputs = jax.lax.scan(
		step_fn,
		state,
		xs=spikes,
		length=steps,
		unroll=unroll,
	)
	return final_outputs, final_state

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def simulate_model_spark(
	currents,
	build_func,
	offset,
	**kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	# Initialize model
	model = build_func(**kwargs)
	# Run simulation
	graph, state = spark.split((model))
	spikes, potentials, times = [], [], []
	t, dt = 0, kwargs['dt']
	outputs, _ = run_model_fidelity(graph, state, currents, steps=len(currents))
	times = [t*dt for t in range(len(currents))]
	potentials = np.array(outputs['potential'].value).reshape(-1) + offset
	spikes = np.array([t*dt for t, s in enumerate(np.array(outputs['spikes'].value).reshape(-1)) if s == 1])
	times = np.array(times).reshape(-1)
	return times, potentials, spikes


#-----------------------------------------------------------------------------------------------------------------------------------------------#

def build_LIF_model(
	dt,
	potential_rest,
	potential_reset,
	potential_tau,
	resistance,
	threshold,
	cooldown,
) -> spark.nn.somas.StrictRefractoryLeakySoma:
	# NOTE: We need to substract 1 timestep from the cooldown to match the Brian2 dynamic 
	# Since we update the refractory the period at the end of the cycle, Brian2 does it at the begining. 
	soma_config = spark.nn.somas.StrictRefractoryLeakySomaConfig(
		_s_units = (1,),
		_s_dt = dt,
		_s_dtype = jnp.float16,
		potential_rest = potential_rest,
		potential_reset = potential_reset,
		potential_tau = potential_tau,
		resistance = resistance,
		threshold = threshold,
		cooldown = cooldown - dt, 
	)
	soma = spark.nn.somas.StrictRefractoryLeakySoma(config=soma_config)
	soma(current=spark.CurrentArray( jnp.zeros((1,)) ))
	return soma

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def build_AdEx_model(
	dt,
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
) -> spark.nn.somas.AdaptiveExponentialSoma:
	# NOTE: We need to substract 1 timestep from the cooldown to match the Brian2 dynamic 
	# Since we update the refractory the period at the end of the cycle, Brian2 does it at the begining. 
	soma_config = spark.nn.somas.AdaptiveExponentialSomaConfig(
		_s_units = (1,),
		_s_dt = dt,
		_s_dtype = jnp.float16,
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
	soma = spark.nn.somas.AdaptiveExponentialSoma(config=soma_config)
	soma(current=spark.CurrentArray( jnp.zeros((1,)) ))
	return soma

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def build_HH_model(
	dt,
	c_m,
	e_leak,
	e_na,
	e_k,
	g_leak,
	g_na,
	g_k,
	threshold,
) -> HodgkinHuxleySoma:
	soma_config = HodgkinHuxleySomaConfig(
		_s_units = (1,),
		_s_dt = dt,
		_s_dtype = jnp.float16,
		c_m = c_m,
		e_leak = e_leak,
		e_na = e_na,
		e_k = e_k,
		g_leak = g_leak,
		g_na = g_na,
		g_k = g_k,
		threshold = threshold, 
	)
	soma = HodgkinHuxleySoma(config=soma_config)
	soma(current=spark.CurrentArray( jnp.zeros((1,)) ))
	return soma

#-----------------------------------------------------------------------------------------------------------------------------------------------#

simulate_LIF_model_spark = partial(simulate_model_spark, build_func=build_LIF_model)

simulate_AdEx_model_spark = partial(simulate_model_spark, build_func=build_AdEx_model)

simulate_HH_model_spark = partial(simulate_model_spark, build_func=build_HH_model)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def adex_specs(
		name, 
		origin, 
		port,
		units,
		**model_params,
		#ker_density,
		#dt,
		#synapse_strength,
		#potential_rest,
		#potential_reset,
		#rheobase_threshold,
		#adaptation_subthreshold,
		#adaptation_delta,
		#threshold,
		#spike_slope,
		#adaptation_tau,
		#potential_tau,
		#resistance, # MΩ -> GΩ
	) -> spark.ModuleSpecs:
	return spark.ModuleSpecs(
		name = name, 
		module_cls = spark.nn.neurons.AdExNeuron, 
		inputs = {
			'in_spikes': [
				spark.PortMap(origin=o, port=p) for o, p in zip(origin, port)
			]
		},
		config = spark.nn.neurons.AdExNeuronConfig(
			_s_units = (units,),
			_s_dt = model_params['dt'],
			_s_dtype = jnp.float16,
			inhibitory_rate = 0.0,
			soma = spark.nn.somas.AdaptiveExponentialSomaConfig(
				potential_rest = model_params['potential_rest'],
				potential_reset = model_params['potential_reset'],
				potential_tau = model_params['potential_tau'],
				resistance = model_params['resistance'],
				threshold = model_params['threshold'],
				rheobase_threshold = model_params['rheobase_threshold'],
				spike_slope = model_params['spike_slope'],
				adaptation_tau = model_params['adaptation_tau'],
				adaptation_delta = model_params['adaptation_delta'],
				adaptation_subthreshold = model_params['adaptation_subthreshold'],
			),
			synapses = spark.nn.synapses.LinearSynapsesConfig(
				units = (units,),
				kernel = spark.nn.initializers.SparseUniformInitializerConfig(
					density = model_params['ker_density'],
					scale = model_params['synapse_strength']
				),
			),
			delays = None,
			learning_rule = None,
		)
	)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def adex_brain_model_1_config(units, **model_params) -> spark.nn.BrainConfig:
	input_map = {
		'spikes': spark.PortSpecs(
			payload_type=spark.FloatArray, 
			shape=(units,), 
			dtype=jnp.float16,
			async_spikes=False,
			inhibition_mask=False,
		)
	}
	output_map = {
		'spikes': {
			'input': spark.PortMap(
				origin='n_D1',
				port='out_spikes'
			),
			'spec': spark.PortSpecs(
				payload_type=spark.SpikeArray,
				shape=(units,),
				dtype=jnp.float16,
			)
		}
	}
	modules_map = {
		'n_A1': adex_specs('n_A1', ['__call__'], ['spikes'], units, **model_params),
		'n_B1': adex_specs('n_B1', ['n_A1'], ['out_spikes'], units, **model_params),
		'n_B2': adex_specs('n_B2', ['n_B1'], ['out_spikes'], units, **model_params),
		'n_B3': adex_specs('n_B3', ['n_B2'], ['out_spikes'], units, **model_params),
		'n_C1': adex_specs('n_C1', ['n_A1'], ['out_spikes'], units, **model_params),
		'n_C2': adex_specs('n_C2', ['n_C1'], ['out_spikes'], units, **model_params),
		'n_C3': adex_specs('n_C3', ['n_C2'], ['out_spikes'], units, **model_params),
		'n_D1': adex_specs('n_D1', ['n_B3', 'n_C3'], ['out_spikes', 'out_spikes'], units, **model_params),
	}
	return spark.nn.BrainConfig(input_map=input_map, output_map=output_map, modules_map=modules_map)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def adex_brain_model_2_config(units, **model_params) -> spark.nn.BrainConfig:

	input_map = {
		'spikes': spark.PortSpecs(
			payload_type=spark.FloatArray, 
			shape=(units,), 
			dtype=jnp.float16,
			async_spikes=False,
			inhibition_mask=False,
		)
	}
	output_map = {
		'spikes': {
			'input': spark.PortMap(
				origin='n_D1',
				port='out_spikes'
			),
			'spec': spark.PortSpecs(
				payload_type=spark.SpikeArray,
				shape=(units,),
				dtype=jnp.float16,
			)
		}
	}
	modules_map = {
		'n_A1': adex_specs('n_A1', ['__call__'], ['spikes'], units, **model_params),
		'n_A2': adex_specs('n_A2', ['n_A1'], ['out_spikes'], units, **model_params),
		'n_B1': adex_specs('n_B1', ['__call__'], ['spikes'], units, **model_params),
		'n_B2': adex_specs('n_B2', ['n_B1'], ['out_spikes'], units, **model_params),
		'n_C1': adex_specs('n_C1', ['__call__'], ['spikes'], units, **model_params),
		'n_C2': adex_specs('n_C2', ['n_C1'], ['out_spikes'], units, **model_params),
		'n_D1': adex_specs('n_D1', ['n_A2', 'n_B2', 'n_C2'], ['out_spikes', 'out_spikes', 'out_spikes'], units, **model_params),
	}

	return spark.nn.BrainConfig(input_map=input_map, output_map=output_map, modules_map=modules_map)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def adex_brain_model_3_config(units, **model_params) -> spark.nn.BrainConfig:

	input_map = {
		'spikes': spark.PortSpecs(
			payload_type=spark.FloatArray, 
			shape=(units,), 
			dtype=jnp.float16,
			async_spikes=False,
			inhibition_mask=False,
		)
	}
	output_map = {
		'spikes': {
			'input': spark.PortMap(
				origin='n_D1',
				port='out_spikes'
			),
			'spec': spark.PortSpecs(
				payload_type=spark.SpikeArray,
				shape=(units,),
				dtype=jnp.float16,
			)
		}
	}
	modules_map = {
		'n_A1': adex_specs('n_A1', ['__call__'], ['spikes'], units, **model_params),
		'n_B1': adex_specs('n_B1', ['n_A1'], ['out_spikes'], units, **model_params),
		'n_C1': adex_specs('n_C1', ['n_A1'], ['out_spikes'], units, **model_params),
		'n_D1': adex_specs('n_D1', ['n_A1'], ['out_spikes'], units, **model_params),
		'n_E1': adex_specs('n_E1', ['n_A1'], ['out_spikes'], units, **model_params),
		'n_F1': adex_specs('n_F1', ['n_A1'], ['out_spikes'], units, **model_params),
		'n_G1': adex_specs(
			'n_G1', 
			['n_B1', 'n_C1', 'n_D1', 'n_E1', 'n_F1'], 
			['out_spikes', 'out_spikes', 'out_spikes', 'out_spikes', 'out_spikes'], units, **model_params),
	}

	return spark.nn.BrainConfig(input_map=input_map, output_map=output_map, modules_map=modules_map)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def run_adex_spark(
	build_func,
	sim_repetitions,
	t_max,
	sim_steps,
	units,
	dt,
	ker_density,
	synapse_strength,
	freq_spike,
	model_params,
):
	# Build brain
	iterations = int(t_max / sim_steps)
	compile_start = time.time()
	brain_config = build_func(
		units,
		dt = dt,
		ker_density = ker_density,
		synapse_strength = synapse_strength,
		**model_params,
	)
	brain = spark.nn.Brain(config=brain_config)
	brain(spikes=spark.SpikeArray( jnp.zeros((units,), dtype=jnp.float16), async_spikes=False, inhibition_mask=False) )
	graph, state = spark.split((brain))
	# Compile iteration function
	spikes = jnp.zeros((sim_steps, units,), dtype=jnp.float16)
	run_model_performance(graph, state, spikes, sim_steps)
	compile_end = time.time()
	compile_time = compile_end - compile_start
	# Benchmark
	times = []
	freq_spike = freq_spike * dt
	for _ in range(sim_repetitions):
		# This effectively resets the brain
		_, state = spark.split((brain))
		start = time.time()
		# Simulate
		for i in range(iterations):
			spikes = (jax.random.uniform(jax.random.key(i), (sim_steps, units,)) < freq_spike).astype(jnp.float16)
			outputs, state = run_model_performance(graph, state, spikes, sim_steps)
		end = time.time()
		times.append(end-start)

	return compile_time, times

#-----------------------------------------------------------------------------------------------------------------------------------------------#

spark_adex_performance_model_1 = partial(run_adex_spark, build_func=adex_brain_model_1_config)

spark_adex_performance_model_2 = partial(run_adex_spark, build_func=adex_brain_model_2_config)

spark_adex_performance_model_3 = partial(run_adex_spark, build_func=adex_brain_model_3_config)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################