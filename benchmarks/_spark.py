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

from functools import partial

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
        potentials.append(np.array(model.soma.potential.value) + kwargs['potential_rest'])
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
            resistance = resistance / 1000, # MΩ -> GΩ
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
			resistance = resistance / 1000, # MΩ -> GΩ
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

simulate_LIF_model_spark = partial(simulate_model_spark, build_func=build_LIF_model)

simulate_AdEx_model_spark = partial(simulate_model_spark, build_func=build_AdEx_model)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################