
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import jax 
import jax.numpy as jnp
import typing as tp
import flax.nnx as nnx
import spark

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@pytest.fixture
def test_brain_config() -> spark.nn.BrainConfig:
    spiker_specs = spark.ModuleSpecs(
        name ='spiker', 
        module_cls = spark.nn.interfaces.TopologicalLinearSpiker, 
        inputs = {
            'signal': [
                spark.PortMap(origin='__call__', port='drive'),
            ]
        },
        config = spark.nn.interfaces.TopologicalLinearSpikerConfig(
            glue = jnp.array(0), 
            mins = jnp.array(-1),  
            maxs = jnp.array(1), 
            resolution = 128, 
            max_freq = 200.0, 
            tau = 30.0
        )
    )
    neurons_specs = spark.ModuleSpecs(
        name ='neurons', 
        module_cls = spark.nn.neurons.ALIFNeuron, 
        inputs = {
            'in_spikes': [
                spark.PortMap(origin='spiker', port='spikes'),
                spark.PortMap(origin='neurons', port='out_spikes'),
            ]
        },
        config = spark.nn.neurons.ALIFNeuronConfig(
            _s_units = (16,),
            _s_async_spikes = True,
            synapses_params__kernel_initializer__scale = 3.0,
            soma_params__threshold_tau = 25.0 * jax.random.uniform(jax.random.key(43), shape=(16,), dtype=jnp.float16)**2,
            soma_params__threshold_delta = 250.0 * jax.random.uniform(jax.random.key(43), shape=(16,), dtype=jnp.float16)**2,
            soma_params__cooldown = 2.0,  
        )
    )
    integrator = spark.ModuleSpecs(
        name ='integrator', 
        module_cls = spark.nn.interfaces.ExponentialIntegrator, 
        inputs = {
            'spikes': [
                spark.PortMap(origin='neurons', port='out_spikes'),
            ]
        },
        config = spark.nn.interfaces.ExponentialIntegratorConfig(
            num_outputs = 2,
        )
    )
    input_map = {
        'drive': spark.InputSpec(
            payload_type=spark.FloatArray, 
            shape=(4,), 
            dtype=jnp.float16,
            is_optional=False,
        )
    }
    output_map = {
        'integrator': {
            'signal': spark.OutputSpec(
                payload_type=spark.FloatArray, 
                shape=(2,), 
                dtype=jnp.float16
            )
        }
    }
    modules_map = {
        'spiker': spiker_specs,
        'neurons': neurons_specs,
        'integrator': integrator,
    }
    return spark.nn.BrainConfig(input_map=input_map, output_map=output_map, modules_map=modules_map)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.jit
def run_module_simplified(
        module: spark.nn.Brain, 
        module_inputs: dict
    ) -> tuple[tp.Any, spark.nn.Module]:
    s = module(**module_inputs)
    return s, module

def test_jax_jit_split(
        test_brain_config: spark.nn.BrainConfig
    ) -> None:
    """
        Validate that the module can run in simplified mode.
        Important for ease of execution.
    """
    brain_model = spark.nn.Brain(config=test_brain_config)
    brain_inputs = {
        'drive': spark.FloatArray(jnp.zeros((4,), dtype=jnp.float16))
    }
    brain_model(**brain_inputs)
    spikes, brain_model = run_module_simplified(brain_model, brain_inputs)
    assert isinstance(spikes['integrator.signal'], spark.FloatArray)
    assert spikes['integrator.signal'].value.shape == (2,)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.jit
def run_module_split(
        graph: nnx.GraphDef, 
        state: nnx.GraphState | nnx.VariableState, 
        module_inputs: dict
    ) -> tuple[tp.Any, nnx.GraphState | nnx.VariableState]:
    module = spark.merge(graph, state)
    s = module(**module_inputs)
    _, state = spark.split((module))
    return s, state

def test_jax_jit_split(
        test_brain_config: spark.nn.BrainConfig
    ) -> None:
    """
        Validate that the module can run in graph/state split. 
        Helps to detect other potential problems since some times the module can be run in simplified form but 
        still fails to split properly (may lead to some undesire bugs?).
    """
    brain_model = spark.nn.Brain(config=test_brain_config)
    brain_inputs = {
        'drive': spark.FloatArray(jnp.zeros((4,), dtype=jnp.float16))
    }
    brain_model(**brain_inputs)
    graph, state = spark.split((brain_model))
    spikes, state = run_module_split(graph, state, brain_inputs)
    assert isinstance(spikes['integrator.signal'], spark.FloatArray)
    assert spikes['integrator.signal'].value.shape == (2,)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################