
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
            glue = 0,#jnp.array(0), 
            mins = -1,#jnp.array(-1),  
            maxs = 1,#jnp.array(1), 
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
            synapses_params__kernel__scale = 3.0,
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
        outputs={
            'action': 'signal'
        },
        config = spark.nn.interfaces.ExponentialIntegratorConfig(
            num_outputs = 2,
        )
    )

    modules_specs = [
        spiker_specs, neurons_specs, integrator
    ]
    return spark.nn.BrainConfig(modules_specs=modules_specs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.jit
def run_module_simplified(
        module: spark.nn.Brain, 
        module_inputs: dict
    ) -> tuple[dict[str, spark.SparkPayload], spark.nn.Module]:
    out = module(**module_inputs)
    return out, module

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
    out, brain_model = run_module_simplified(brain_model, brain_inputs)
    assert isinstance(out['action'], spark.FloatArray)
    assert out['action'].value.shape == (2,)
    assert jnp.sum(jnp.isnan(out['action'].value)) == 0
    assert jnp.sum(jnp.isinf(out['action'].value)) == 0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.jit
def run_module_split(
        graph: nnx.GraphDef, 
        state: nnx.GraphState | nnx.VariableState, 
        module_inputs: dict
    ) -> tuple[dict[str, spark.SparkPayload], nnx.GraphState | nnx.VariableState]:
    module = spark.merge(graph, state)
    out = module(**module_inputs)
    _, state = spark.split((module))
    return out, state

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
    out, state = run_module_split(graph, state, brain_inputs)
    assert isinstance(out['action'], spark.FloatArray)
    assert out['action'].value.shape == (2,)
    assert jnp.sum(jnp.isnan(out['action'].value)) == 0
    assert jnp.sum(jnp.isinf(out['action'].value)) == 0

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################