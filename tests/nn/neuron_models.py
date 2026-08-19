
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import jax 
import jax.numpy as jnp
import numpy as np
import typing as tp
import flax.nnx as nnx
import spark
np.random.seed(42)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

data_test = [
    # Neurons
    (
        spark.nn.neurons.LIFNeuron, 
        {'in_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),}, 
        {'_s_units':(2,3),}
    ),
    (
        spark.nn.neurons.LIFNeuron, 
        {'in_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),}, 
        {'_s_units':(2,3),}
    ),
    (
        spark.nn.neurons.ALIFNeuron, 
        {'in_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),}, 
        {'_s_units':(2,3),}
    ),
    (
        spark.nn.neurons.ALIFNeuron, 
        {'in_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),}, 
        {'_s_units':(2,3),}
    ),
]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@spark.jit
def run_module_simplified(
        module: spark.nn.Module, 
        module_inputs: dict
    ) -> tuple[dict[str, spark.SparkPayload], spark.nn.Module]:
    s = module(**module_inputs)
    return s, module

@pytest.mark.parametrize('module_cls, module_inputs, module_config_kwargs', data_test)
def run_jax_jit_simplified(
        module_cls: type[spark.nn.Module], 
        module_inputs: dict[str, spark.SparkPayload], 
        module_config_kwargs: dict[str, tp.Any]
    ) -> None:
    """
        Validate that the module can run in simplified mode.
        Important for ease of execution.
    """
    module = module_cls(**module_config_kwargs)
    module(**module_inputs)
    output, new_module = run_module_simplified(module, module_inputs)
    for payloads in output.values():
        assert jnp.sum(jnp.isnan(payloads.value)) == 0
        assert jnp.sum(jnp.isinf(payloads.value)) == 0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.jit
def run_module_split(
        graph: nnx.GraphDef, 
        state: nnx.State, 
        module_inputs: dict
    ) -> tuple[dict[str, spark.SparkPayload], nnx.State]:
    module = spark.merge(graph, state)
    s = module(**module_inputs)
    _, state = spark.split((module))
    return s, state

@pytest.mark.parametrize('module_cls, module_inputs, module_config_kwargs', data_test)
def test_jax_jit_split(
        module_cls: type[spark.nn.Module], 
        module_inputs: dict[str, spark.SparkPayload], 
        module_config_kwargs: dict[str, tp.Any]
    ) -> None:
    """
        Validate that the module can run in graph/state split. 
        Helps to detect other potential problems since some times the module can be run in simplified form but 
        still fails to split properly (may lead to some undesire bugs?).
    """
    module = module_cls(**module_config_kwargs)
    module(**module_inputs)
    graph, state = spark.split((module))
    output, new_state = run_module_split(graph, state, module_inputs)
    for payloads in output.values():
        assert jnp.sum(jnp.isnan(payloads.value)) == 0
        assert jnp.sum(jnp.isinf(payloads.value)) == 0

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################