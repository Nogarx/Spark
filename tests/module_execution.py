
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
    # Input interfaces
    (
        spark.nn.interfaces.PoissonSpiker, 
        {'signal': spark.FloatArray(jnp.array(np.random.rand(10,), dtype=jnp.float16)),}, 
        {}
    ),
    (
        spark.nn.interfaces.LinearSpiker, 
        {'signal': spark.FloatArray(jnp.array(np.random.rand(10,), dtype=jnp.float16)),}, 
        {}
    ),
    (
        spark.nn.interfaces.TopologicalPoissonSpiker, 
        {'signal': spark.FloatArray(jnp.array(np.random.rand(10,), dtype=jnp.float16)),}, 
        {}
    ),
    (
        spark.nn.interfaces.TopologicalLinearSpiker, 
        {'signal': spark.FloatArray(jnp.array(np.random.rand(10,), dtype=jnp.float16)),}, 
        {}
    ),
    # Output interfaces
    (
        spark.nn.interfaces.ExponentialIntegrator, 
        {'spikes': spark.SpikeArray(jnp.array(np.random.rand(10,) < 0.5)),}, 
        {'num_outputs':2,}
    ),
    # Control interfaces
    (
        spark.nn.interfaces.Concat, 
        {'inputs': [spark.FloatArray(jnp.array(np.random.rand(*s), dtype=jnp.float16)) for s in [(5,5,5),(50,),(10,10)]],}, 
        {'num_inputs':3, 'payload_type':spark.SpikeArray,}
    ),
    (
        spark.nn.interfaces.ConcatReshape, 
        {'inputs': [spark.FloatArray(jnp.array(np.random.rand(*s), dtype=jnp.float16)) for s in [(5,5,4),(10,10),(100,)]],}, 
        {'num_inputs':3, 'reshape':(30,10), 'payload_type':spark.SpikeArray,}
    ),
    (
        spark.nn.interfaces.Sampler, 
        {'inputs': spark.FloatArray(jnp.array(np.random.rand(10,10,10), dtype=jnp.float16)),}, 
        {'sample_size':10,}
    ),
    # Delays
    (
        spark.nn.delays.NDelays, 
        {'in_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,4) < 0.5)),}, 
        {'max_delay':5,}
    ),
    (
        spark.nn.delays.N2NDelays, 
        {'in_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,4) < 0.5)),}, 
        {'units':(2,2), 'max_delay':5,}
    ),
    # Synapses
    (
        spark.nn.synapses.LinearSynapses, 
        {'spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.synapses.LinearSynapses, 
        {'spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.synapses.TracedSynapses, 
        {'spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.synapses.TracedSynapses, 
        {'spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.synapses.RDTracedSynapses, 
        {'spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.synapses.RDTracedSynapses, 
        {'spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),}, 
        {'units':(2,3),}
    ),
    # Somas
    (
        spark.nn.somas.LeakySoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.somas.RefractoryLeakySoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.somas.AdaptiveLeakySoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.somas.ExponentialSoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.somas.RefractoryExponentialSoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.somas.AdaptiveExponentialSoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.somas.IzhikevichSoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3),}
    ),
    # Learning rules
    (
        spark.nn.learning_rules.HebbianRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
        spark.nn.learning_rules.HebbianRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.learning_rules.OjaRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.learning_rules.OjaRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.learning_rules.ZenkeRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.learning_rules.ZenkeRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
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

@jax.jit
def run_module_simplified(
        module: spark.nn.Brain, 
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
        state: nnx.GraphState | nnx.VariableState, 
        module_inputs: dict
    ) -> tuple[dict[str, spark.SparkPayload], nnx.GraphState | nnx.VariableState]:
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