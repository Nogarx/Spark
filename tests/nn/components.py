
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
        {f'input_{idx}': spark.FloatArray(jnp.array(np.random.rand(*s), dtype=jnp.float16)) for idx, s in enumerate([(5,5,5),(50,),(10,10)])}, 
        {},
    ),
    (
        spark.nn.interfaces.ConcatReshape, 
        {f'input_{idx}': spark.FloatArray(jnp.array(np.random.rand(*s), dtype=jnp.float16)) for idx, s in enumerate([(5,5,4),(10,10),(100,)])}, 
        {'reshape':(30,10)}
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
        spark.nn.plasticity.HebbianRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
        spark.nn.plasticity.HebbianRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.plasticity.OjaRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.plasticity.OjaRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.plasticity.ZenkeRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.plasticity.ZenkeRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
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
def test_jax_jit_simplified(
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

traced_synapses_test = [
    (spark.nn.synapses.TracedSynapses, {'tau': 5.0}, True),
    (spark.nn.synapses.TracedSynapses, {'tau': jnp.full((8, 1), 5.0, dtype=jnp.float16)}, True),
    (spark.nn.synapses.TracedSynapses, {'tau': jnp.linspace(3, 20, 8 * 12, dtype=jnp.float16).reshape(8, 12)}, False),
    (spark.nn.synapses.RDTracedSynapses, {'tau_rise': 1.0, 'tau_decay': 5.0}, True),
    (spark.nn.synapses.RDTracedSynapses,
        {'tau_rise': jnp.linspace(1, 3, 8 * 12, dtype=jnp.float16).reshape(8, 12), 'tau_decay': 5.0}, False),
    (spark.nn.synapses.RFSTracedSynapses, {'tau_rise': 1.0, 'tau_fast_decay': 5.0, 'tau_slow_decay': 50.0}, True),
]

def run_traced_synapses(module_cls, config_kwargs, force_full: bool) -> tuple[spark.nn.Module, np.ndarray]:
    """
        Runs a traced synapse over a fixed spike train, optionally forcing the untouched tracer.
    """
    import spark.core.utils as utils
    original = utils.contract_axes
    if force_full:
        utils.contract_axes = lambda array, axes, shape: (array, False)
    try:
        module = module_cls(units=(8,), seed=7, dtype=jnp.float16, dt=1.0, **config_kwargs)
        spikes = np.random.RandomState(0).rand(16, 12) < 0.2
        outputs = [list(module(spikes=spark.SpikeArray(jnp.array(s))).values())[0].value for s in spikes]
    finally:
        utils.contract_axes = original
    return module, np.stack([np.asarray(o, dtype=np.float32) for o in outputs])

@pytest.mark.parametrize('module_cls, config_kwargs, contractible', traced_synapses_test)
def test_traced_synapses_contraction(
        module_cls: type[spark.nn.Module],
        config_kwargs: dict[str, tp.Any],
        contractible: bool
    ) -> None:
    """
        Validate that a traced synapse holds one trace per postsynaptic neuron exactly when nothing its
        tracer is made of varies per synapse, and that it answers the same either way.
    """
    module, contracted = run_traced_synapses(module_cls, config_kwargs, force_full=False)
    _, full = run_traced_synapses(module_cls, config_kwargs, force_full=True)
    assert module._contracted_tracer is contractible
    assert np.max(np.abs(contracted - full)) <= 1e-2 * max(np.max(np.abs(full)), 1e-6)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
