
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
        spark.nn.interfaces.SignalAccumulator, 
        {'signal': spark.FloatArray(jnp.array(np.random.rand(4,), dtype=jnp.float16)),}, 
        {'tau': 5.0,}
    ),
    (
        spark.nn.interfaces.SignalAccumulator, 
        {
            'signal': spark.FloatArray(jnp.array(np.random.rand(4,), dtype=jnp.float16)),
            'trace': spark.FloatArray(jnp.array(np.random.rand(4,), dtype=jnp.float16)),
        }, 
        {'tau': 5.0,}
    ),
    (
        spark.nn.interfaces.SignalAverage, 
        {'signal': spark.FloatArray(jnp.array(np.random.rand(4,), dtype=jnp.float16)),}, 
        {'tau': 5.0,}
    ),
    (
        spark.nn.interfaces.SignalAverage, 
        {
            'signal': spark.FloatArray(jnp.array(np.random.rand(4,), dtype=jnp.float16)),
            'trace': spark.FloatArray(jnp.array(np.random.rand(4,), dtype=jnp.float16)),
        }, 
        {'tau': 5.0,}
    ),
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
        spark.nn.somas.ExponentialSoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.somas.IzhikevichSoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3),}
    ),
    # Adaptive somas
    (
        spark.nn.somas.AdaptiveLeakySoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3),}
    ),
    (
        spark.nn.somas.AdaptiveLeakySoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3), 'cooldown': 2.0}
    ),
    (
        spark.nn.somas.AdaptiveLeakySoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3), 'cooldown': 2.0, 'clamp_duration': 1.0, 'threshold_delta': 100.0, 'adaptation_delta': 7.0}
    ),
    (
        spark.nn.somas.AdaptiveExponentialSoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3), 'adaptation_delta': 7.0}
    ),
    (
        spark.nn.somas.AdaptiveExponentialSoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3), 'cooldown': 2.0, 'clamp_duration': 1.0, 'threshold_delta': 100.0, 'adaptation_delta': 7.0}
    ),
    (
        spark.nn.somas.AdaptiveIzhikevichSoma, 
        {'current': spark.CurrentArray(jnp.array(np.random.rand(2,3), dtype=jnp.float16)),}, 
        {'units':(2,3), 'cooldown': 2.0, 'clamp_duration': 1.0, 'threshold_delta': 100.0, 'adaptation_delta': 7.0}
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
    (
    spark.nn.plasticity.QuadrupletRule, 
        {
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    # Modulated learning rules
    (
    spark.nn.plasticity.ModulatedHebbianRule, 
        {
            'modulation': spark.FloatArray(jnp.array(np.random.rand(), dtype=jnp.float16)),
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.plasticity.ModulatedHebbianRule, 
        {
            'modulation': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.plasticity.ModulatedQuadrupletRule, 
        {
            'modulation': spark.FloatArray(jnp.array(np.random.rand(), dtype=jnp.float16)),
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.plasticity.ModulatedOjaRule, 
        {
            'modulation': spark.FloatArray(jnp.array(np.random.rand(), dtype=jnp.float16)),
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.plasticity.ModulatedOjaRule, 
        {
            'modulation': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.plasticity.ModulatedZenkeRule, 
        {
            'modulation': spark.FloatArray(jnp.array(np.random.rand(), dtype=jnp.float16)),
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(4,5) < 0.5), async_spikes=False),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
    (
    spark.nn.plasticity.ModulatedZenkeRule, 
        {
            'modulation': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
            'pre_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3,4,5) < 0.5), async_spikes=True),
            'post_spikes': spark.SpikeArray(jnp.array(np.random.rand(2,3) < 0.5)),
            'kernel': spark.FloatArray(jnp.array(np.random.rand(2,3,4,5), dtype=jnp.float16)),
        }, 
        {'units':(2,3),}
    ),
]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def assert_finite(payload: spark.SparkPayload) -> None:
    """
        Validate that every floating point leaf of a payload is free of nan/inf.
        Payloads may carry several channels, so the check walks the pytree instead of
        assuming a single "value" array.
    """
    for leaf in jax.tree.leaves(payload):
        if not jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact):
            continue
        assert jnp.sum(jnp.isnan(leaf)) == 0
        assert jnp.sum(jnp.isinf(leaf)) == 0

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
        assert_finite(payloads)

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
        assert_finite(payloads)

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

#-----------------------------------------------------------------------------------------------------------------------------------------------#

delays_test = [
    (spark.nn.delays.NDelays, {}),
    (spark.nn.delays.N2NDelays, {'units': (2,)}),
]

@pytest.mark.parametrize('module_cls, config_kwargs', delays_test)
def test_every_delay_releases_its_spike_that_many_steps_later(module_cls, config_kwargs) -> None:
    """
        A spike reaches the output after its delay, up to the longest one, ``ceil(max_delay / dt)`` steps.
    """
    module = module_cls(max_delay=4.0, **config_kwargs)
    silent = spark.SpikeArray(jnp.zeros((4,), dtype=jnp.float16))
    module(in_spikes=silent)
    module.reset()
    delays = [1, 2, 3, 4]
    module._kernel = spark.Constant(jnp.broadcast_to(jnp.array(delays, dtype=jnp.uint8), module.kernel.value.shape), dtype=jnp.uint8)
    spike = spark.SpikeArray(jnp.ones((4,), dtype=jnp.float16))
    released = [
        np.asarray(module(in_spikes=spike if step == 0 else silent)['out_spikes'].spikes).reshape(-1, 4) for step in range(8)
    ]
    for unit, delay in enumerate(delays):
        assert [step for step in range(8) if released[step][:, unit].any()] == [delay]

@pytest.mark.parametrize('module_cls, config_kwargs', delays_test)
def test_the_drawn_delays_reach_the_longest_and_no_further(module_cls, config_kwargs) -> None:
    module = module_cls(max_delay=4.0, **config_kwargs)
    module(in_spikes=spark.SpikeArray(jnp.zeros((256,), dtype=jnp.float16)))
    assert set(np.unique(np.asarray(module.kernel.value)).tolist()) == {1, 2, 3, 4}

@pytest.mark.parametrize('module_cls, config_kwargs', delays_test)
def test_a_given_array_of_delays_is_taken_as_it_is(module_cls, config_kwargs) -> None:
    module = module_cls(max_delay=4.0, delays=jnp.array([1, 2, 3, 4]), **config_kwargs)
    module(in_spikes=spark.SpikeArray(jnp.zeros((4,), dtype=jnp.float16)))
    assert (np.asarray(module.kernel.value).reshape(-1, 4) == [1, 2, 3, 4]).all()

@pytest.mark.parametrize('module_cls, config_kwargs', delays_test)
@pytest.mark.parametrize('delays', [[0, 1, 2, 3], [1, 2, 3, 5]])
def test_a_given_delay_of_no_step_or_past_the_longest_is_refused(module_cls, config_kwargs, delays) -> None:
    module = module_cls(max_delay=4.0, delays=jnp.array(delays), **config_kwargs)
    with pytest.raises(ValueError, match='Delays are of 1 to 4 steps'):
        module(in_spikes=spark.SpikeArray(jnp.zeros((4,), dtype=jnp.float16)))

@pytest.mark.parametrize('module_cls, config_kwargs', delays_test)
@pytest.mark.parametrize('initializer', [
    spark.nn.initializers.ConstantInitializerConfig(scale=3),
    spark.nn.initializers.SparseUniformInitializerConfig(),
])
def test_every_initializer_draws_delays_within_the_range(module_cls, config_kwargs, initializer) -> None:
    module = module_cls(max_delay=4.0, delays=initializer, **config_kwargs)
    module(in_spikes=spark.SpikeArray(jnp.zeros((256,), dtype=jnp.float16)))
    kernel = np.asarray(module.kernel.value)
    assert kernel.min() >= 1 and kernel.max() <= 4

@pytest.mark.parametrize('module_cls, config_kwargs', delays_test)
def test_delays_past_255_steps_are_held_and_released_on_time(module_cls, config_kwargs) -> None:
    """
        The kernel takes a dtype holding the longest delay, so that no delay wraps around to zero steps.
    """
    module = module_cls(max_delay=300.0, **config_kwargs)
    silent = spark.SpikeArray(jnp.zeros((256,), dtype=jnp.float16))
    module(in_spikes=silent)
    kernel = np.asarray(module.kernel.value)
    assert kernel.min() >= 1 and 255 < kernel.max() <= 300
    module.reset()
    module._kernel = spark.Constant(jnp.full(kernel.shape, 300), dtype=kernel.dtype)
    spike = spark.SpikeArray(jnp.ones((256,), dtype=jnp.float16))
    released = [bool(np.asarray(module(in_spikes=spike if step == 0 else silent)['out_spikes'].spikes).any()) for step in range(302)]
    assert [step for step, any_spike in enumerate(released) if any_spike] == [300]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _entries(size: int) -> spark.FloatArray:
    return spark.FloatArray(jnp.arange(size, dtype=jnp.float16))

def test_a_sampler_gives_one_port_per_output() -> None:
    module = spark.nn.interfaces.Sampler(sample_size=3, num_outputs=4)
    outputs = module(a=_entries(10))
    assert list(outputs) == ['output_0', 'output_1', 'output_2', 'output_3']
    assert list(module.get_output_specs()) == list(outputs)
    # Each output reads the entries of its row of indices.
    for k, output in enumerate(outputs.values()):
        np.testing.assert_array_equal(np.asarray(output.value), np.asarray(module.indices[k]).astype(np.float16))

def test_disjoint_outputs_share_no_entry_while_the_input_holds_enough() -> None:
    module = spark.nn.interfaces.Sampler(sample_size=5, num_outputs=4, disjoint=True)
    module(a=_entries(12), b=_entries(8))
    assert len(set(np.asarray(module.indices).ravel().tolist())) == 20

def test_disjoint_outputs_draw_every_entry_evenly_past_that() -> None:
    module = spark.nn.interfaces.Sampler(sample_size=4, num_outputs=5, disjoint=True)
    module(a=_entries(7))
    indices = np.asarray(module.indices)
    counts = np.bincount(indices.ravel(), minlength=7)
    assert counts.max() - counts.min() <= 1
    # Each output takes fewer entries than the input holds, and holds none twice.
    assert all(len(set(row.tolist())) == 4 for row in indices)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
