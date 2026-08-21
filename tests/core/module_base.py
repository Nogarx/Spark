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

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class CounterOutput(tp.TypedDict):
    total: spark.FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class CounterConfig(spark.nn.Config):
    start: float = 0.0
    step: float = 1.0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@spark.register_module('probe_counter')
class Counter(spark.nn.Module):
    """
        The smallest module that still has everything a module has: a configuration, a constant, a state
        that moves, a lazy build and a typed call.
    """
    config: CounterConfig

    def __init__(self, config: CounterConfig | None = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.step = spark.Constant(jnp.array(self.config.step, dtype=jnp.float32))

    def build(self, signal: spark.FloatArray):
        self.total = spark.Variable(
            jnp.full(signal.value.shape, self.config.start, dtype=jnp.float32)
        )

    def reset(self) -> None:
        super().reset()
        self.total.value = jnp.full(self.total.value.shape, self.config.start, dtype=jnp.float32)

    def __call__(self, signal: spark.FloatArray) -> CounterOutput:
        self.total.value = self.total.value + self.step.value * signal.value
        return {'total': spark.FloatArray(self.total.value)}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def signal():
    return spark.FloatArray(jnp.ones((4,), dtype=jnp.float32))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestLifecycle:
    """
        What a module is before and after it has been called once.
    """

    def test_the_configuration_is_reachable_after_super_init(self) -> None:
        module = Counter(step=2.0)
        assert module.config.step == 2.0
        assert module.step.value == 2.0

    def test_build_is_deferred_until_the_first_call(self, signal) -> None:
        module = Counter()
        assert not hasattr(module, 'total')
        module(signal=signal)
        assert module.total.value.shape == (4,)

    def test_build_reads_the_shape_of_what_it_was_given(self) -> None:
        module = Counter()
        module(signal=spark.FloatArray(jnp.ones((2, 3), dtype=jnp.float32)))
        assert module.total.value.shape == (2, 3)

    def test_the_state_moves_between_calls(self, signal) -> None:
        module = Counter(step=2.0)
        first = module(signal=signal)['total'].value
        second = module(signal=signal)['total'].value
        assert np.allclose(np.asarray(first), 2.0)
        assert np.allclose(np.asarray(second), 4.0)

    def test_the_building_pass_does_not_leak_into_the_state(self, signal) -> None:
        module = Counter(step=2.0)
        assert np.allclose(np.asarray(module(signal=signal)['total'].value), 2.0)

    def test_a_constant_does_not_move(self, signal) -> None:
        module = Counter(step=2.0)
        module(signal=signal)
        module(signal=signal)
        assert module.step.value == 2.0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestContract:
    """
        What the type hints of a module buy the framework.
    """

    def test_the_call_answers_with_what_it_declared(self, signal) -> None:
        output = Counter()(signal=signal)
        assert set(output) == {'total'}
        assert isinstance(output['total'], spark.FloatArray)

    def test_the_inputs_are_known_before_the_call(self) -> None:
        module = Counter()
        assert 'signal' in module._get_input_specs()

    def test_the_outputs_are_known_before_the_call(self) -> None:
        module = Counter()
        assert 'total' in module._get_output_specs()

    def test_an_unexpected_input_is_refused(self, signal) -> None:
        with pytest.raises(Exception):
            Counter()(not_a_port=signal)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@spark.jit
def _run_simplified(module: spark.nn.Module, module_inputs: dict) -> tuple[dict, spark.nn.Module]:
    outputs = module(**module_inputs)
    return outputs, module

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.jit
def _run_split(graph: nnx.GraphDef, state: nnx.State, module_inputs: dict) -> tuple[dict, nnx.State]:
    module = spark.merge(graph, state)
    outputs = module(**module_inputs)
    _, state = spark.split((module))
    return outputs, state

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestExecution:
    """
        The two ways a model is run (tutorial #2).
    """

    def test_simplified(self, signal) -> None:
        module = Counter(step=2.0)
        module(signal=signal)
        outputs, module = _run_simplified(module, {'signal': signal})
        assert np.allclose(np.asarray(outputs['total'].value), 4.0)

    def test_split_and_merge(self, signal) -> None:
        module = Counter(step=2.0)
        module(signal=signal)
        graph, state = spark.split((module))
        outputs, state = _run_split(graph, state, {'signal': signal})
        assert np.allclose(np.asarray(outputs['total'].value), 4.0)

    def test_the_state_carries_across_jitted_calls(self, signal) -> None:
        module = Counter(step=1.0)
        module(signal=signal)
        graph, state = spark.split((module))
        for _ in range(3):
            outputs, state = _run_split(graph, state, {'signal': signal})
        assert np.allclose(np.asarray(outputs['total'].value), 4.0)

    def test_the_simplified_form_moves_the_module_it_was_given(self, signal) -> None:
        module = Counter(step=1.0)
        module(signal=signal)
        outputs, returned = _run_simplified(module, {'signal': signal})
        assert returned is module
        assert np.allclose(np.asarray(module.total.value), 2.0)
        assert np.allclose(np.asarray(outputs['total'].value), 2.0)

    def test_the_outputs_are_finite(self, signal) -> None:
        outputs = Counter()(signal=signal)
        for payload in outputs.values():
            assert jnp.sum(jnp.isnan(payload.value)) == 0
            assert jnp.sum(jnp.isinf(payload.value)) == 0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRegistration:
    """
        What registering a module buys.
    """

    def test_the_module_is_in_the_registry(self) -> None:
        entry = spark.REGISTRY.Components.get('probe_counter')
        assert entry is not None
        assert entry.get_cls() is Counter

    def test_it_is_found_by_its_class(self) -> None:
        assert spark.REGISTRY.Components.get_by_cls(Counter).name == 'probe_counter'

    def test_a_name_that_is_not_registered_is_answered_with_a_default(self) -> None:
        assert spark.REGISTRY.Components.get('nothing_by_that_name') is None
        assert spark.REGISTRY.Components.get('nothing_by_that_name', 'default') == 'default'

    def test_a_class_of_another_namespace_is_not_found_here(self) -> None:
        assert spark.REGISTRY.Neurons.get_by_cls(Counter) is None

    def test_a_name_is_normalized(self) -> None:
        assert spark.REGISTRY.Components.get('N2NDelays') is spark.REGISTRY.Components.get('n2n_delays')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestProperties:
    """
        A property exposed to the rest of the framework.
    """

    def test_a_property_is_declared_and_readable(self) -> None:
        neuron = spark.nn.neurons.ALIFNeuron(units=(8,))
        neuron(in_spikes=spark.SpikeArray(jnp.zeros((8,), dtype=jnp.uint8)))
        assert 'inhibition_mask' in neuron.get_properties()
        assert isinstance(neuron.inhibition_mask, spark.BooleanMask)
        assert neuron.inhibition_mask.value.shape == (8,)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
