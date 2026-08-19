#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import typing as tp
import dataclasses as dc
import jax.numpy as jnp
import numpy as np
import spark
from spark.core.config import unflatten_kwargs
from spark.nn.interfaces.input.topological import TopologicalLinearSpikerConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ProbeOutput(tp.TypedDict):
    probe_output: spark.FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ProbeConfig(spark.nn.Config):
    foo: int
    bar: float = 2.0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ProbeModule(spark.nn.Module):
    config: ProbeConfig

    def __init__(self, config: ProbeConfig = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.foo = spark.Constant(jnp.array(self.config.foo))
        self.bar = spark.Variable(jnp.array(self.config.bar))

    def __call__(self, probe_input: spark.FloatArray) -> ProbeOutput:
        return {'probe_output': spark.FloatArray(self.foo + self.bar + probe_input)}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ValidatedConfig(spark.nn.Config):
    foo: int
    bar: float = dc.field(
        default = 2.0,
        metadata = {
            'units': 'nA',
            'valid_types': tp.Any,
            'validators': [
                spark.validation.TypeValidator,
                spark.validation.PositiveValidator,
            ],
            'description': 'A bar that has to be positive.',
        }
    )
    baz: list[int] = dc.field(default_factory = lambda: [i for i in range(10)])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ChildConfig(spark.nn.Config):
    foo: int
    bar: float = 2.0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ParentConfig(spark.nn.Config):
    foo: int
    child_bar: ChildConfig
    child_baz: ChildConfig = dc.field(
        default_factory = lambda **kwargs: ChildConfig(**{**{'foo': 1}, **kwargs})
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@spark.register_module('probe_named_module')
class ProbeNamedModule(spark.nn.Module):
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class UnconventionalConfig(spark.nn.Config):
    __class_ref__ = 'probe_named_module'

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestInitialization:
    """
        The three ways a module is given its configuration (tutorial #1).
    """

    def test_from_keyword_arguments(self) -> None:
        module = ProbeModule(foo=1)
        assert module.config.foo == 1
        assert module.config.bar == 2.0

    def test_from_a_configuration(self) -> None:
        module = ProbeModule(config=ProbeConfig(foo=1))
        assert module.config.foo == 1

    def test_keyword_arguments_win_over_the_configuration(self) -> None:
        module = ProbeModule(config=ProbeConfig(foo=1), bar=-1)
        assert (module.config.foo, module.config.bar) == (1, -1)

    def test_a_required_field_is_required(self) -> None:
        with pytest.raises(TypeError):
            ProbeConfig()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestValidation:
    """
        Validators and metadata carried by a field.
    """

    def test_a_validator_refuses_a_bad_value(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            ValidatedConfig(foo=1, bar=-1)

    def test_a_value_that_cannot_be_promoted_is_refused(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            ValidatedConfig(foo=1, bar=[2.0])

    def test_the_validators_of_a_field_are_reachable(self) -> None:
        field = {f.name: f for f in dc.fields(ValidatedConfig)}['bar']
        validator = spark.validation.PositiveValidator(field)
        validator.validate(1.0)
        with pytest.raises((TypeError, ValueError)):
            validator.validate(-1.0)

    def test_a_valid_value_passes(self) -> None:
        config = ValidatedConfig(foo=1, bar=3.0)
        assert config.bar == 3.0

    def test_a_mutable_default_is_frozen_and_not_shared(self) -> None:
        first, second = ValidatedConfig(foo=1), ValidatedConfig(foo=2)
        assert first.baz == tuple(range(10)) == second.baz
        assert not isinstance(first.baz, list)

    def test_metadata_survives(self) -> None:
        field = {f.name: f for f in dc.fields(ValidatedConfig)}['bar']
        assert field.metadata['units'] == 'nA'
        assert field.metadata['description'] == 'A bar that has to be positive.'
        assert spark.validation.PositiveValidator in field.metadata['validators']

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestWhatIsNotValidated:
    """
        What the validators deliberately stay quiet about.
    """

    def test_a_value_that_is_not_set_yet(self) -> None:
        partial = spark.nn.synapses.LinearSynapsesConfig.partial()
        assert partial.units is None
        assert spark.nn.NeuronConfig.partial().units is None

    def test_a_field_holding_an_initializer(self) -> None:
        from spark.nn.initializers import ConstantInitializerConfig
        config = spark.nn.synapses.LinearSynapsesConfig(units=(4,), kernel=ConstantInitializerConfig(scale=2.0))
        assert isinstance(config.kernel, ConstantInitializerConfig)

    def test_an_annotation_that_cannot_be_read(self) -> None:
        field = {f.name: f for f in dc.fields(spark.nn.somas.LeakySomaConfig)}['threshold']
        validator = spark.validation.TypeValidator(field, valid_types=('float', 'Initializer'))
        validator.validate('not a number at all')

    def test_a_value_that_is_still_being_traced(self) -> None:
        import jax
        @jax.jit
        def build(dt):
            return spark.nn.somas.LeakySomaConfig(dt=dt).threshold
        assert float(build(jnp.asarray(1.0))) == pytest.approx(-40.0)

    def test_a_numpy_array_where_a_device_array_is_declared(self) -> None:
        from spark.nn.interfaces.input.topological import TopologicalLinearSpikerConfig
        config = TopologicalLinearSpikerConfig(glue=np.array(0), mins=np.array(-1), maxs=np.array(1))
        assert np.asarray(config.glue).tolist() == 0

    def test_an_integer_where_a_float_is_declared(self) -> None:
        assert spark.nn.somas.LeakySomaConfig(dt=1).dt == 1

    def test_validation_can_be_suspended(self) -> None:
        with spark.validation.NoValidation():
            config = ValidatedConfig(foo=1, bar=-1)
        assert config.bar == -1
        with pytest.raises((TypeError, ValueError)):
            ValidatedConfig(foo=1, bar=-1)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestInitializerFields:
    """
        A field that may hold an initializer instead of a value.
    """

    def test_a_field_that_takes_an_initializer_is_marked(self) -> None:
        fields = {f.name: f for f in dc.fields(spark.nn.synapses.LinearSynapsesConfig)}
        assert fields['kernel'].metadata['allows_init']
        assert not fields['units'].metadata['allows_init']

    def test_an_initializer_configuration_keeps_the_class_it_was_given(self) -> None:
        from spark.nn.initializers import ConstantInitializerConfig
        config = spark.nn.synapses.LinearSynapsesConfig(units=(4,), kernel=ConstantInitializerConfig(scale=2.0))
        assert isinstance(config.kernel, ConstantInitializerConfig)
        assert config.kernel.scale == 2.0

    def test_it_survives_a_merge(self) -> None:
        from spark.nn.initializers import ConstantInitializerConfig
        config = spark.nn.synapses.LinearSynapsesConfig(units=(4,), kernel=ConstantInitializerConfig(scale=2.0))
        merged = config.merge(units=(8,))
        assert isinstance(merged.kernel, ConstantInitializerConfig)
        assert merged.kernel.scale == 2.0

    def test_the_default_is_left_alone(self) -> None:
        from spark.nn.initializers import SparseUniformInitializerConfig
        assert isinstance(spark.nn.synapses.LinearSynapsesConfig(units=(4,)).kernel,
                          SparseUniformInitializerConfig)

    def test_the_initializer_namespace_builds_the_array(self) -> None:
        import jax
        from spark.nn.initializers import ConstantInitializerConfig
        config = spark.nn.synapses.LinearSynapsesConfig(units=(4,), kernel=ConstantInitializerConfig(scale=2.0))
        kernel = config.init.kernel(key=jax.random.key(0), shape=(4, 3))
        assert kernel.shape == (4, 3)
        assert float(np.asarray(kernel).max()) == pytest.approx(2.0)

    def test_the_module_is_built_with_the_initializer_it_was_given(self) -> None:
        import jax.numpy as jnp
        from spark.nn.initializers import ConstantInitializerConfig
        config = spark.nn.synapses.LinearSynapsesConfig(units=(4,), kernel=ConstantInitializerConfig(scale=2.0))
        synapses = spark.nn.synapses.LinearSynapses(config=config)
        synapses(spikes=spark.SpikeArray(jnp.zeros((3,), dtype=jnp.uint8)))
        assert float(np.asarray(synapses.kernel.value).max()) == pytest.approx(2.0)

    def test_a_value_is_answered_as_it_is(self) -> None:
        config = spark.nn.somas.LeakySomaConfig(units=(4,), threshold=-40.0)
        assert config.init.threshold() == -40.0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestNestedConfigurations:
    """
        The four ways a nested configuration is filled in (tutorial #1).
    """

    def test_by_nested_keyword(self) -> None:
        config = ParentConfig(foo=1, child_bar__foo=2, child_baz__foo=3)
        assert (config.foo, config.child_bar.foo, config.child_baz.foo) == (1, 2, 3)

    def test_by_dictionary(self) -> None:
        config = ParentConfig(foo=4, child_bar={'foo': 5}, child_baz={'foo': 6})
        assert (config.foo, config.child_bar.foo, config.child_baz.foo) == (4, 5, 6)

    def test_by_shared_argument(self) -> None:
        config = ParentConfig(_s_foo=7)
        assert (config.foo, config.child_bar.foo, config.child_baz.foo) == (7, 7, 7)

    def test_by_direct_instance(self) -> None:
        config = ParentConfig(foo=8, child_bar=ChildConfig(foo=9), child_baz=ChildConfig(foo=10))
        assert (config.foo, config.child_bar.foo, config.child_baz.foo) == (8, 9, 10)

    def test_a_nested_default_is_not_shared_between_configurations(self) -> None:
        first, second = ParentConfig(_s_foo=1), ParentConfig(_s_foo=2)
        assert first.child_bar is not second.child_bar
        assert (first.child_bar.foo, second.child_bar.foo) == (1, 2)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestNestedKeywords:
    """
        How a flattened keyword is read.
    """

    @pytest.mark.parametrize('kwargs, expected', [
        ({'kernel__scale': 1, 'kernel_size': 7}, {'kernel_size': 7, 'kernel': {'scale': 1}}),
        ({'kernel__scale': 1, 'kernel__density': 0.2}, {'kernel': {'scale': 1, 'density': 0.2}}),
        ({'a__b__c': 1, 'a__b_d': 2}, {'a': {'b_d': 2, 'b': {'c': 1}}}),
        ({'__metadata__': {'x': 1}, 'plain': 2}, {'__metadata__': {'x': 1}, 'plain': 2}),
        ({'units': (4,)}, {'units': (4,)}),
    ])
    def test_unflattening(self, kwargs: dict, expected: dict) -> None:
        assert unflatten_kwargs(kwargs) == expected

    def test_a_shared_argument_reaches_every_level(self) -> None:
        assert unflatten_kwargs({'_s_units': (8,), 'kernel__scale': 2}) == {
            'units': (8,), 'kernel': {'units': (8,), 'scale': 2},
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestPartialAndMerge:
    """
        The two ways a configuration is derived from another.
    """

    def test_partial_leaves_what_is_unset_as_none(self) -> None:
        config = ProbeConfig.partial()
        assert config.foo is None
        assert config.bar == 2.0

    def test_merge_answers_with_a_new_configuration(self) -> None:
        config = ProbeConfig(foo=1)
        merged = config.merge(foo=5)
        assert (merged.foo, config.foo) == (5, 1)
        assert merged is not config

    def test_merge_keeps_what_it_was_not_given(self) -> None:
        merged = ProbeConfig(foo=1, bar=9.0).merge(foo=2)
        assert merged.bar == 9.0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestControllerSharedFields:
    """
        A controller hands its own shape and its own clock to every module it holds.
    """

    def test_units_reach_the_modules(self) -> None:
        config = spark.nn.neurons.ALIFNeuronConfig(units=(8,))
        modules = {m.name: m.config for m in config.modules_specs}
        assert modules['delays'].units == (8,)
        assert modules['synapses'].units == (8,)

    def test_dt_reaches_the_modules(self) -> None:
        config = spark.nn.neurons.ALIFNeuronConfig(units=(8,), dt=2.0)
        assert all(m.config.dt == 2.0 for m in config.modules_specs)

    def test_a_module_is_addressed_by_its_name(self) -> None:
        config = spark.nn.neurons.ALIFNeuronConfig(units=(8,), synapses__kernel__scale=2000, synapses__tau=11.0)
        synapses = {m.name: m.config for m in config.modules_specs}['synapses']
        assert (synapses.kernel.scale, synapses.tau) == (2000, 11.0)

    def test_naming_a_module_wins_for_what_the_controller_does_not_share(self) -> None:
        config = spark.nn.neurons.ALIFNeuronConfig(_s_units=(8,), synapses__tau=11.0)
        modules = {m.name: m.config for m in config.modules_specs}
        assert modules['synapses'].tau == 11.0
        assert modules['synapses'].units == (8,)

    def test_a_shape_of_its_own_does_not_survive_the_controller(self) -> None:
        config = spark.nn.neurons.ALIFNeuronConfig(units=(8,), synapses__units=(4,))
        modules = {m.name: m.config for m in config.modules_specs}
        assert modules['synapses'].units == (8,)

    def test_a_brain_hands_its_clock_to_every_pool(self) -> None:
        pool = lambda name: spark.ModuleSpecs(
            name=name, module_cls=spark.nn.neurons.ALIFNeuron,
            inputs={'in_spikes': [spark.PortMap(origin='__call__', port='in_spikes')]},
            config=spark.nn.neurons.ALIFNeuronConfig(units=(8,)))
        config = spark.nn.BrainConfig(dt=0.5, modules_specs=[pool('a'), pool('b')])
        for spec in config.modules_specs:
            assert spec.config.dt == 0.5
            assert all(inner.config.dt == 0.5 for inner in spec.config.modules_specs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestModuleSpecsFields:
    """
        A field holding module specifications is recognized however it is spelled.
    """

    SPEC = spark.ModuleSpecs(
        name='synapses', module_cls=spark.nn.synapses.LinearSynapses,
        inputs={'spikes': [spark.PortMap(origin='__call__', port='in_spikes')]})

    @pytest.mark.parametrize('annotation', [
        list[spark.ModuleSpecs],
        tuple[spark.ModuleSpecs, ...],
        'tuple[ModuleSpecs, ...]',
        tp.List['spark.ModuleSpecs'],
        list,
    ])
    def test_the_specifications_are_filled_in(self, annotation) -> None:
        config_cls = type('SpelledConfig', (spark.nn.NeuronConfig,), {
            '__annotations__': {'modules_specs': annotation},
            'modules_specs': [self.SPEC],
        })
        specs = config_cls(units=(4,)).modules_specs
        assert [spec.config.units for spec in specs] == [(4,)]

    def test_the_default_is_not_shared_between_configurations(self) -> None:
        config_cls = type('SharedSpecsConfig', (spark.nn.NeuronConfig,), {
            '__annotations__': {'modules_specs': list[spark.ModuleSpecs]},
            'modules_specs': [self.SPEC],
        })
        first, second = config_cls(units=(4,)), config_cls(units=(9,))
        assert first.modules_specs[0] is not second.modules_specs[0]
        assert (first.modules_specs[0].config.units, second.modules_specs[0].config.units) == ((4,), (9,))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestArrayFields:
    """
        A configuration that holds an array.
    """

    @staticmethod
    def _config() -> TopologicalLinearSpikerConfig:
        return TopologicalLinearSpikerConfig(glue=jnp.array(0), mins=jnp.array(-1), maxs=jnp.array(1))

    def test_an_array_reads_as_an_array(self) -> None:
        config = self._config()
        assert np.asarray(config.glue).tolist() == 0
        assert config.glue.shape == ()

    def test_the_initializer_namespace_answers_with_the_array(self) -> None:
        assert np.asarray(self._config().init.glue()).tolist() == 0

    def test_the_array_is_not_a_leaf_of_the_module(self) -> None:
        from spark.nn.interfaces.input.topological import TopologicalLinearSpiker
        module = TopologicalLinearSpiker(config=self._config())
        module(signal=spark.FloatArray(jnp.zeros((4,), dtype=jnp.float16)))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestClassReference:
    """
        The module a configuration stands for.
    """

    def test_by_naming_convention(self) -> None:
        assert spark.nn.somas.LeakySomaConfig(units=(4,)).class_ref is spark.nn.somas.LeakySoma

    def test_by_explicit_reference(self) -> None:
        assert UnconventionalConfig().class_ref is ProbeNamedModule

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
