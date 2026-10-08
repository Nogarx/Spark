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

class TestNewSeeds:
    """
        Reseeding a configuration, which is what makes a run repeatable.
    """

    @staticmethod
    def _seeds(config):
        return {spec.name: spec.config.seed for spec in config.modules_specs}

    def test_the_modules_are_reseeded(self) -> None:
        config = spark.nn.neurons.ALIFNeuronConfig(units=(8,))
        before, after = self._seeds(config), self._seeds(config.with_new_seeds(seed=42))
        assert all(before[name] != after[name] for name in before)

    def test_the_controller_is_reseeded(self) -> None:
        config = spark.nn.neurons.ALIFNeuronConfig(units=(8,))
        assert config.with_new_seeds(seed=42).seed != config.seed

    def test_the_same_seed_gives_the_same_model(self) -> None:
        config = spark.nn.neurons.ALIFNeuronConfig(units=(8,))
        first, second = config.with_new_seeds(seed=42), config.with_new_seeds(seed=42)
        assert first.seed == second.seed
        assert self._seeds(first) == self._seeds(second)

    def test_another_seed_gives_another_model(self) -> None:
        config = spark.nn.neurons.ALIFNeuronConfig(units=(8,))
        first, second = config.with_new_seeds(seed=42), config.with_new_seeds(seed=7)
        assert self._seeds(first) != self._seeds(second)

    def test_the_original_is_left_alone(self) -> None:
        config = spark.nn.neurons.ALIFNeuronConfig(units=(8,))
        before = self._seeds(config)
        config.with_new_seeds(seed=42)
        assert self._seeds(config) == before

    def test_it_reaches_every_level(self) -> None:
        pool = lambda name: spark.ModuleSpecs(
            name=name, module_cls=spark.nn.neurons.ALIFNeuron,
            inputs={'in_spikes': [spark.PortMap(origin='__call__', port='in_spikes')]},
            config=spark.nn.neurons.ALIFNeuronConfig(units=(8,)))
        brain = spark.nn.BrainConfig(modules_specs=[pool('a'), pool('b')])
        reseeded = brain.with_new_seeds(seed=42)
        for before_spec, after_spec in zip(brain.modules_specs, reseeded.modules_specs):
            assert before_spec.config.seed != after_spec.config.seed
            assert all(b.config.seed != a.config.seed
                       for b, a in zip(before_spec.config.modules_specs, after_spec.config.modules_specs))
        # NOTE: Two pools reseeded from one seed must not end up identical to each other.
        assert reseeded.modules_specs[0].config.seed != reseeded.modules_specs[1].config.seed

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestDerivedSeeds:
    """
        A module given no seed takes one derived from the controller holding it, which is what makes a seeded brain
        the same in every process.
    """

    @staticmethod
    def _brain(brain_seed: int, **first) -> spark.nn.BrainConfig:
        pool = lambda name, **kwargs: spark.ModuleSpecs(
            name=name, module_cls=spark.nn.neurons.ALIFNeuron,
            inputs={'in_spikes': [spark.PortMap(origin='__call__', port='in_spikes')]},
            config=spark.nn.neurons.ALIFNeuronConfig(units=(8,), **kwargs))
        return spark.nn.BrainConfig(seed=brain_seed, modules_specs=[pool('a', **first), pool('b')])

    @staticmethod
    def _seeds(config) -> list:
        return [(spec.config.seed, [inner.config.seed for inner in spec.config.modules_specs]) for spec in config.modules_specs]

    def test_a_configuration_on_its_own_leaves_its_seeds_unset(self) -> None:
        config = spark.nn.neurons.ALIFNeuronConfig(units=(8,))
        assert config.seed is None
        assert all(spec.config.seed is None for spec in config.modules_specs)

    def test_a_seeded_brain_seeds_every_level(self) -> None:
        assert all(seed is not None for pool, components in self._seeds(self._brain(7)) for seed in (pool, *components))

    def test_the_same_seed_gives_the_same_seeds(self) -> None:
        assert self._seeds(self._brain(7)) == self._seeds(self._brain(7))

    def test_the_seeds_are_the_same_in_every_process(self) -> None:
        # NOTE: The derivation reads nothing but the seed and the name.
        from spark.core.config import _derived_seed
        assert (_derived_seed(7, 'a'), _derived_seed(7, 'b'), _derived_seed(8, 'a')) == (1060464539, 2550262055, 1284789400)

    def test_another_seed_gives_other_seeds(self) -> None:
        assert self._seeds(self._brain(7)) != self._seeds(self._brain(8))

    def test_two_modules_of_one_brain_differ(self) -> None:
        (first, first_components), (second, second_components) = self._seeds(self._brain(7))
        assert first != second
        assert all(a != b for a, b in zip(first_components, second_components))

    def test_a_seed_given_is_kept(self) -> None:
        config = self._brain(7, seed=123)
        assert config.modules_specs[0].config.seed == 123
        assert self._seeds(config)[1] == self._seeds(self._brain(7))[1]

    def test_a_file_keeps_its_seeds(self, tmp_path) -> None:
        config = self._brain(7).with_new_seeds(seed=42)
        config.to_file(str(tmp_path / 'brain.scfg'))
        assert self._seeds(spark.nn.BrainConfig.from_file(str(tmp_path / 'brain.scfg'))) == self._seeds(config)

    def test_a_module_on_its_own_draws_a_seed_and_keeps_it(self) -> None:
        soma = spark.nn.somas.LeakySoma(units=(4,))
        neuron = spark.nn.neurons.ALIFNeuron(units=(8,))
        assert soma.config.seed is not None
        assert neuron.config.seed is not None and all(spec.config.seed is not None for spec in neuron.config.modules_specs)

    def test_two_pools_of_one_brain_draw_different_delays_and_weights(self, spikes) -> None:
        brain = spark.nn.Brain(config=self._brain(7))
        brain(in_spikes=spikes(4))
        assert not np.array_equal(brain.a.delays.kernel.value, brain.b.delays.kernel.value)
        assert not np.array_equal(brain.a.synapses.kernel.value, brain.b.synapses.kernel.value)

    def test_the_pools_of_a_registered_neuron_differ(self, spikes) -> None:
        name = f'DerivedSeedsNeuron{id(self)}'
        spark.register_neuron_from_config(name, spark.nn.neurons.ALIFNeuronConfig(units=(8,), seed=5))
        neuron_cls = spark.REGISTRY.Neurons.get(name).get_cls()
        pool = lambda pool_name: spark.ModuleSpecs(
            name=pool_name, module_cls=neuron_cls, config=neuron_cls.default_config(),
            inputs={'in_spikes': [spark.PortMap(origin='__call__', port='in_spikes')]})
        brain = spark.nn.Brain(config=spark.nn.BrainConfig(seed=7, modules_specs=[pool('a'), pool('b')]))
        brain(in_spikes=spikes(4))
        assert not np.array_equal(brain.a.delays.kernel.value, brain.b.delays.kernel.value)

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

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestUpdate:
    """
        Parts of a configuration read and replaced by their address, or by a pattern matching several.
    """

    @staticmethod
    def _brain() -> spark.nn.BrainConfig:
        pool = lambda name: spark.ModuleSpecs(
            name=name, module_cls=spark.nn.neurons.ALIFNeuron,
            inputs={'in_spikes': [spark.PortMap(origin='__call__', port='in_spikes')]},
            config=spark.nn.neurons.ALIFNeuronConfig(units=(8,)))
        return spark.nn.BrainConfig(seed=7, modules_specs=[pool('a_excitatory'), pool('b_excitatory'), pool('a_inhibitory')])

    def test_an_address_reads_a_field_or_the_configuration_of_a_module(self) -> None:
        config = self._brain()
        assert config['a_excitatory.soma.threshold'] == -40.0
        assert type(config['a_excitatory.soma']) is spark.nn.somas.AdaptiveLeakySomaConfig
        assert config['a_excitatory.synapses.kernel.density'] == 0.2

    def test_the_addresses_are_every_module_nested_configuration_and_field(self) -> None:
        addresses = self._brain().addresses()
        assert {'seed', 'a_excitatory', 'a_excitatory.soma', 'a_excitatory.soma.threshold', 'a_excitatory.synapses.kernel',
                'a_excitatory.synapses.kernel.density'} <= set(addresses)
        assert not any(address.endswith('modules_specs') for address in addresses)

    def test_a_pattern_selects_addresses(self) -> None:
        config = self._brain()
        assert config.addresses('*_excitatory.soma') == ('a_excitatory.soma', 'b_excitatory.soma')
        assert config.addresses('**.soma.threshold') == ('a_excitatory.soma.threshold', 'b_excitatory.soma.threshold', 'a_inhibitory.soma.threshold')

    def test_a_field_is_set_in_a_copy(self) -> None:
        config = self._brain()
        updated = config.update({'a_excitatory.soma.threshold': -45.0})
        assert updated['a_excitatory.soma.threshold'] == -45.0
        assert (config['a_excitatory.soma.threshold'], updated['b_excitatory.soma.threshold']) == (-40.0, -40.0)

    def test_a_pattern_sets_every_part_it_matches(self) -> None:
        updated = self._brain().update({'*_excitatory.soma.threshold': -45.0})
        assert [updated[f'{pool}.soma.threshold'] for pool in ('a_excitatory', 'b_excitatory', 'a_inhibitory')] == [-45.0, -45.0, -40.0]

    def test_a_module_takes_a_configuration_of_its_class(self) -> None:
        soma = spark.nn.somas.AdaptiveLeakySomaConfig(threshold=-42.0)
        updated = self._brain().update({'*_excitatory.soma': soma})
        assert (updated['a_excitatory.soma.threshold'], updated['b_excitatory.soma.threshold']) == (-42.0, -42.0)
        assert soma.seed is None

    def test_two_modules_given_one_configuration_take_different_seeds_the_same_each_time(self) -> None:
        soma = spark.nn.somas.AdaptiveLeakySomaConfig(threshold=-42.0)
        first, second = (self._brain().update({'*_excitatory.soma': soma}) for _ in range(2))
        assert first['a_excitatory.soma.seed'] != first['b_excitatory.soma.seed']
        assert (first['a_excitatory.soma.seed'], first['b_excitatory.soma.seed']) == (second['a_excitatory.soma.seed'], second['b_excitatory.soma.seed'])

    def test_the_controller_hands_its_units_and_dt_again(self) -> None:
        synapses = spark.nn.synapses.TracedSynapsesConfig.partial(tau=4.0)
        updated = self._brain().update({'a_excitatory.synapses': synapses})
        assert (updated['a_excitatory.synapses.units'], updated['a_excitatory.synapses.dt']) == ((8,), 1.0)

    def test_a_module_specs_replaces_the_class_and_the_wiring(self, spikes) -> None:
        lif = spark.ModuleSpecs(name='a_inhibitory', module_cls=spark.nn.neurons.LIFNeuron,
                                inputs={'in_spikes': [spark.PortMap(origin='a_excitatory', port='out_spikes')]},
                                config=spark.nn.neurons.LIFNeuronConfig(units=(8,)))
        updated = self._brain().update({'a_inhibitory': lif})
        spec = next(spec for spec in updated.modules_specs if spec.name == 'a_inhibitory')
        assert spec.module_cls is spark.nn.neurons.LIFNeuron and spec.inputs['in_spikes'][0].origin == 'a_excitatory'
        brain = spark.nn.Brain(config=updated)
        brain(in_spikes=spikes(4))
        assert type(brain.a_inhibitory) is spark.nn.neurons.LIFNeuron

    def test_the_pools_given_one_configuration_draw_different_values(self, spikes) -> None:
        kernel = spark.nn.initializers.UniformInitializerConfig(scale=3.0)
        brain = spark.nn.Brain(config=self._brain().update({'*_excitatory.synapses.kernel': kernel}))
        brain(in_spikes=spikes(4))
        a, b = (np.asarray(pool.synapses.kernel.value) for pool in (brain.a_excitatory, brain.b_excitatory))
        assert 0 < a.min() and a.max() <= 3.0 and not np.array_equal(a, b)

    def test_a_module_keeps_its_name(self) -> None:
        renamed = spark.ModuleSpecs(name='other', module_cls=spark.nn.neurons.LIFNeuron,
                                    inputs={'in_spikes': [spark.PortMap(origin='__call__', port='in_spikes')]})
        with pytest.raises(ValueError, match='a module keeps its name'):
            self._brain().update({'a_inhibitory': renamed})

    @pytest.mark.parametrize('value, match', [
        (spark.nn.somas.LeakySomaConfig(), 'configured by AdaptiveLeakySomaConfig, not by LeakySomaConfig'),
        (3, 'replaced by a configuration or a ModuleSpecs'),
    ])
    def test_a_module_refuses_anything_else(self, value, match) -> None:
        with pytest.raises(TypeError, match=match):
            self._brain().update({'a_excitatory.soma': value})

    def test_an_address_matching_nothing_is_refused_with_the_closest_ones(self) -> None:
        with pytest.raises(KeyError, match='Did you mean: .*a_excitatory.soma.threshold'):
            self._brain().update({'a_excitatory.soma.thresold': -45.0})

    def test_a_pattern_matching_a_part_and_a_part_within_it_is_refused(self) -> None:
        with pytest.raises(ValueError, match='ambiguous'):
            self._brain().update({'**': 1})

    def test_an_index_is_an_address_not_a_pattern(self) -> None:
        with pytest.raises(KeyError, match='addresses'):
            self._brain()['*.soma']

    def test_ipython_completes_the_addresses(self) -> None:
        config = self._brain()
        assert config._ipython_key_completions_() == config.addresses()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
