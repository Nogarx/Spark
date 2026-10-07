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
from math import prod
from spark.core.utils import validate_shape

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

SYNAPSES_SPEC = spark.ModuleSpecs(
    name = 'synapses',
    module_cls = spark.nn.synapses.LinearSynapses,
    inputs = {
        'spikes': [spark.PortMap(origin='__call__', port='incoming_spikes')],
    },
)

SOMA_SPEC = spark.ModuleSpecs(
    name = 'soma',
    module_cls = spark.nn.somas.LeakySoma,
    inputs = {
        'current': [spark.PortMap(origin='synapses', port='currents')],
        'inhibition_mask': [spark.PortMap(origin='__self__', port='inhibition_mask', is_property=True)],
    },
    outputs = {
        'my_awesome_spikes': 'spikes',
    },
)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@spark.register_config
class ProbeLIFNeuronConfig(spark.nn.NeuronConfig):
    modules_specs: list[spark.ModuleSpecs] = [SYNAPSES_SPEC, SOMA_SPEC]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@spark.register_neuron
class ProbeLIFNeuron(spark.nn.Neuron):
    config: ProbeLIFNeuronConfig

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ProbeNeuronOutput(tp.TypedDict):
    out_spikes: spark.SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@spark.register_config
class ProbeHandwiredConfig(spark.nn.Config):
    units: tuple[int, ...]
    inhibitory_rate: float = 0.2
    soma: spark.nn.somas.LeakySomaConfig
    synapses: spark.nn.synapses.LinearSynapsesConfig

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@spark.register_module
class ProbeHandwired(spark.nn.Module):
    config: ProbeHandwiredConfig

    def __init__(self, config: ProbeHandwiredConfig | None = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.units = validate_shape(self.config.units)
        self._units = prod(self.units)
        inhibitory_units = int(self._units * self.config.inhibitory_rate)
        indices = jax.random.permutation(self.get_rng_keys(1), jnp.arange(self._units), independent=True)[:inhibitory_units]
        mask = jnp.zeros((self._units,), dtype=jnp.bool).at[indices].set(True).reshape(self.units)
        self._inhibition_mask = spark.Constant(mask, dtype=jnp.bool)

    @spark.property
    def inhibition_mask(self,) -> spark.BooleanMask:
        return spark.BooleanMask(self._inhibition_mask.value)

    def build(self, **abc_args: spark.SparkPayload):
        self.soma = self.config.soma.class_ref(config=self.config.soma)
        self.synapses = self.config.synapses.class_ref(config=self.config.synapses)

    def __call__(self, incoming_spikes: spark.SpikeArray) -> ProbeNeuronOutput:
        currents = self.synapses(spikes=incoming_spikes)['currents']
        return {'out_spikes': self.soma(current=currents, inhibition_mask=self.inhibition_mask)['spikes']}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.jit
def _run_split(graph: nnx.GraphDef, state: nnx.State, inputs: dict) -> tuple[dict, nnx.State]:
    model = spark.merge(graph, state)
    outputs = model(**inputs)
    _, state = spark.split((model))
    return outputs, state

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _brain(pool_units=(16,), second_units=(8,), outputs=3):
    """
        The brain of tutorial #4, shrunk: a spiker, two pools wired forward with a self connection, and an
        integrator reading the second pool.
    """
    pool = lambda name, units, sources: spark.ModuleSpecs(
        name = name,
        module_cls = spark.nn.neurons.ALIFNeuron,
        inputs = {'in_spikes': [spark.PortMap(origin=origin, port=port) for origin, port in sources]},
        config = spark.nn.neurons.ALIFNeuronConfig(
            _s_units = units, synapses__kernel__scale = 200, inhibitory_rate = 0.3,
        ),
    )
    return spark.nn.BrainConfig(modules_specs=[
        spark.ModuleSpecs(
            name = 'spiker',
            module_cls = spark.nn.interfaces.PoissonSpiker,
            inputs = {'signal': [spark.PortMap(origin='__call__', port='signal')]},
        ),
        pool('first_pool', pool_units, [('spiker', 'spikes')]),
        pool('second_pool', second_units, [('first_pool', 'out_spikes'), ('second_pool', 'out_spikes')]),
        spark.ModuleSpecs(
            name = 'integrator',
            module_cls = spark.nn.interfaces.ExponentialIntegrator,
            inputs = {'spikes': [spark.PortMap(origin='second_pool', port='out_spikes')]},
            outputs = {'my_awesome_signal': 'signal'},
            config = spark.nn.interfaces.ExponentialIntegratorConfig(num_outputs=outputs),
        ),
    ])

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestNeuronFromSpecs:
    """
        A neuron assembled out of module specifications (tutorial #3).
    """

    def test_it_runs_and_answers_with_what_the_specs_named(self, spikes) -> None:
        neuron = ProbeLIFNeuron(units=(8,), inhibitory_rate=0.2)
        outputs = neuron(incoming_spikes=spikes(16))
        assert set(outputs) == {'my_awesome_spikes'}
        assert outputs['my_awesome_spikes'].value.shape == (8,)

    def test_the_components_are_reachable_by_name(self, spikes) -> None:
        neuron = ProbeLIFNeuron(units=(8,))
        neuron(incoming_spikes=spikes(16))
        assert neuron.synapses.kernel.shape == (8, 16)

    def test_the_shape_of_the_pool_reaches_its_components(self) -> None:
        neuron = ProbeLIFNeuron(units=(8,))
        modules = {m.name: m.config for m in neuron.config.modules_specs}
        assert modules['synapses'].units == (8,)

    def test_the_inhibition_mask_matches_the_rate(self) -> None:
        neuron = ProbeLIFNeuron(units=(100,), inhibitory_rate=0.2)
        assert int(np.asarray(neuron.inhibition_mask.value).sum()) == 20

    def test_it_survives_a_jitted_step(self, spikes) -> None:
        neuron = ProbeLIFNeuron(units=(8,))
        inputs = {'incoming_spikes': spikes(16)}
        neuron(**inputs)
        graph, state = spark.split((neuron))
        outputs, state = _run_split(graph, state, inputs)
        assert outputs['my_awesome_spikes'].value.shape == (8,)

    def test_a_checkpoint_gives_back_the_neuron(self, spikes, tmp_path) -> None:
        neuron = ProbeLIFNeuron(units=(8,), inhibitory_rate=0.2)
        for _ in range(3):
            neuron(incoming_spikes=spikes(16))
        restored = spark.nn.Neuron.from_checkpoint(neuron.checkpoint(tmp_path / 'neuron', verbose=False), verbose=False)
        assert type(restored) is ProbeLIFNeuron
        for got, want in zip(jax.tree.leaves(spark.split((restored))[1]), jax.tree.leaves(spark.split((neuron))[1])):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_a_checkpoint_is_checked_against_its_sha256(self, spikes, tmp_path, capsys) -> None:
        import hashlib
        neuron = ProbeLIFNeuron(units=(8,), inhibitory_rate=0.2)
        neuron(incoming_spikes=spikes(16))
        path = neuron.checkpoint(tmp_path / 'neuron')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Logged when written, and accepted however it is written out.
        assert digest in capsys.readouterr().out
        assert type(spark.nn.Neuron.from_checkpoint(path, verbose=False, sha256=f' {digest.upper()} ')) is ProbeLIFNeuron
        with pytest.raises(RuntimeError, match='not the file published'):
            spark.nn.Neuron.from_checkpoint(path, verbose=False, sha256='0' * 64)
        # Changed after it was published: refused.
        with open(path, 'ab') as file:
            file.write(b'\0')
        with pytest.raises(RuntimeError, match='not the file published'):
            spark.nn.Neuron.from_checkpoint(path, verbose=False, sha256=digest)

    def test_a_checkpoint_writes_its_sha256_beside_it(self, spikes, tmp_path) -> None:
        import shutil
        import subprocess
        neuron = ProbeLIFNeuron(units=(8,), inhibitory_rate=0.2)
        neuron(incoming_spikes=spikes(16))
        path = neuron.checkpoint(tmp_path / 'neuron', verbose=False, sha256=True)
        hashed = tmp_path / 'neuron.spark.sha256'
        digest, name = hashed.read_text().split()
        assert name == 'neuron.spark' and spark.nn.Neuron.from_checkpoint(path, verbose=False, sha256=digest) is not None
        if shutil.which('sha256sum'):
            assert subprocess.run(['sha256sum', '-c', hashed.name], cwd=tmp_path, capture_output=True).returncode == 0
        # Written again without it, the one left no longer matches and is removed.
        neuron.checkpoint(tmp_path / 'neuron', overwrite=True, verbose=False)
        assert path.exists() and not hashed.exists()

    def test_a_missing_shape_is_refused(self, spikes) -> None:
        with pytest.raises(Exception):
            ProbeLIFNeuron()(incoming_spikes=spikes(16))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestHandwiredNeuron:
    """
        The same model written as a plain module (tutorial #3, second half).
    """

    def test_it_runs(self, spikes) -> None:
        neuron = ProbeHandwired(_s_units=(8,), inhibitory_rate=0.2)
        outputs = neuron(incoming_spikes=spikes(16))
        assert outputs['out_spikes'].value.shape == (8,)

    def test_the_shared_shape_reached_the_nested_configurations(self) -> None:
        neuron = ProbeHandwired(_s_units=(8,))
        assert neuron.config.synapses.units == (8,)

    def test_it_survives_a_jitted_step(self, spikes) -> None:
        neuron = ProbeHandwired(_s_units=(8,))
        inputs = {'incoming_spikes': spikes(16)}
        neuron(**inputs)
        graph, state = spark.split((neuron))
        outputs, state = _run_split(graph, state, inputs)
        assert outputs['out_spikes'].value.shape == (8,)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestBrain:
    """
        A brain built from specifications (tutorial #4).
    """

    @pytest.fixture
    def brain(self):
        return spark.nn.Brain(config=_brain())

    @staticmethod
    def _signal(size=16):
        return spark.FloatArray(jnp.zeros((size,), dtype=jnp.float16))

    def test_it_builds_and_answers_with_the_name_it_was_given(self, brain) -> None:
        outputs = brain(signal=self._signal())
        assert set(outputs) == {'my_awesome_signal'}
        assert outputs['my_awesome_signal'].value.shape == (3,)

    def test_every_pool_keeps_its_own_shape(self, brain) -> None:
        brain(signal=self._signal())
        assert (brain.first_pool.units, brain.second_pool.units) == ((16,), (8,))

    def test_a_pool_may_read_its_own_output(self, brain) -> None:
        brain(signal=self._signal())
        assert brain.second_pool.units == (8,)

    def test_it_runs_for_several_jitted_steps(self, brain) -> None:
        brain(signal=self._signal())
        graph, state = spark.split((brain))
        signal = spark.FloatArray(jnp.array(np.full((16,), 0.5), dtype=jnp.float16))
        for _ in range(5):
            outputs, state = _run_split(graph, state, {'signal': signal})
        value = np.asarray(outputs['my_awesome_signal'].value)
        assert value.shape == (3,)
        assert np.isfinite(value).all()

    def test_the_state_can_be_read_back(self, brain) -> None:
        brain(signal=self._signal())
        readout = brain.read_state((
            spark.PortMap('spiker', 'spikes'),
            spark.PortMap('first_pool', 'out_spikes'),
            spark.PortMap('second_pool', 'out_spikes'),
        ))
        assert readout['first_pool']['out_spikes'].value.shape == (16,)
        assert readout['second_pool']['out_spikes'].value.shape == (8,)

    def test_a_checkpoint_gives_back_the_brain(self, brain, tmp_path) -> None:
        signal = spark.FloatArray(jnp.array(np.full((16,), 0.5), dtype=jnp.float16))
        for _ in range(3):
            brain(signal=signal)
        path = brain.checkpoint(tmp_path / 'brain', verbose=False)
        restored = spark.nn.Brain.from_checkpoint(path, verbose=False)
        for got, want in zip(jax.tree.leaves(spark.split((restored))[1]), jax.tree.leaves(spark.split((brain))[1])):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        # The values drawn from the seeds at build time, such as the delays, come back too.
        for _ in range(3):
            np.testing.assert_array_equal(
                np.asarray(restored(signal=signal)['my_awesome_signal'].value), np.asarray(brain(signal=signal)['my_awesome_signal'].value),
            )

    def test_a_checkpoint_of_a_brain_is_not_read_as_a_neuron(self, brain, tmp_path) -> None:
        brain(signal=self._signal())
        path = brain.checkpoint(tmp_path / 'brain', verbose=False)
        with pytest.raises(RuntimeError, match='holds a Brain, not a Neuron'):
            spark.nn.Neuron.from_checkpoint(path, verbose=False)

    def test_a_module_that_names_an_origin_that_is_not_there_is_refused(self) -> None:
        config = spark.nn.BrainConfig(modules_specs=[
            spark.ModuleSpecs(
                name = 'integrator',
                module_cls = spark.nn.interfaces.ExponentialIntegrator,
                inputs = {'spikes': [spark.PortMap(origin='a_pool_that_does_not_exist', port='out_spikes')]},
                config = spark.nn.interfaces.ExponentialIntegratorConfig(num_outputs=2),
            ),
        ])
        with pytest.raises(Exception):
            spark.nn.Brain(config=config)(signal=self._signal())

    def test_a_property_named_as_an_output_is_refused(self) -> None:
        config = spark.nn.BrainConfig(modules_specs=[
            spark.ModuleSpecs(
                name = 'spiker',
                module_cls = spark.nn.interfaces.PoissonSpiker,
                inputs = {'signal': [spark.PortMap(origin='__call__', port='signal')]},
            ),
            spark.ModuleSpecs(
                name = 'synapses',
                module_cls = spark.nn.synapses.LinearSynapses,
                inputs = {'spikes': [spark.PortMap(origin='spiker', port='spikes')]},
                outputs = {'weights': 'kernel'},
                config = spark.nn.synapses.LinearSynapsesConfig(units=(4,)),
            ),
        ])
        with pytest.raises(ValueError, match='"kernel" is a property of module "synapses", not an output port'):
            spark.nn.Brain(config=config)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestBrainWithEffects:
    """
        A brain whose modules include a plasticity rule writing the kernel of a synapse.
    """

    @pytest.fixture
    def brain(self):
        config = spark.nn.BrainConfig(modules_specs=[
            spark.ModuleSpecs(
                name = 'spiker',
                module_cls = spark.nn.interfaces.PoissonSpiker,
                inputs = {'signal': [spark.PortMap(origin='__call__', port='signal')]},
            ),
            spark.ModuleSpecs(
                name = 'synapses',
                module_cls = spark.nn.synapses.LinearSynapses,
                inputs = {'spikes': [spark.PortMap(origin='spiker', port='spikes')]},
                effects = {'kernel': [spark.PortMap(origin='rule', port='kernel')]},
                config = spark.nn.synapses.LinearSynapsesConfig(units=(4,), kernel__scale=20000),
            ),
            spark.ModuleSpecs(
                name = 'soma',
                module_cls = spark.nn.somas.LeakySoma,
                inputs = {'current': [spark.PortMap(origin='synapses', port='currents')]},
                outputs = {'spikes': 'spikes'},
                config = spark.nn.somas.LeakySomaConfig(units=(4,)),
            ),
            spark.ModuleSpecs(
                name = 'rule',
                module_cls = spark.nn.plasticity.HebbianRule,
                inputs = {
                    'pre_spikes': [spark.PortMap(origin='spiker', port='spikes')],
                    'post_spikes': [spark.PortMap(origin='soma', port='spikes')],
                    'kernel': [spark.PortMap(origin='synapses', port='kernel', is_property=True)],
                },
            ),
        ])
        return spark.nn.Brain(config=config)

    def test_the_rule_writes_the_kernel_after_each_step(self, brain) -> None:
        signal = spark.FloatArray(jnp.ones((8,), dtype=jnp.float16))
        brain(signal=signal)
        first = np.asarray(brain.synapses.kernel.value)
        for _ in range(50):
            brain(signal=signal)
            np.testing.assert_array_equal(np.asarray(brain.synapses.kernel.value), np.asarray(brain._cache['rule', 'kernel'].value))
        assert not np.array_equal(np.asarray(brain.synapses.kernel.value), first)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestBrainWithArrayConfigurations:
    """
        A brain holding a module whose configuration carries arrays.
    """

    @staticmethod
    def _config(units=(16,)):
        spiker = spark.ModuleSpecs(
            name = 'spiker',
            module_cls = spark.nn.interfaces.TopologicalLinearSpiker,
            inputs = {'signal': [spark.PortMap(origin='__call__', port='drive')]},
            config = spark.nn.interfaces.TopologicalLinearSpikerConfig(
                glue = jnp.array(0), mins = jnp.array(-1), maxs = jnp.array(1),
                resolution = 128, max_freq = 200.0, tau = 30.0,
            ),
        )
        neurons = spark.ModuleSpecs(
            name = 'neurons',
            module_cls = spark.nn.neurons.ALIFNeuron,
            inputs = {'in_spikes': [
                spark.PortMap(origin='spiker', port='spikes'),
                spark.PortMap(origin='neurons', port='out_spikes'),
            ]},
            config = spark.nn.neurons.ALIFNeuronConfig(
                _s_units = units,
                synapses__kernel__scale = 3.0,
                soma__threshold_delta = 250.0,
                soma__cooldown = 2.0,
            ),
        )
        integrator = spark.ModuleSpecs(
            name = 'integrator',
            module_cls = spark.nn.interfaces.ExponentialIntegrator,
            inputs = {'spikes': [spark.PortMap(origin='neurons', port='out_spikes')]},
            outputs = {'action': 'signal'},
            config = spark.nn.interfaces.ExponentialIntegratorConfig(num_outputs=2),
        )
        return spark.nn.BrainConfig(modules_specs=[spiker, neurons, integrator])

    def test_it_builds_and_runs(self) -> None:
        brain = spark.nn.Brain(config=self._config())
        outputs = brain(drive=spark.FloatArray(jnp.zeros((4,), dtype=jnp.float16)))
        assert outputs['action'].value.shape == (2,)

    def test_what_was_asked_for_reached_the_modules(self) -> None:
        config = self._config()
        neurons = {m.name: m.config for m in config.modules_specs}['neurons']
        modules = {m.name: m.config for m in neurons.modules_specs}
        assert modules['synapses'].kernel.scale == 3.0
        assert modules['soma'].threshold_delta == 250.0
        assert modules['soma'].cooldown == 2.0

    def test_it_survives_several_jitted_steps(self) -> None:
        brain = spark.nn.Brain(config=self._config())
        drive = spark.FloatArray(jnp.array(np.full((4,), 0.5), dtype=jnp.float16))
        brain(drive=drive)
        graph, state = spark.split((brain))
        for _ in range(3):
            outputs, state = _run_split(graph, state, {'drive': drive})
        action = np.asarray(outputs['action'].value)
        assert action.shape == (2,)
        assert np.isfinite(action).all()

    def test_it_comes_back_from_a_file(self, tmp_path) -> None:
        path = tmp_path / 'array_brain.scfg'
        self._config().to_file(path, verbose=False)
        brain = spark.nn.Brain(config=spark.nn.BrainConfig.from_file(path))
        outputs = brain(drive=spark.FloatArray(jnp.zeros((4,), dtype=jnp.float16)))
        assert outputs['action'].value.shape == (2,)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestBrainRoundTrip:
    """
        A brain written to a file and built again from it.
    """

    def test_it_comes_back_and_runs(self, tmp_path) -> None:
        path = tmp_path / 'brain.scfg'
        _brain().to_file(path, verbose=False)
        config = spark.nn.BrainConfig.from_file(path)
        brain = spark.nn.Brain(config=config)
        outputs = brain(signal=spark.FloatArray(jnp.zeros((16,), dtype=jnp.float16)))
        assert outputs['my_awesome_signal'].value.shape == (3,)
        assert (brain.first_pool.units, brain.second_pool.units) == ((16,), (8,))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
