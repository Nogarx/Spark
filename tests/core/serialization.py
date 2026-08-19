#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import json
import lzma
import pytest
import numpy as np
import jax.numpy as jnp
import typing as tp
import spark
from spark.nn.interfaces.input.topological import TopologicalLinearSpikerConfig
from spark.nn.neurons.alif import ALIFNeuronConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _read(path) -> dict:
    """
        The document a configuration was written as.
    """
    return json.loads(lzma.open(path).read().decode())

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _find(payload, key: str) -> tp.Any | None:
    """
        The first value stored under a key, however deep it sits.
    """
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find(value, key)
            if found is not None:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _find(value, key)
            if found is not None:
                return found
    return None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def neuron_config() -> ALIFNeuronConfig:
    return spark.nn.neurons.ALIFNeuronConfig(units=(8,), synapses__kernel__scale=7.0)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def spiker_config() -> TopologicalLinearSpikerConfig:
    return TopologicalLinearSpikerConfig(glue=jnp.array(0), mins=jnp.array(-1), maxs=jnp.array(1))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestRoundTrip:
    """
        A configuration written to a file and read back.
    """

    def test_a_flat_configuration(self, tmp_path, spiker_config) -> None:
        path = tmp_path / 'spiker.scfg'
        spiker_config.to_file(path, verbose=False)
        back = TopologicalLinearSpikerConfig.from_file(path)
        assert back.resolution == spiker_config.resolution
        assert np.asarray(back.glue).tolist() == np.asarray(spiker_config.glue).tolist()

    def test_a_configuration_holding_modules(self, tmp_path, neuron_config) -> None:
        path = tmp_path / 'neuron.scfg'
        neuron_config.to_file(path, verbose=False)
        back = spark.nn.neurons.ALIFNeuronConfig.from_file(path)
        assert [m.name for m in back.modules_specs] == [m.name for m in neuron_config.modules_specs]
        synapses = {m.name: m.config for m in back.modules_specs}['synapses']
        assert (synapses.units, synapses.kernel.scale) == ((8,), 7.0)

    def test_what_came_back_still_builds(self, tmp_path, neuron_config) -> None:
        path = tmp_path / 'neuron.scfg'
        neuron_config.to_file(path, verbose=False)
        neuron = spark.nn.neurons.ALIFNeuron(config=spark.nn.neurons.ALIFNeuronConfig.from_file(path))
        outputs = neuron(in_spikes=spark.SpikeArray(jnp.zeros((8,), dtype=jnp.uint8)))
        assert outputs['out_spikes'].value.shape == (8,)

    def test_the_wiring_survives(self, tmp_path, neuron_config) -> None:
        path = tmp_path / 'neuron.scfg'
        neuron_config.to_file(path, verbose=False)
        back = spark.nn.neurons.ALIFNeuronConfig.from_file(path)
        original = {m.name: m.inputs for m in neuron_config.modules_specs}
        reloaded = {m.name: m.inputs for m in back.modules_specs}
        assert set(original) == set(reloaded)
        for name, inputs in original.items():
            assert set(inputs) == set(reloaded[name])
            for port, port_maps in inputs.items():
                assert [(p.origin, p.port, p.is_property) for p in port_maps] == \
                       [(p.origin, p.port, p.is_property) for p in reloaded[name][port]]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestOnDisk:
    """
        What the document itself holds.
    """

    def test_an_array_is_written_as_an_array(self, tmp_path, spiker_config) -> None:
        path = tmp_path / 'spiker.scfg'
        spiker_config.to_file(path, verbose=False)
        assert _find(_read(path), 'glue') == {'__type__': 'array', 'dtype': 'int32', 'shape': [], 'data': 0}

    def test_a_module_records_the_namespace_it_lives_in(self, tmp_path) -> None:
        config = spark.nn.BrainConfig(modules_specs=[
            spark.ModuleSpecs(name='spiker', module_cls=spark.nn.interfaces.PoissonSpiker,
                              inputs={'signal': [spark.PortMap(origin='__call__', port='signal')]}),
        ])
        path = tmp_path / 'brain.scfg'
        config.to_file(path, verbose=False)
        assert _find(_read(path), 'module_cls') == {
            '__module_type__': 'poisson_spiker', '__subregistry__': 'Interfaces',
        }

    def test_a_dtype_is_written_by_name(self, tmp_path, spiker_config) -> None:
        path = tmp_path / 'spiker.scfg'
        spiker_config.to_file(path, verbose=False)
        assert _find(_read(path), 'dtype') == {'__type__': 'dtype', 'name': 'float16'}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestFailedWrite:
    """
        What is left behind when a configuration cannot be written.
    """

    @staticmethod
    def _unwritable(config):
        class Unencodable:
            pass
        object.__setattr__(config, 'resolution', Unencodable())
        return config

    def test_the_file_that_was_there_survives(self, tmp_path, spiker_config) -> None:
        path = tmp_path / 'spiker.scfg'
        spiker_config.to_file(path, verbose=False)
        with pytest.raises(TypeError):
            self._unwritable(spiker_config).to_file(path, verbose=False)
        assert _find(_read(path), 'glue') is not None

    def test_nothing_is_written_where_there_was_nothing(self, tmp_path, spiker_config) -> None:
        path = tmp_path / 'never.scfg'
        with pytest.raises(TypeError):
            self._unwritable(spiker_config).to_file(path, verbose=False)
        assert not path.exists()

    def test_no_half_written_file_is_left_beside_it(self, tmp_path, spiker_config) -> None:
        path = tmp_path / 'spiker.scfg'
        with pytest.raises(TypeError):
            self._unwritable(spiker_config).to_file(path, verbose=False)
        assert list(tmp_path.glob('*.partial')) == []

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRegisteringFromAFile:
    """
        A neuron built out of a configuration, as the sharding tutorial loads one.
    """

    def test_a_pair_is_registered_and_reachable(self, tmp_path) -> None:
        path = tmp_path / 'probe_neuron.scfg'
        spark.nn.neurons.ALIFNeuronConfig(units=(8,)).to_file(path, verbose=False)
        spark.register_neuron_from_config_file('ProbeFileNeuron', path)
        entry = spark.REGISTRY.Neurons.get('ProbeFileNeuron')
        assert entry is not None
        neuron_cls = entry.get_cls()
        assert neuron_cls.__name__ == 'ProbeFileNeuron'
        assert spark.REGISTRY.Configs.get('ProbeFileNeuronConfig') is not None

    def test_the_registered_neuron_runs(self, tmp_path) -> None:
        path = tmp_path / 'probe_neuron_run.scfg'
        spark.nn.neurons.ALIFNeuronConfig(units=(8,)).to_file(path, verbose=False)
        spark.register_neuron_from_config_file('ProbeRunNeuron', path)
        neuron = spark.REGISTRY.Neurons.get('ProbeRunNeuron').get_cls()()
        outputs = neuron(in_spikes=spark.SpikeArray(jnp.zeros((8,), dtype=jnp.uint8)))
        assert outputs['out_spikes'].value.shape == (8,)

    def test_a_name_already_taken_is_refused(self, tmp_path) -> None:
        path = tmp_path / 'probe_neuron_twice.scfg'
        spark.nn.neurons.ALIFNeuronConfig(units=(8,)).to_file(path, verbose=False)
        spark.register_neuron_from_config_file('ProbeTwiceNeuron', path)
        with pytest.raises(Exception):
            spark.register_neuron_from_config_file('ProbeTwiceNeuron', path)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
