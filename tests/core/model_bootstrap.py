#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import json
import pytest
import spark
from spark.core.registry import register_models_from_payload

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _brain_with_a_model_of_its_own(tmp_path, name: str):
    """
        A brain built on a neuron that exists only as a configuration, written uncompressed so that the
        document it holds can be read as it is.
    """
    neuron_path = tmp_path / f'{name}.scfg'
    spark.nn.neurons.ALIFNeuronConfig(units=(8,)).to_file(neuron_path, verbose=False)
    spark.register_neuron_from_config_file(name, neuron_path)
    neuron_cls = spark.REGISTRY.Neurons.get(name).get_cls()
    brain = spark.nn.BrainConfig(modules_specs=[
        spark.ModuleSpecs(
            name='spiker',
            module_cls=spark.nn.interfaces.PoissonSpiker,
            inputs={'signal': [spark.PortMap(origin='__call__', port='signal')]},
        ),
        spark.ModuleSpecs(
            name='pool',
            module_cls=neuron_cls,
            inputs={'in_spikes': [spark.PortMap(origin='spiker', port='spikes')]},
        ),
    ])
    brain_path = tmp_path / f'{name}_brain.scfg'
    brain.to_file(brain_path, compress=False, verbose=False)
    return brain_path

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _under_another_name(brain_path, name: str, other_name: str):
    """
        The same document, naming a model nothing has ever heard of.
    """
    renamed = brain_path.read_text().replace(name, other_name)
    other_path = brain_path.with_name(f'{other_name}_brain.scfg')
    other_path.write_text(renamed)
    return other_path

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestAFileBringingItsOwnModels:
    """
        A saved model carries the models it is built on, so reading one needs nothing to have been
        imported beforehand.
    """

    def test_a_model_the_registry_lacks_is_built_from_the_file(self, tmp_path) -> None:
        brain_path = _brain_with_a_model_of_its_own(tmp_path, 'bootstrap_probe_neuron')
        other_path = _under_another_name(brain_path, 'bootstrap_probe_neuron', 'bootstrap_absent_neuron')
        assert spark.REGISTRY.Neurons.get('bootstrap_absent_neuron') is None
        registered = register_models_from_payload(json.loads(other_path.read_text()))
        assert registered == ['bootstrap_absent_neuron']
        assert spark.REGISTRY.Neurons.get('bootstrap_absent_neuron') is not None

    def test_a_model_already_there_is_left_alone(self, tmp_path) -> None:
        brain_path = _brain_with_a_model_of_its_own(tmp_path, 'bootstrap_kept_neuron')
        payload = json.loads(brain_path.read_text())
        registered_cls = spark.REGISTRY.Neurons.get('bootstrap_kept_neuron').get_cls()
        assert register_models_from_payload(payload) == []
        assert spark.REGISTRY.Neurons.get('bootstrap_kept_neuron').get_cls() is registered_cls

    def test_reading_the_file_is_enough(self, tmp_path) -> None:
        brain_path = _brain_with_a_model_of_its_own(tmp_path, 'bootstrap_read_neuron')
        other_path = _under_another_name(brain_path, 'bootstrap_read_neuron', 'bootstrap_unheard_neuron')
        assert spark.REGISTRY.Neurons.get('bootstrap_unheard_neuron') is None
        config = spark.nn.BrainConfig.from_file(other_path)
        assert [spec.name for spec in config.modules_specs] == ['spiker', 'pool']
        assert spark.REGISTRY.Neurons.get('bootstrap_unheard_neuron') is not None

    def test_the_model_built_from_the_file_is_the_same_one(self, tmp_path) -> None:
        brain_path = _brain_with_a_model_of_its_own(tmp_path, 'bootstrap_same_neuron')
        other_path = _under_another_name(brain_path, 'bootstrap_same_neuron', 'bootstrap_twin_neuron')
        from_registry = spark.nn.BrainConfig.from_file(brain_path)
        from_file = spark.nn.BrainConfig.from_file(other_path)
        pool_registry = [spec for spec in from_registry.modules_specs if spec.name == 'pool'][0]
        pool_file = [spec for spec in from_file.modules_specs if spec.name == 'pool'][0]
        assert pool_file.config.units == pool_registry.config.units
        assert [m.name for m in pool_file.config.modules_specs] == [m.name for m in pool_registry.config.modules_specs]

    def test_a_document_naming_nothing_is_answered_with_nothing(self) -> None:
        assert register_models_from_payload({'__cfg__': {'units': [8]}}) == []
        assert register_models_from_payload([1, 'two', None]) == []

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
