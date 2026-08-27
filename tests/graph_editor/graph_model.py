#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import pytest
import dataclasses as dc
import spark
from spark.core.registry import REGISTRY

pytest.importorskip('PySide6', reason='the graph editor needs PySide6')

import typing as tp
if tp.TYPE_CHECKING:
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication
    from spark.graph_editor.editor import GraphEditorWindow

from spark.graph_editor.models.graph_model import GraphModel
from spark.graph_editor.models.controller_profile import (
    NEURON_PROFILE, BRAIN_PROFILE, get_controller_profile, profile_for_config,
)
from spark.graph_editor.models import graph_export
from spark.graph_editor.models.model_import import ImportedGraph, expand_controller_config
from spark.core.registry import RegistryNamespace

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@pytest.fixture
def neuron_graph(qapp: QCoreApplication | QApplication) -> GraphModel:
    """
        An empty graph that builds a Neuron.
    """
    model = GraphModel()
    model.set_profile(NEURON_PROFILE, force=True)
    return model

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _import_into(graph, config) -> ImportedGraph:
    """
        Expands a configuration into a graph, the way the canvas does when a model is imported.
    """
    imported = expand_controller_config(config, graph, graph.profile)
    for node in imported.nodes:
        graph.add_node(node)
    for edge in imported.edges:
        graph.add_edge(edge)
    return imported

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _lif_config(**kwargs):
    """
        The configuration of the registered LIF neuron.
    """
    config_cls = REGISTRY.Neurons.get('lif_neuron').get_cls().get_config_spec()
    return config_cls(**kwargs) if kwargs else config_cls.partial()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def imported_graph(neuron_graph):
    """
        A graph holding an imported LIF neuron, expanded into its components.
    """
    _import_into(neuron_graph, _lif_config(units=(8,)))
    return neuron_graph

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestControllerProfiles:
    """
        What the editor knows about the controller a graph builds.
    """

    def test_a_profile_is_found_by_key(self) -> None:
        assert get_controller_profile('neuron') is NEURON_PROFILE
        assert get_controller_profile('brain') is BRAIN_PROFILE
        assert get_controller_profile(None) is None

    def test_a_profile_knows_its_controller(self) -> None:
        assert NEURON_PROFILE.controller_cls is spark.nn.Neuron
        assert NEURON_PROFILE.config_cls is spark.nn.NeuronConfig

    def test_a_configuration_names_its_profile(self) -> None:
        assert profile_for_config(spark.nn.BrainConfig(modules_specs=())) is BRAIN_PROFILE
        assert profile_for_config(spark.nn.neurons.ALIFNeuronConfig(units=(4,))) is NEURON_PROFILE

    def test_the_two_controllers_hold_different_things(self) -> None:
        assert NEURON_PROFILE.is_atomic(RegistryNamespace.Components)
        assert BRAIN_PROFILE.is_atomic(RegistryNamespace.Neurons)
        assert NEURON_PROFILE.is_importable(RegistryNamespace.Neurons)
        assert not BRAIN_PROFILE.is_importable(RegistryNamespace.Neurons)

    def test_a_brain_reads_everything_from_the_previous_step(self) -> None:
        assert BRAIN_PROFILE.allows_cycle()
        assert not NEURON_PROFILE.allows_cycle()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestControllerSettings:
    """
        The settings of the controller itself, which live on the graph rather than on a node.
    """

    def test_they_start_from_the_defaults_of_the_controller(self, neuron_graph) -> None:
        config = neuron_graph.controller_config
        assert isinstance(config, spark.nn.NeuronConfig)
        assert config.units is None
        assert config.modules_specs == ()

    def test_a_setting_is_written_and_read_back(self, neuron_graph) -> None:
        neuron_graph.set_node_config_value([GraphModel.CONTROLLER_ID, 'units'], (12,), force=True)
        assert neuron_graph.controller_config.units == (12,)
        assert neuron_graph.get_node_config_value([GraphModel.CONTROLLER_ID, 'units']) == (12,)

    def test_two_graphs_do_not_share_them(self, qapp) -> None:
        first, second = GraphModel(), GraphModel()
        first.set_profile(NEURON_PROFILE, force=True)
        second.set_profile(NEURON_PROFILE, force=True)
        first.set_node_config_value([GraphModel.CONTROLLER_ID, 'units'], (4,), force=True)
        assert first.controller_config is not second.controller_config
        assert second.controller_config.units is None

    def test_adopting_settings_leaves_the_modules_to_the_canvas(self, neuron_graph) -> None:
        adopted = spark.nn.neurons.ALIFNeuronConfig(units=(24,), inhibitory_rate=0.4)
        assert neuron_graph.adopt_controller_config(adopted)
        assert neuron_graph.controller_config.units == (24,)
        assert neuron_graph.controller_config.inhibitory_rate == 0.4
        assert neuron_graph.controller_config.modules_specs == ()

    def test_changing_the_controller_starts_the_settings_over(self, neuron_graph) -> None:
        neuron_graph.set_node_config_value([GraphModel.CONTROLLER_ID, 'units'], (12,), force=True)
        neuron_graph.set_profile(BRAIN_PROFILE, force=True)
        assert isinstance(neuron_graph.controller_config, spark.nn.BrainConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestImport:
    """
        A registered model opened into the canvas (Neuron mode).
    """

    def test_every_module_becomes_a_node(self, imported_graph) -> None:
        names = {node.name for node in imported_graph.nodes}
        assert {'delays', 'synapses', 'soma', 'hebbian_rule'} <= names

    def test_the_ports_of_the_controller_become_nodes_too(self, imported_graph) -> None:
        names = {node.name for node in imported_graph.nodes}
        assert 'in_spikes' in names
        assert 'out_spikes' in names

    def test_a_wired_self_property_becomes_a_node(self, neuron_graph) -> None:
        # The shipped models let the neuron supply the inhibition mask, so nothing wires
        # __self__ any more. A configuration that does wire one still gets its node.
        config = _lif_config(units=(8,))
        soma = next(spec for spec in config.modules_specs if spec.name == 'soma')
        soma.inputs['inhibition_mask'] = [
            spark.PortMap(origin='__self__', port='inhibition_mask', is_property=True)
        ]
        _import_into(neuron_graph, config)
        names = {node.name for node in neuron_graph.nodes}
        assert 'inhibition_mask' in names
        wiring = {(e.source_port.node.name, e.target_port.node.name) for e in neuron_graph.edges}
        assert ('inhibition_mask', 'soma') in wiring

    def test_the_wiring_is_carried_over(self, imported_graph) -> None:
        wiring = {(edge.source_port.node.name, edge.target_port.node.name) for edge in imported_graph.edges}
        assert ('delays', 'synapses') in wiring
        assert ('synapses', 'soma') in wiring

    def test_the_configuration_of_each_module_comes_along(self, imported_graph) -> None:
        synapses = next(node for node in imported_graph.nodes if node.name == 'synapses')
        assert synapses.config.units == (8,)

    def test_importing_twice_keeps_both_and_renames(self, imported_graph) -> None:
        _import_into(imported_graph, _lif_config(units=(8,)))
        names = [node.name for node in imported_graph.nodes]
        assert len(names) == len(set(names))
        assert sum(name.startswith('soma') for name in names) == 2

    def test_the_nodes_do_not_land_on_top_of_each_other(self, imported_graph) -> None:
        positions = [tuple(node.pos) for node in imported_graph.nodes]
        assert len(positions) == len(set(positions))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestExport:
    """
        A graph translated back into the configuration of its controller.
    """

    def test_an_empty_graph_describes_nothing_to_build(self, neuron_graph) -> None:
        exported = graph_export.build_controller_config(neuron_graph, strict=True)
        assert exported.config is not None
        assert any('does not contain any module' in problem for problem in exported.problems)

    def test_an_imported_model_is_described(self, imported_graph) -> None:
        exported = graph_export.build_controller_config(imported_graph, strict=False)
        assert {spec.name for spec in exported.specs} >= {'delays', 'synapses', 'soma'}

    def test_a_shape_that_is_not_set_is_reported(self, neuron_graph) -> None:
        _import_into(neuron_graph, _lif_config())
        problems = graph_export.build_controller_config(neuron_graph, strict=True).problems
        assert any('units' in problem for problem in problems)

    def test_a_complete_graph_has_nothing_to_report(self, imported_graph) -> None:
        imported_graph.set_node_config_value([GraphModel.CONTROLLER_ID, 'units'], (8,), force=True)
        exported = graph_export.build_controller_config(imported_graph, strict=True)
        assert exported.problems == []
        assert exported.is_complete

    def test_what_was_exported_instantiates(self, imported_graph) -> None:
        imported_graph.set_node_config_value([GraphModel.CONTROLLER_ID, 'units'], (8,), force=True)
        config = graph_export.build_controller_config(imported_graph, strict=True).config
        neuron = spark.nn.neurons.LIFNeuron(config=config)
        import jax.numpy as jnp
        outputs = neuron(in_spikes=spark.SpikeArray(jnp.zeros((8,), dtype=jnp.uint8)))
        assert outputs['out_spikes'].value.shape == (8,)

    def test_the_layout_travels_with_the_graph(self, imported_graph) -> None:
        exported = graph_export.build_controller_config(imported_graph, strict=False)
        assert set(exported.layout) >= {'delays', 'synapses', 'soma'}
        assert all(len(position) == 2 for position in exported.layout.values())

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
