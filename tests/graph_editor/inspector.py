#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import pytest
from spark.core.registry import REGISTRY

pytest.importorskip('PySide6', reason='the graph editor needs PySide6')

import typing as tp
if tp.TYPE_CHECKING:
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

from shiboken6 import isValid
from PySide6.QtCore import QEvent
from PySide6.QtTest import QTest
from spark.graph_editor.models.graph_model import GraphModel
from spark.graph_editor.models.controller_profile import BRAIN_PROFILE
from spark.graph_editor.models.node_factory import NODE_REGISTRY
from spark.graph_editor.widgets.inspector_view import InspectorView
from spark.graph_editor.widgets.attribute_view import QAttribute
from spark.graph_editor.view.graph_view import GraphScene, GraphView
from spark.graph_editor.view.node_item import NodeItem

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _settle(qapp) -> None:
    """
        Answers the queued events, the widgets deleted later included.
    """
    qapp.processEvents()
    # NOTE: processEvents() outside of an event loop leaves the deferred deletions queued.
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def pool_inspector(qapp: QCoreApplication | QApplication) -> tp.Generator[tuple[InspectorView, tp.Any, GraphModel], tp.Any, None]:
    """
        An inspector showing a neuron pool placed in a brain.
    """
    graph = GraphModel()
    graph.set_profile(BRAIN_PROFILE, force=True)
    node = NODE_REGISTRY.get(REGISTRY.Neurons.get('alif_neuron').get_cls())(name='pool')
    graph.add_node(node)
    inspector = InspectorView()
    inspector.show()
    inspector.set_node(node, graph)
    _settle(qapp)
    yield inspector, node, graph
    inspector.close()
    inspector.deleteLater()
    _settle(qapp)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestShapeEditor:
    """
        The editor of a shape, such as the units of a pool.
    """

    def test_typing_keeps_the_field_being_edited(self, pool_inspector, qapp) -> None:
        inspector, node, graph = pool_inspector
        attribute = next(w for w in inspector.findChildren(QAttribute) if w.config_path == [node.id, 'units'])
        spin = attribute._input_widget._spins[0]
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, '3')
        _settle(qapp)
        assert node.config.units == (3,)
        assert isValid(spin)
        assert attribute._input_widget._spins[0] is spin
        QTest.keyClicks(spin, '2')
        _settle(qapp)
        assert isValid(spin)
        assert node.config.units == (32,)
        # Consecutive keystrokes are one edit.
        assert graph.undo_stack.count() == 1

    def test_an_undo_reaches_the_field(self, pool_inspector, qapp) -> None:
        inspector, node, graph = pool_inspector
        attribute = next(w for w in inspector.findChildren(QAttribute) if w.config_path == [node.id, 'units'])
        editor = attribute._input_widget
        before = editor.value()
        editor.add_dim()
        _settle(qapp)
        assert len(node.config.units) == len(before) + 1
        graph.undo_stack.undo()
        _settle(qapp)
        assert editor.value() == before
        assert len(editor._spins) == len(before)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestOutputsOfASampler:
    """
        The number of outputs of a Sampler, typed in the inspector.
    """

    @pytest.fixture
    def setup(self, qapp: QCoreApplication | QApplication) -> tp.Generator[tuple, tp.Any, None]:
        graph = GraphModel()
        graph.set_profile(BRAIN_PROFILE, force=True)
        scene = GraphScene(graph)
        view = GraphView(scene)
        node = lambda registered, name: NODE_REGISTRY.get(REGISTRY.Interfaces.get(registered).get_cls())(name=name)
        spiker, sampler, readout = node('poisson_spiker', 'spiker'), node('sampler', 'sampler'), node('exponential_integrator', 'readout')
        for each in (spiker, sampler, readout):
            graph.add_node(each)
        graph.connect(spiker.get_port_by_name('spikes', False), sampler.get_port_by_name('inputs', True))
        inspector = InspectorView()
        inspector.show()
        inspector.set_node(sampler, graph)
        _settle(qapp)
        attribute = next(w for w in inspector.findChildren(QAttribute) if w.config_path == [sampler.id, 'num_outputs'])
        yield attribute._input_widget, scene, graph, sampler, readout
        inspector.close()
        inspector.deleteLater()
        view.deleteLater()
        _settle(qapp)

    @staticmethod
    def _outputs(node) -> list[str]:
        return [port.name for port in node.call_section.ports if not port.is_input]

    def test_typing_a_count_lays_out_its_ports(self, setup, qapp) -> None:
        spin, scene, graph, sampler, _ = setup
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, '3')
        _settle(qapp)
        assert self._outputs(sampler) == ['output_0', 'output_1', 'output_2']
        item = next(each for each in scene.items() if isinstance(each, NodeItem) and each.model is sampler)
        assert [port.model.name for port in item._port_items()] == ['inputs', 'output_0', 'output_1', 'output_2']
        # The pipe of the input moved to the new item of its port, and the field being edited stayed.
        inputs = next(port for port in item._port_items() if port.model.name == 'inputs')
        assert [pipe.target_port for pipe in inputs.connected_pipes] == [inputs]
        assert isValid(spin)

    def test_an_undo_brings_back_the_connections_of_the_ports_removed(self, setup, qapp) -> None:
        spin, scene, graph, sampler, readout = setup
        spin.setValue(3)
        _settle(qapp)
        graph.connect(sampler.get_port_by_name('output_2', False), readout.get_port_by_name('spikes', True))
        spin.setValue(1)
        _settle(qapp)
        assert self._outputs(sampler) == ['output_0']
        assert [edge.target_port.node.name for edge in graph.edges] == ['sampler']
        graph.undo_stack.undo()
        _settle(qapp)
        assert self._outputs(sampler) == ['output_0', 'output_1', 'output_2']
        assert {edge.target_port.node.name for edge in graph.edges} == {'sampler', 'readout'}
        assert spin.value() == 3

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
