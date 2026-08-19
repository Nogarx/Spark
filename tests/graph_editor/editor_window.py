#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import pytest
import spark
from spark.core.registry import REGISTRY

pytest.importorskip('PySide6', reason='the graph editor needs PySide6')

import typing as tp
if tp.TYPE_CHECKING:
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication
    from spark.graph_editor.editor import GraphEditorWindow

from spark.graph_editor.editor import GraphEditorWindow
from spark.graph_editor.models import session_io, recent_files
from spark.graph_editor.models.graph_model import GraphModel
from spark.graph_editor.models.controller_profile import NEURON_PROFILE, BRAIN_PROFILE

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _pump(qapp, times: int = 4) -> None:
    """
        Lets the window answer everything it has queued.
    """
    for _ in range(times):
        qapp.processEvents()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _fill_shapes(model, units=(8,)) -> None:
    """
        Sets every shape a user would set, on the controller and on the modules that ask for one.
    """
    import dataclasses as dc
    model.set_node_config_value([GraphModel.CONTROLLER_ID, 'units'], units, force=True)
    for node in model.nodes:
        if node.config is not None and any(f.name == 'units' for f in dc.fields(node.config)):
            model.set_node_config_value([node.id, 'units'], units, force=True)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def imported(editor: GraphEditorWindow, qapp: QCoreApplication | QApplication) -> GraphEditorWindow:
    """
        An editor holding one imported LIF neuron, ready to export.
    """
    editor.new_graph(NEURON_PROFILE)
    editor.view.import_model(REGISTRY.Neurons.get('lif_neuron'))
    _pump(qapp)
    _fill_shapes(editor._scene.model)
    _pump(qapp)
    return editor

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestDocuments:
    """
        The editor holds one document per model being edited.
    """

    def test_it_opens_on_the_start_screen(self, editor) -> None:
        assert editor._documents == []
        assert editor._stack.currentWidget() is editor._start_view

    def test_a_new_session_opens_a_tab(self, editor, qapp) -> None:
        editor.new_graph(NEURON_PROFILE)
        _pump(qapp)
        assert len(editor._documents) == 1
        assert editor._scene.model.profile is NEURON_PROFILE
        assert editor._stack.currentWidget() is editor._tabs

    def test_a_second_session_opens_beside_the_first(self, editor, qapp) -> None:
        editor.new_graph(NEURON_PROFILE)
        editor.new_graph(BRAIN_PROFILE)
        _pump(qapp)
        assert len(editor._documents) == 2
        assert editor._scene.model.profile is BRAIN_PROFILE

    def test_each_tab_keeps_its_own_graph(self, editor, qapp) -> None:
        editor.new_graph(NEURON_PROFILE)
        editor.view.import_model(REGISTRY.Neurons.get('lif_neuron'))
        editor.new_graph(NEURON_PROFILE)
        _pump(qapp)
        assert editor._scene.model.nodes == []
        editor._tabs.setCurrentIndex(0)
        _pump(qapp)
        assert editor._scene.model.nodes != []

    def test_the_settings_of_one_tab_stay_there(self, editor, qapp) -> None:
        editor.new_graph(NEURON_PROFILE)
        editor.new_graph(NEURON_PROFILE)
        _pump(qapp)
        editor._scene.model.set_node_config_value([GraphModel.CONTROLLER_ID, 'units'], (12,), force=True)
        first, second = (document.model.controller_config for document in editor._documents)
        assert (first.units, second.units) == (None, (12,))

    def test_duplicates_are_told_apart(self, editor, qapp) -> None:
        editor.new_graph(NEURON_PROFILE)
        editor.new_graph(NEURON_PROFILE)
        _pump(qapp)
        labels = [editor._tabs.tabText(index) for index in range(editor._tabs.count())]
        assert labels == ['Untitled', 'Untitled (2)']

    def test_closing_the_last_one_goes_back_to_the_start_screen(self, editor, qapp) -> None:
        editor.new_graph(NEURON_PROFILE)
        _pump(qapp)
        assert editor.close_document(0)
        _pump(qapp)
        assert editor._documents == []
        assert editor._stack.currentWidget() is editor._start_view

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestImporting:
    """
        A registered model opened into the canvas.
    """

    def test_a_model_becomes_nodes(self, editor, qapp) -> None:
        editor.new_graph(NEURON_PROFILE)
        editor.view.import_model(REGISTRY.Neurons.get('lif_neuron'))
        _pump(qapp)
        names = {node.name for node in editor._scene.model.nodes}
        assert {'delays', 'synapses', 'soma'} <= names

    def test_importing_can_be_undone(self, editor, qapp) -> None:
        editor.new_graph(NEURON_PROFILE)
        editor.view.import_model(REGISTRY.Neurons.get('lif_neuron'))
        _pump(qapp)
        editor._scene.model.undo_stack.undo()
        _pump(qapp)
        assert editor._scene.model.nodes == []

    def test_importing_twice_adds_to_the_same_session(self, editor, qapp) -> None:
        editor.new_graph(NEURON_PROFILE)
        editor.view.import_model(REGISTRY.Neurons.get('lif_neuron'))
        _pump(qapp)
        before = len(editor._scene.model.nodes)
        editor.view.import_model(REGISTRY.Neurons.get('lif_neuron'))
        _pump(qapp)
        assert len(editor._documents) == 1
        assert len(editor._scene.model.nodes) > before

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestFiles:
    """
        Saving, exporting and opening from the window.
    """

    def test_a_session_is_saved_and_named(self, imported, qapp, answers, tmp_path) -> None:
        answers.picks(tmp_path / 'work.sge')
        assert imported.save_session_as() is True
        assert (tmp_path / 'work.sge').exists()
        assert imported._tabs.tabText(0) == 'work'

    def test_a_saved_session_is_not_modified(self, imported, qapp, answers, tmp_path) -> None:
        answers.picks(tmp_path / 'work.sge')
        imported.save_session_as()
        _pump(qapp)
        assert not imported.document.is_modified

    def test_a_model_is_exported(self, imported, qapp, answers, tmp_path) -> None:
        answers.picks(tmp_path / 'model.scfg')
        assert imported.export_model_as() is True
        assert (tmp_path / 'model.scfg').exists()

    def test_an_incomplete_model_is_refused_and_reported(self, editor, qapp, answers, tmp_path) -> None:
        editor.new_graph(NEURON_PROFILE)
        editor.view.import_model(REGISTRY.Neurons.get('lif_neuron'))
        _pump(qapp)
        answers.picks(tmp_path / 'model.scfg')
        assert editor.export_model_as() is False
        assert not (tmp_path / 'model.scfg').exists()
        assert answers.reported and 'units' in answers.reported[-1]

    def test_an_incomplete_session_still_saves(self, editor, qapp, answers, tmp_path) -> None:
        editor.new_graph(NEURON_PROFILE)
        editor.view.import_model(REGISTRY.Neurons.get('lif_neuron'))
        _pump(qapp)
        answers.picks(tmp_path / 'work.sge')
        assert editor.save_session_as() is True
        assert (tmp_path / 'work.sge').exists()

    def test_a_session_is_opened_again(self, imported, qapp, answers, tmp_path) -> None:
        answers.picks(tmp_path / 'work.sge')
        imported.save_session_as()
        _pump(qapp)
        names_before = sorted(node.name for node in imported._scene.model.nodes)
        imported.load_session()
        _pump(qapp)
        assert len(imported._documents) == 2
        assert sorted(node.name for node in imported._scene.model.nodes) == names_before

    def test_an_empty_session_is_taken_over_rather_than_left_behind(self, editor, qapp, answers, tmp_path) -> None:
        editor.new_graph(NEURON_PROFILE)
        editor.view.import_model(REGISTRY.Neurons.get('lif_neuron'))
        _pump(qapp)
        answers.picks(tmp_path / 'work.sge')
        editor.save_session_as()
        editor.new_graph(BRAIN_PROFILE)
        _pump(qapp)
        assert len(editor._documents) == 2
        editor.load_session()
        _pump(qapp)
        assert len(editor._documents) == 2
        assert editor._scene.model.profile is NEURON_PROFILE

    def test_a_model_opens_as_a_session_of_its_own(self, imported, qapp, answers, tmp_path) -> None:
        answers.picks(tmp_path / 'model.scfg')
        imported.export_model_as()
        _pump(qapp)
        assert imported.open_model_file(tmp_path / 'model.scfg') is True
        _pump(qapp)
        assert imported.document.model_path == tmp_path / 'model.scfg'
        assert {node.name for node in imported._scene.model.nodes} >= {'delays', 'synapses', 'soma'}

    def test_a_file_that_is_gone_is_dropped_from_the_recent_list(self, editor, qapp, answers, tmp_path) -> None:
        missing = tmp_path / 'not_there.sge'
        recent_files.remember(missing)
        assert editor.open_path(missing) is False
        assert missing not in recent_files.recent_files(existing_only=False)

    def test_what_was_written_is_remembered(self, imported, qapp, answers, tmp_path) -> None:
        answers.picks(tmp_path / 'work.sge')
        imported.save_session_as()
        assert (tmp_path / 'work.sge').resolve() in [p.resolve() for p in recent_files.recent_files()]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestChecking:
    """
        The verdict on the model, without writing a file.
    """

    def test_a_complete_model_is_confirmed(self, imported, qapp, answers) -> None:
        assert imported.check_model() is True
        assert answers.reported == []

    def test_an_incomplete_model_is_reported(self, editor, qapp, answers) -> None:
        editor.new_graph(NEURON_PROFILE)
        editor.view.import_model(REGISTRY.Neurons.get('lif_neuron'))
        _pump(qapp)
        assert editor.check_model() is False
        assert answers.reported

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
