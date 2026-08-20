#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import pytest
import spark

pytest.importorskip('PySide6', reason='the graph editor needs PySide6')

from spark.graph_editor.models import model_library

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@pytest.fixture
def library(qapp, tmp_path):
    """
        A library of its own, put back where it was when the test ends.
    """
    previous = model_library._settings().value(model_library.SETTINGS_KEY, '')
    root = tmp_path / 'library'
    root.mkdir()
    model_library.set_library_path(root)
    yield root
    model_library.set_library_path(previous or None)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _model_file(path, units=(8,)):
    """
        A model saved where it is asked for.
    """
    spark.nn.neurons.ALIFNeuronConfig(units=units).to_file(path, verbose=False)
    return path

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestWhereTheLibrarySits:

    def test_the_chosen_location_is_remembered(self, library) -> None:
        assert model_library.library_path() == library

    def test_choosing_nothing_returns_to_the_default(self, library) -> None:
        model_library.set_library_path(None)
        assert model_library.library_path() == model_library.default_path()

    def test_a_location_that_is_not_there_holds_no_models(self, library, tmp_path) -> None:
        assert model_library.model_files(tmp_path / 'nowhere') == []

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestTakingAModelIn:

    def test_a_model_is_copied_and_becomes_available(self, library, tmp_path) -> None:
        source = _model_file(tmp_path / 'library_probe_neuron.scfg')
        destination = model_library.import_model(source)
        assert destination == library / 'library_probe_neuron.scfg'
        assert destination.is_file()
        assert spark.REGISTRY.Neurons.get('library_probe_neuron') is not None

    def test_the_file_keeps_its_name_as_the_name_of_the_model(self, library, tmp_path) -> None:
        source = _model_file(tmp_path / 'library_named_neuron.scfg')
        assert model_library.model_name(model_library.import_model(source)) == 'library_named_neuron'

    def test_a_second_copy_is_refused_unless_it_is_meant(self, library, tmp_path) -> None:
        source = _model_file(tmp_path / 'library_twice_neuron.scfg')
        model_library.import_model(source)
        with pytest.raises(FileExistsError):
            model_library.import_model(source)
        assert model_library.import_model(source, overwrite=True).is_file()

    def test_what_is_not_a_model_is_refused(self, library, tmp_path) -> None:
        brain_path = tmp_path / 'library_brain.scfg'
        spark.nn.BrainConfig(modules_specs=()).to_file(brain_path, verbose=False)
        with pytest.raises(TypeError):
            model_library.import_model(brain_path)
        assert model_library.model_files() == []

    def test_a_file_that_is_not_there_is_refused(self, library, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            model_library.import_model(tmp_path / 'nothing.scfg')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestReadingTheLibrary:

    def test_every_model_is_made_available(self, library, tmp_path) -> None:
        _model_file(library / 'library_first_neuron.scfg')
        _model_file(library / 'library_second_neuron.scfg')
        registered, failed = model_library.register_library()
        assert sorted(registered) == ['library_first_neuron', 'library_second_neuron']
        assert failed == []

    def test_one_broken_file_does_not_take_the_rest_with_it(self, library) -> None:
        _model_file(library / 'library_sound_neuron.scfg')
        (library / 'library_broken.scfg').write_text('not a model')
        registered, failed = model_library.register_library()
        assert registered == ['library_sound_neuron']
        assert [path.name for path, _ in failed] == ['library_broken.scfg']

    def test_a_name_already_taken_is_left_alone(self, library) -> None:
        _model_file(library / 'library_kept_neuron.scfg')
        assert model_library.register_library()[0] == ['library_kept_neuron']
        registered_cls = spark.REGISTRY.Neurons.get('library_kept_neuron').get_cls()
        assert model_library.register_library()[0] == []
        assert spark.REGISTRY.Neurons.get('library_kept_neuron').get_cls() is registered_cls

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestTheEditorReachingTheLibrary:

    def test_the_action_is_there_and_needs_no_document(self, editor) -> None:
        actions = {action.text(): action for action in editor._file_menu.actions()}
        assert 'Add Model to Library...' in actions
        assert actions['Add Model to Library...'] not in editor._document_actions

    def test_the_action_takes_a_model_in(self, editor, library, tmp_path, monkeypatch) -> None:
        from PySide6.QtWidgets import QFileDialog
        source = _model_file(tmp_path / 'library_action_neuron.scfg')
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(source), '')))
        editor.add_model_to_library()
        assert (library / 'library_action_neuron.scfg').is_file()
        assert spark.REGISTRY.Neurons.get('library_action_neuron') is not None

    def test_the_action_asks_before_replacing(self, editor, library, tmp_path, monkeypatch) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        source = _model_file(tmp_path / 'library_replace_neuron.scfg')
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(source), '')))
        editor.add_model_to_library()
        destination = library / 'library_replace_neuron.scfg'
        destination.write_text('touched')
        monkeypatch.setattr(QMessageBox, 'question',
                            staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
        editor.add_model_to_library()
        assert destination.read_text() == 'touched'
        monkeypatch.setattr(QMessageBox, 'question',
                            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
        editor.add_model_to_library()
        assert destination.read_bytes() == source.read_bytes()

    def test_the_preferences_hold_the_location(self, editor, library) -> None:
        from spark.graph_editor.widgets.preferences_dialog import PreferencesDialog
        dialog = PreferencesDialog(editor)
        assert 'Model Library' in [dialog.sidebar.item(row).text() for row in range(dialog.sidebar.count())]
        assert dialog.library_path_edit.text() == str(library)
        dialog.library_path_edit.setText(str(library.parent / 'elsewhere'))
        dialog._commit_library()
        assert model_library.library_path() == library.parent / 'elsewhere'
        dialog.deleteLater()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestReachingAModelTheEditorLearnsLate:
    """
        A model arrives while the editor is already up, either with the file being opened or from the
        library, and has to be as placeable as one that was there from the start.
    """

    def test_a_model_taken_in_becomes_placeable(self, library, tmp_path) -> None:
        from spark.graph_editor.models.node_factory import NODE_REGISTRY
        model_library.import_model(_model_file(tmp_path / 'library_late_neuron.scfg'))
        neuron_cls = spark.REGISTRY.Neurons.get('library_late_neuron').get_cls()
        assert NODE_REGISTRY.get(neuron_cls) is not None
        assert NODE_REGISTRY.get_namespace(neuron_cls) is not None

    def test_a_brain_bringing_its_own_model_opens_whole(self, editor, tmp_path) -> None:
        neuron_path = _model_file(tmp_path / 'editor_probe_neuron.scfg')
        spark.register_neuron_from_config_file('editor_probe_neuron', neuron_path)
        brain = spark.nn.BrainConfig(modules_specs=[
            spark.ModuleSpecs(
                name='spiker',
                module_cls=spark.nn.interfaces.PoissonSpiker,
                inputs={'signal': [spark.PortMap(origin='__call__', port='signal')]},
            ),
            spark.ModuleSpecs(
                name='pool',
                module_cls=spark.REGISTRY.Neurons.get('editor_probe_neuron').get_cls(),
                inputs={'in_spikes': [spark.PortMap(origin='spiker', port='spikes')]},
            ),
        ])
        brain_path = tmp_path / 'editor_probe_brain.scfg'
        brain.to_file(brain_path, compress=False, verbose=False)
        # The same brain, built on a model nothing has ever heard of.
        unheard_path = tmp_path / 'editor_unheard_brain.scfg'
        unheard_path.write_text(brain_path.read_text().replace('editor_probe_neuron', 'editor_unheard_neuron'))
        assert spark.REGISTRY.Neurons.get('editor_unheard_neuron') is None
        assert editor.open_model_file(unheard_path)
        placed = {node.name for node in editor._scene.model.nodes}
        assert {'spiker', 'pool'} <= placed

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
