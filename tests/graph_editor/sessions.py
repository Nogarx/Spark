#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import jax.numpy as jnp
import spark
from spark.core.registry import REGISTRY

pytest.importorskip('PySide6', reason='the graph editor needs PySide6')

from spark.graph_editor.models import session_io
from spark.graph_editor.models.graph_model import GraphModel
from spark.graph_editor.models.model_import import expand_controller_config
from spark.graph_editor.models.controller_profile import NEURON_PROFILE, BRAIN_PROFILE

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _graph(profile=NEURON_PROFILE, *, units=None, complete=False):
    """
        A graph, optionally holding a LIF neuron with every shape filled in.
    """
    model = GraphModel()
    model.set_profile(profile, force=True)
    if units is not None:
        config_cls = REGISTRY.Neurons.get('lif_neuron').get_cls().get_config_spec()
        imported = expand_controller_config(config_cls(units=units), model, model.profile)
        for node in imported.nodes:
            model.add_node(node)
        for edge in imported.edges:
            model.add_edge(edge)
    if complete:
        model.set_node_config_value([GraphModel.CONTROLLER_ID, 'units'], units, force=True)
    return model

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestModelExport:
    """
        A model is refused unless the framework could build it.
    """

    def test_a_complete_graph_is_written(self, tmp_path, qapp) -> None:
        path = session_io.export_model(_graph(units=(8,), complete=True), tmp_path / 'model.scfg')
        assert path.exists()
        assert path.suffix == '.scfg'

    def test_an_empty_graph_is_refused(self, tmp_path, qapp) -> None:
        with pytest.raises(ValueError):
            session_io.export_model(_graph(), tmp_path / 'model.scfg')

    def test_a_graph_with_shapes_missing_is_refused(self, tmp_path, qapp) -> None:
        model = _graph()
        config_cls = REGISTRY.Neurons.get('lif_neuron').get_cls().get_config_spec()
        imported = expand_controller_config(config_cls.partial(), model, model.profile)
        for node in imported.nodes:
            model.add_node(node)
        for edge in imported.edges:
            model.add_edge(edge)
        with pytest.raises(ValueError):
            session_io.export_model(model, tmp_path / 'model.scfg')

    def test_a_graph_missing_a_connection_is_refused(self, tmp_path, qapp) -> None:
        model = _graph(units=(8,), complete=True)
        model.remove_node(next(node for node in model.nodes if node.name == 'delays'))
        with pytest.raises(ValueError):
            session_io.export_model(model, tmp_path / 'model.scfg')

    def test_nothing_is_left_behind_when_it_is_refused(self, tmp_path, qapp) -> None:
        path = tmp_path / 'model.scfg'
        with pytest.raises(ValueError):
            session_io.export_model(_graph(), path)
        assert not path.exists()
        assert list(tmp_path.glob('*.partial')) == []

    def test_what_was_written_instantiates(self, tmp_path, qapp) -> None:
        path = session_io.export_model(_graph(units=(8,), complete=True), tmp_path / 'model.scfg')
        neuron = spark.nn.neurons.LIFNeuron(config=session_io.load_model(path))
        outputs = neuron(in_spikes=spark.SpikeArray(jnp.zeros((8,), dtype=jnp.uint8)))
        assert outputs['out_spikes'].value.shape == (8,)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestAFileBringingItsOwnModels:
    """
        A session holds the same models a model file does, and has to bring them along just the same:
        decoding one asks the registry for every model it names.
    """

    def _brain_on_a_model_of_its_own(self, tmp_path, name: str):
        """
            A graph built on a neuron that exists only as a configuration.
        """
        neuron_path = tmp_path / f'{name}.scfg'
        spark.nn.neurons.ALIFNeuronConfig(units=(8,)).to_file(neuron_path, verbose=False)
        spark.register_neuron_from_config_file(name, neuron_path)
        model = GraphModel()
        model.set_profile(BRAIN_PROFILE, force=True)
        config = spark.nn.BrainConfig(modules_specs=[
            spark.ModuleSpecs(
                name='pool',
                module_cls=REGISTRY.Neurons.get(name).get_cls(),
                inputs={'in_spikes': [spark.PortMap(origin='__call__', port='in_spikes')]},
            ),
        ])
        imported = expand_controller_config(config, model, model.profile)
        for node in imported.nodes:
            model.add_node(node)
        for edge in imported.edges:
            model.add_edge(edge)
        return model

    def _under_another_name(self, path, name: str, other_name: str):
        """
            The same file, naming a model nothing has ever heard of.
        """
        other_path = path.with_name(f'{other_name}{path.suffix}')
        other_path.write_text(path.read_text().replace(name, other_name))
        return other_path

    def test_a_session_brings_the_models_it_names(self, tmp_path, qapp) -> None:
        model = self._brain_on_a_model_of_its_own(tmp_path, 'session_probe_neuron')
        path = session_io.save_session(model, tmp_path / 'session.sge')
        # Written uncompressed so the document can be read as it is.
        payload = session_io._read_json(path)
        session_io._write_json(tmp_path / 'plain.sge', payload, compress=False)
        other_path = self._under_another_name(
            tmp_path / 'plain.sge', 'session_probe_neuron', 'session_unheard_neuron',
        )
        assert REGISTRY.Neurons.get('session_unheard_neuron') is None
        session = session_io.load_session(other_path)
        assert [spec.name for spec in session.config.modules_specs] == ['pool']
        assert REGISTRY.Neurons.get('session_unheard_neuron') is not None

    def test_a_model_brings_the_models_it_names(self, tmp_path, qapp) -> None:
        model = self._brain_on_a_model_of_its_own(tmp_path, 'model_probe_neuron')
        model.set_node_config_value([GraphModel.CONTROLLER_ID, 'units'], (8,), force=True)
        path = session_io.export_model(model, tmp_path / 'model.scfg')
        plain = tmp_path / 'plain.scfg'
        spark.nn.BrainConfig.from_file(path).to_file(
            plain, compress=False, verbose=False, metadata={'layout': session_io.model_layout(path)},
        )
        other_path = self._under_another_name(plain, 'model_probe_neuron', 'model_unheard_neuron')
        assert REGISTRY.Neurons.get('model_unheard_neuron') is None
        config = session_io.load_model(other_path)
        assert [spec.name for spec in config.modules_specs] == ['pool']
        assert REGISTRY.Neurons.get('model_unheard_neuron') is not None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestModelLayout:
    """
        A model says where its nodes were left, beside the configuration and without disturbing it.
    """

    def test_the_layout_travels_with_the_model(self, tmp_path, qapp) -> None:
        model = _graph(units=(8,), complete=True)
        placed = {node.name: (float(node.pos[0]), float(node.pos[1])) for node in model.nodes}
        path = session_io.export_model(model, tmp_path / 'model.scfg')
        layout = session_io.model_layout(path)
        assert layout
        for name, position in layout.items():
            assert tuple(position) == pytest.approx(placed[name])

    def test_the_configuration_is_untouched_by_it(self, tmp_path, qapp) -> None:
        path = session_io.export_model(_graph(units=(8,), complete=True), tmp_path / 'model.scfg')
        config = spark.core.config.SparkConfig.from_file(path)
        assert config.units == (8,)
        assert {spec.name for spec in config.modules_specs}

    def test_a_model_written_elsewhere_says_nothing_about_it(self, tmp_path, qapp) -> None:
        path = tmp_path / 'plain.scfg'
        spark.nn.neurons.ALIFNeuronConfig(units=(8,)).to_file(path, verbose=False)
        assert session_io.model_layout(path) == {}

    def test_a_file_that_is_not_there_says_nothing_about_it(self, tmp_path, qapp) -> None:
        assert session_io.model_layout(tmp_path / 'nothing.scfg') == {}

    def _placed_model(self, editor, tmp_path, name: str = 'placed.scfg'):
        """
            A neuron laid out by hand and exported, with where every node was left.
        """
        editor.new_graph(NEURON_PROFILE)
        model = editor._scene.model
        config_cls = REGISTRY.Neurons.get('lif_neuron').get_cls().get_config_spec()
        editor.view.import_config(config_cls(units=(8,)), label='probe')
        model.set_node_config_value([GraphModel.CONTROLLER_ID, 'units'], (8,), force=True)
        for index, node in enumerate(model.nodes):
            node.pos = (120.0 * index, -60.0 * index)
        placed = {node.name: (float(node.pos[0]), float(node.pos[1])) for node in model.nodes}
        return session_io.export_model(model, tmp_path / name), placed

    @staticmethod
    def _shape(positions):
        """
            Where a set of nodes sits relative to itself, which is what survives being moved as a block.
        """
        positions = list(positions)
        origin_x = min(x for x, _ in positions)
        origin_y = min(y for _, y in positions)
        return sorted((round(x - origin_x, 3), round(y - origin_y, 3)) for x, y in positions)

    def test_importing_into_an_empty_session_keeps_the_placement(self, editor, tmp_path, monkeypatch) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, placed = self._placed_model(editor, tmp_path)
        editor.new_graph(NEURON_PROFILE)
        assert not editor._scene.model.nodes
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(path), '')))
        editor.import_model_file()
        imported = {node.name: (float(node.pos[0]), float(node.pos[1])) for node in editor._scene.model.nodes}
        assert imported == pytest.approx(placed)

    def test_merging_moves_the_block_without_reshaping_it(self, editor, tmp_path, monkeypatch) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, placed = self._placed_model(editor, tmp_path)
        editor.new_graph(NEURON_PROFILE)
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(path), '')))
        editor.import_model_file()
        standing = {node.id for node in editor._scene.model.nodes}
        monkeypatch.setattr(editor, '_ask_import_mode', lambda *a, **k: 'merge')
        editor.import_model_file()
        added = [node for node in editor._scene.model.nodes if node.id not in standing]
        assert len(added) == len(placed)
        assert self._shape((float(node.pos[0]), float(node.pos[1])) for node in added) == self._shape(placed.values())

    def test_the_editor_puts_the_nodes_back(self, editor, tmp_path) -> None:
        editor.new_graph(NEURON_PROFILE)
        model = editor._scene.model
        config_cls = REGISTRY.Neurons.get('lif_neuron').get_cls().get_config_spec()
        editor.view.import_config(config_cls(units=(8,)), label='probe')
        model.set_node_config_value([GraphModel.CONTROLLER_ID, 'units'], (8,), force=True)
        for index, node in enumerate(model.nodes):
            node.pos = (100.0 * index, -50.0 * index)
        placed = {node.name: (float(node.pos[0]), float(node.pos[1])) for node in model.nodes}
        path = session_io.export_model(model, tmp_path / 'placed.scfg')
        assert editor.open_model_file(path)
        reopened = {node.name: (float(node.pos[0]), float(node.pos[1])) for node in editor._scene.model.nodes}
        assert reopened == pytest.approx(placed)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestModelCheck:
    """
        The same verdict, without writing anything.
    """

    def test_a_complete_graph_has_nothing_to_report(self, qapp) -> None:
        assert session_io.check_model(_graph(units=(8,), complete=True)) == []

    def test_an_empty_graph_is_reported(self, qapp) -> None:
        assert session_io.check_model(_graph()) != []

    def test_the_report_names_what_is_missing(self, qapp) -> None:
        model = _graph(units=(8,), complete=True)
        model.remove_node(next(node for node in model.nodes if node.name == 'delays'))
        problems = session_io.check_model(model)
        assert any('not connected' in problem for problem in problems)

    def test_checking_writes_nothing(self, tmp_path, qapp) -> None:
        session_io.check_model(_graph(units=(8,), complete=True))
        assert list(tmp_path.iterdir()) == []

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestSessionSave:
    """
        A session is written however incomplete it is. This is the difference that matters: work in progress
        must always be storable, only a finished model has to pass.
    """

    @pytest.mark.parametrize('build', [
        pytest.param(lambda: _graph(), id='empty'),
        pytest.param(lambda: _graph(units=(8,)), id='shapes_unset'),
        pytest.param(lambda: _graph(BRAIN_PROFILE), id='another_controller'),
    ])
    def test_it_is_written(self, tmp_path, qapp, build) -> None:
        path = session_io.save_session(build(), tmp_path / 'session.sge')
        assert path.exists()
        assert path.suffix == '.sge'

    def test_a_graph_missing_a_connection_is_written_too(self, tmp_path, qapp) -> None:
        model = _graph(units=(8,), complete=True)
        model.remove_node(next(node for node in model.nodes if node.name == 'delays'))
        assert session_io.save_session(model, tmp_path / 'session.sge').exists()

    def test_a_graph_with_no_controller_is_not_a_session(self, tmp_path, qapp) -> None:
        with pytest.raises(ValueError):
            session_io.save_session(GraphModel(), tmp_path / 'session.sge')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestSessionRoundTrip:
    """
        What was saved comes back.
    """

    def test_the_controller_comes_back(self, tmp_path, qapp) -> None:
        session_io.save_session(_graph(BRAIN_PROFILE), tmp_path / 'session.sge')
        assert session_io.load_session(tmp_path / 'session.sge').profile is BRAIN_PROFILE

    def test_the_modules_come_back(self, tmp_path, qapp) -> None:
        session_io.save_session(_graph(units=(8,)), tmp_path / 'session.sge')
        session = session_io.load_session(tmp_path / 'session.sge')
        assert {spec.name for spec in session.config.modules_specs} >= {'delays', 'synapses', 'soma'}

    def test_an_incomplete_session_comes_back(self, tmp_path, qapp) -> None:
        model = _graph(units=(8,), complete=True)
        model.remove_node(next(node for node in model.nodes if node.name == 'delays'))
        session_io.save_session(model, tmp_path / 'session.sge')
        session = session_io.load_session(tmp_path / 'session.sge')
        assert 'delays' not in {spec.name for spec in session.config.modules_specs}

    def test_the_layout_comes_back(self, tmp_path, qapp) -> None:
        model = _graph(units=(8,))
        placed = {node.name: (float(node.pos[0]), float(node.pos[1])) for node in model.nodes}
        session_io.save_session(model, tmp_path / 'session.sge')
        session = session_io.load_session(tmp_path / 'session.sge')
        assert session.layout
        for name, position in session.layout.items():
            assert tuple(position) == pytest.approx(placed[name])

    def test_the_controller_settings_come_back(self, tmp_path, qapp) -> None:
        model = _graph(units=(8,), complete=True)
        model.set_node_config_value([GraphModel.CONTROLLER_ID, 'inhibitory_rate'], 0.42, force=True)
        session_io.save_session(model, tmp_path / 'session.sge')
        session = session_io.load_session(tmp_path / 'session.sge')
        assert session.config.inhibitory_rate == 0.42
        assert session.config.units == (8,)

    def test_a_model_is_not_a_session(self, tmp_path, qapp) -> None:
        path = session_io.export_model(_graph(units=(8,), complete=True), tmp_path / 'model.scfg')
        with pytest.raises(Exception):
            session_io.load_session(path)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
