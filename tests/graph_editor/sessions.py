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
