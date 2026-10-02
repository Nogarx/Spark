#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import gc
import sys
import json
import time
import shutil
import sqlite3
import pathlib
import textwrap
import weakref
import subprocess
import pytest
import numpy as np
import jax.numpy as jnp
import spark

pytest.importorskip('PySide6', reason='the run viewer needs PySide6')
from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import QLabel

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

R = spark.recording
SIGNAL = np.full((8,), 1.0, dtype=np.float16)
RECORDING_TESTS = str(pathlib.Path(__file__).resolve().parents[1] / 'recording')
INPUT_RATE = R.SummaryProbe('first_pool.__call__:in_spikes', reduce=('active_fraction',))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _brain():
    pool = lambda name, units, origins: spark.ModuleSpecs(
        name = name,
        module_cls = spark.nn.neurons.ALIFNeuron,
        inputs = {'in_spikes': [spark.PortMap(origin=o, port=p) for o, p in origins]},
        config = spark.nn.neurons.ALIFNeuronConfig(_s_units=units, inhibitory_rate=0.3, synapses__kernel__scale=3000),
    )
    config = spark.nn.BrainConfig(modules_specs=[
        spark.ModuleSpecs(name='spiker', module_cls=spark.nn.interfaces.PoissonSpiker,
                          inputs={'signal': [spark.PortMap('__call__', 'signal')]}),
        pool('first_pool', (16,), [('spiker', 'spikes')]),
        pool('second_pool', (8,), [('first_pool', 'out_spikes'), ('second_pool', 'out_spikes')]),
        spark.ModuleSpecs(name='integrator', module_cls=spark.nn.interfaces.ExponentialIntegrator,
                          inputs={'spikes': [spark.PortMap('second_pool', 'out_spikes')]}, outputs={'action': 'signal'},
                          config=spark.nn.interfaces.ExponentialIntegratorConfig(num_outputs=2)),
    ], seed=7)
    brain = spark.nn.Brain(config=config)
    brain(signal=spark.FloatArray(jnp.asarray(SIGNAL)))
    return brain

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _record(root, calls=6, hparams=None):
    """
        Six calls of ten steps: summary on every step, in groups of ten, activity on every other call with an
        observation logged before each call, weights on every third.
    """
    brain = _brain()
    measurements = [
        R.Measurements('summary', (*R.presets.summary(brain), INPUT_RATE), trigger=R.Always(), group=10),
        R.Measurements('activity', R.presets.activity(brain), trigger=R.Every(20, length=10), raw=('env/observation',),
                    views={'env/observation': {'kind': 'vector', 'labels': ['a', 'b', 'c']}}),
        R.Measurements('weights', R.presets.weights(brain), trigger=R.Every(30, length=10), group=10),
    ]
    recorder = R.Recorder(root, brain, measurements, hparams=hparams or {'lr': 0.1})
    runner = R.Runner(brain, recorder)
    for call in range(calls):
        recorder.tag(episode=call // 2)
        recorder.raw('env/observation', np.array([call, call * 2, call * 3], dtype=np.float32))
        runner.run(10, {'signal': SIGNAL})
        recorder.log(reward=float(call))
    runner.close()
    return brain, recorder.path

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _copy(recorded, tmp_path):
    path = tmp_path / recorded[1].name
    shutil.copytree(recorded[1], path)
    return path

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _python(script):
    """
        Runs ``script`` in a new interpreter, on the processor, and returns the last line it printed.
    """
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(script)], capture_output=True, text=True, timeout=600, env=env)
    return result.stdout.strip().splitlines()[-1]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _wait_for(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, 'timed out'
        time.sleep(0.05)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture(scope='module')
def recorded(tmp_path_factory):
    return _record(tmp_path_factory.mktemp('runs'))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture(autouse=True)
def _explorations(tmp_path_factory, monkeypatch):
    """
        A folder of explorations per test, so that no window resumes what the window of another test kept.
    """
    import spark.graph_editor.runs.workspace as workspace
    root = tmp_path_factory.mktemp('explorations')
    monkeypatch.setattr(workspace, 'explorations_path', lambda: root)
    return root

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture(autouse=True)
def _close_left_open(qapp):
    """
        Closes the recorders and the viewer windows a test left open, whatever its outcome.
    """
    yield
    from spark.graph_editor.runs.viewer import RunViewerWindow
    for recorder in list(spark.recording.recorder._OPEN):
        try:
            recorder.close()
        except Exception:
            pass
    for widget in qapp.topLevelWidgets():
        if isinstance(widget, RunViewerWindow) and widget.isVisible():
            widget.close()
    qapp.processEvents()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def viewer(qapp, recorded):
    from spark.graph_editor.runs.viewer import RunViewerWindow
    window = RunViewerWindow(recorded[1])
    window.show()
    qapp.processEvents()
    yield window
    window.close()
    qapp.processEvents()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _select(viewer, qapp, name):
    from spark.graph_editor.view.node_item import NodeItem
    viewer.scene.clearSelection()
    for item in viewer.scene.items():
        if isinstance(item, NodeItem) and item.model.name == name:
            item.setSelected(True)
    qapp.processEvents()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _plots(panel, kind=None):
    return [w for w in panel.plots() if kind is None or isinstance(w, kind)]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRunData:

    def test_probes_belong_to_their_node(self, recorded) -> None:
        from spark.graph_editor.runs.data import RunData
        data = RunData(recorded[1])
        keys = {probe.key for _, probe in data.probes_of('first_pool')}
        assert 'first_pool.soma:spikes@summary' in keys and 'first_pool.synapses.kernel@snapshot' in keys
        assert {probe.key for _, probe in data.probes_of('signal')} == {'__call__:signal@trace'}
        assert data.probes_of('integrator') and not data.probes_of('nothing')

    def test_activity_is_the_rate_of_the_spikes_of_the_node_in_hertz(self, recorded) -> None:
        from spark.graph_editor.runs.data import RunData
        data = RunData(recorded[1])
        data.dt = 0.5                                                   # milliseconds per step
        run = R.load(recorded[1])
        t, rate = run.scalar('summary/first_pool.soma:spikes/active_fraction')
        _, inputs = run.scalar(f'summary/{INPUT_RATE.key}/active_fraction')
        assert not np.allclose(rate, inputs)                            # the rate of its inputs is not counted
        for step, value in zip(t, rate):
            assert data.activity('first_pool', int(step) + 3) == pytest.approx(value * 2000.0)
        assert data.activity('spiker', 30) is None

    def test_segments_spans_groups_and_rows(self, recorded) -> None:
        from spark.graph_editor.runs.data import RunData
        data = RunData(recorded[1])
        assert data.segments('summary') == [(0, 60)]
        assert data.segments('activity') == [(0, 10), (20, 30), (40, 50)]
        assert data.span_at('activity', 25)[0] == 20 and data.span_at('activity', 35)[0] == 20
        assert data.span_at('activity', -1) is None
        assert data.groups('summary')[0].tolist() == [0, 10, 20, 30, 40, 50] and data.groups('weights')[0].tolist() == [0, 30]
        assert data.group_at('weights', 45)[0] == 30 and data.group_at('summary', -1) is None
        times, values = data.rows('activity', 'first_pool.soma.potential@trace', 25)
        np.testing.assert_array_equal(times, np.arange(20, 30))
        whole = R.load(recorded[1]).timeline('activity')
        np.testing.assert_array_equal(values, whole['first_pool.soma.potential@trace'][10:20])
        start, rates = data.group_value('summary', 'first_pool.soma:spikes@summary#active_fraction', 45)
        assert start == 40 and rates == pytest.approx(data.value_at('summary/first_pool.soma:spikes/active_fraction', 40))
        assert data.logged_keys() == ['reward']

    def test_the_last_frame_at_or_before_a_step(self, recorded) -> None:
        from spark.graph_editor.runs.data import RunData
        data = RunData(recorded[1])
        # Frames are logged before the calls they are the input of: steps 0, 20 and 40 are recorded.
        for step, (t, frame) in ((0, (0, [0, 0, 0])), (25, (20, [2, 4, 6])), (59, (40, [4, 8, 12]))):
            found = data.raw_at('activity', 'env/observation', step)
            assert found[0] == t and list(found[1]) == frame
        assert data.raw_at('activity', 'env/observation', -1) is None

    def test_spans_and_groups_of_a_window_never_written_are_not_shown_once_the_run_is_over(self, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.data import RunData
        path = _copy(recorded, tmp_path)
        with sqlite3.connect(path / 'index.sqlite') as connection:
            connection.execute("INSERT INTO spans VALUES ('activity', 99, 60, 10)")
            connection.execute("INSERT INTO groups VALUES ('weights', 99, 60, 10)")
        data = RunData(path)
        assert data.span_at('activity', 65)[2] == 99 and data.span_at('activity', 65, written=True)[0] == 40
        assert data.group_at('weights', 65)[0] == 30
        assert data.segments('activity') == [(0, 10), (20, 30), (40, 50)]
        assert data.rows('activity', 'first_pool.soma.potential@trace', 65)[0][0] == 40

    def test_a_damaged_window_file_reads_as_empty(self, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.data import RunData
        path = _copy(recorded, tmp_path)
        (path / 'windows' / 'activity' / '000000.npz').write_bytes(b'not a zip file')
        data = RunData(path)
        assert data.rows('activity', 'first_pool.soma.potential@trace', 25) is None
        assert data.arrays('activity', 0, ('span_t0',)) == {}
        assert data.segments('summary') == [(0, 60)]

    def test_arrays_are_kept_up_to_a_size(self, recorded) -> None:
        from spark.graph_editor.runs.data import RunData
        data = RunData(recorded[1], cached_bytes=1)
        key = 'first_pool.soma.potential@trace'
        first = data.arrays('activity', 0, (key,))[key]
        data.arrays('weights', 0, ('span_t0',))
        assert len(data._cache) == 1 and data._cached_bytes > 0
        np.testing.assert_array_equal(data.arrays('activity', 0, (key,))[key], first)

    def test_an_envelope_keeps_the_extremes_beside_nans(self) -> None:
        from spark.graph_editor.runs.data import _Table
        values = np.sin(np.arange(4000) / 30.0)
        values[::97] = np.nan
        values[1500], values[3100] = 5.0, -5.0
        table = _Table(np.int64, np.float64)
        table.extend(np.column_stack([np.arange(4000), values]))
        table.reduce(100)
        t, v = table.columns()
        assert len(t) <= 160 and np.all(np.diff(t) >= 0)
        assert np.nanmax(v) == 5.0 and np.nanmin(v) == -5.0 and np.isnan(v).any()
        # Every span keeps its finite maximum, whether or not it holds a NaN.
        width = 3999 // 50 + 1
        for span in range(50):
            inside = slice(span * width, (span + 1) * width)
            if np.isfinite(values[inside]).any():
                kept = v[(t >= span * width) & (t < (span + 1) * width)]
                assert np.nanmax(kept) == np.nanmax(values[inside])

    def test_rows_rolled_back_are_read_again(self, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.data import RunData
        path = _copy(recorded, tmp_path)
        data = RunData(path)
        assert len(data.scalar('reward')[0]) == 6
        # As a commit a process left unfinished, read before a recorder rolled it back: the last reward.
        connection = sqlite3.connect(path / 'index.sqlite')
        connection.execute("DELETE FROM scalars WHERE rowid = (SELECT MAX(rowid) FROM scalars)")
        connection.commit()
        connection.close()
        assert 'scalars' in data.refresh() and len(data.scalar('reward')[0]) == 5

    def test_the_live_tail_reads_only_new_rows(self, recorded, tmp_path, monkeypatch) -> None:
        from spark.graph_editor.runs.data import RunData
        data = RunData(_copy(recorded, tmp_path))
        data.scalar('reward')
        statements = []
        query = data.run._query
        monkeypatch.setattr(data.run, '_query', lambda sql, args=(): (statements.append(sql), query(sql, args))[1])
        data.refresh()
        reads = [sql for sql in statements if 'FROM scalars' in sql]
        assert reads and all('rowid >' in sql or 'MAX(rowid)' in sql for sql in reads)
        assert not any('key =' in sql or 'key IN' in sql for sql in reads)

    def test_badges_do_not_read_whole_series(self, recorded) -> None:
        from spark.graph_editor.runs.data import RunData
        data = RunData(recorded[1])
        key = 'summary/first_pool.soma:spikes/active_fraction'
        steps, values = R.load(recorded[1]).scalar(key)
        assert data.value_at(key, int(steps[2]) + 1) == values[2] and key not in data.scalars
        assert data.activity('first_pool', int(steps[3])) == pytest.approx(values[3] / data.dt * 1000.0)
        assert key not in data.scalars

    def test_a_run_whose_directory_cannot_be_listed_opens(self, qapp, recorded, monkeypatch) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        import spark.graph_editor.runs.workspace as module
        def unlistable(root):
            raise PermissionError(f'cannot list {root}')
        monkeypatch.setattr(module, 'list_runs', unlistable)
        viewer = RunViewerWindow(recorded[1])
        assert viewer.project.runs == [] and viewer.data.path == recorded[1]
        viewer.close()

    def test_a_window_that_cannot_be_read_is_not_read_on_every_move(self, recorded, tmp_path, monkeypatch) -> None:
        from spark.graph_editor.runs.data import RunData
        path = _copy(recorded, tmp_path)
        data = RunData(path)
        (path / 'windows' / 'activity' / '000000.npz').write_bytes(b'damaged')
        reads = []
        arrays = data.arrays
        monkeypatch.setattr(data, 'arrays', lambda *args: (reads.append(args[1]), arrays(*args))[1])
        assert data.raw_at('activity', 'env/observation', 45) is None
        assert data.raw_at('activity', 'env/observation', 45) is None
        assert reads == [0]

    def test_spans_grow_with_the_calls_of_a_run_being_written(self, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.data import RunData, _Table
        from spark.recording import store
        path = _copy(recorded, tmp_path)
        info = json.loads((path / 'run.json').read_text())
        store.write_json(path / 'run.json', {**info, 'status': 'running', 'heartbeat': store.now(), 'heartbeat_every': 60})
        data = RunData(path)
        rng = np.random.default_rng(0)
        for trial in range(200):
            table = data._recorded['trial'] = _Table(np.int64, np.int64, np.int64)
            data._segments.pop('trial', None)
            t0 = 0
            for _ in range(6):
                # Spans in order of their first step, some overlapping the spans before, some apart.
                starts = t0 + np.cumsum(rng.integers(-40, 60, rng.integers(1, 5)).clip(0))
                table.extend(np.column_stack([starts, rng.integers(1, 90, len(starts)), np.zeros(len(starts))]))
                t0 = int(starts[-1])
                incremental = data.spans('trial')
                data._segments.pop('trial')
                np.testing.assert_array_equal(incremental, data.spans('trial'))

    def test_images_reduce_their_values_in_blocks(self) -> None:
        from spark.graph_editor.runs.plots import _shrink
        values = np.random.default_rng(0).random((1003, 37))
        for axis, limit in ((0, 100), (1, 5), (0, 1003), (0, 1)):
            n, block = values.shape[axis], -(-values.shape[axis] // limit)
            pad = [(0, 0)] * 2
            pad[axis] = (0, (-n) % block)
            padded = np.pad(values, pad, mode='edge')
            shape = list(padded.shape)
            shape[axis:axis + 1] = [shape[axis] // block, block]
            expected = values if n <= limit else padded.reshape(shape).mean(axis=axis + 1)
            got = _shrink(values, axis, limit)
            # The last block is the mean of its own values rather than of values repeated.
            if n > limit and n % block:
                last = [slice(None)] * 2
                last[axis] = slice(-1, None)
                expected[tuple(last)] = values.take(np.arange(n - n % block, n), axis=axis).mean(axis=axis, keepdims=True)
            np.testing.assert_allclose(got, expected)
        bits = np.random.default_rng(1).random((1001, 9)) < 0.01
        assert _shrink(bits, 0, 100).tolist() == [bits[i:i + 11].any(axis=0).tolist() for i in range(0, 1001, 11)]

    def test_rows_out_of_order_are_sorted(self) -> None:
        from spark.graph_editor.runs.data import _Table
        table = _Table(np.int64, np.float64)
        table.extend([(5, 1.0), (7, 2.0)])
        before = table.columns()
        table.extend([(6, 3.0), (9, None)])
        t, v = table.columns()
        assert t.tolist() == [5, 6, 7, 9] and v[:3].tolist() == [1.0, 3.0, 2.0] and np.isnan(v[3])
        assert before[0].tolist() == [5, 7]

    def test_a_run_that_stops_writing_turns_crashed(self, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.data import RunData
        from spark.recording import store
        path = _copy(recorded, tmp_path)
        info = json.loads((path / 'run.json').read_text())
        store.write_json(path / 'run.json', {**info, 'status': 'running', 'heartbeat': store.now(), 'heartbeat_every': 60})
        data = RunData(path)
        assert data.status == 'running' and data.segments('activity') == [(0, 10), (20, 30), (40, 50)]
        store.write_json(path / 'run.json', {**info, 'status': 'running', 'heartbeat': '2000-01-01T00:00:00+00:00', 'heartbeat_every': 60})
        assert 'info' in data.refresh() and data.status == 'crashed'

    def test_a_long_series_is_kept_as_its_envelope(self, tmp_path, monkeypatch) -> None:
        from spark.graph_editor.runs import data as module
        monkeypatch.setattr(module, 'POINTS', 20)
        values = np.cos(np.arange(300) / 20.0)
        values[123] = 9.0
        recorder = R.Recorder(tmp_path, _brain(), [R.Measurements('a', (INPUT_RATE,), trigger=R.Manual(), group=1)])
        for step, value in enumerate(values[:200]):
            recorder.log(x=value, step=step)
        recorder.flush()
        data = module.RunData(recorder.path)
        steps, got = data.scalar('x')
        assert len(steps) <= 30 and got.max() == 9.0 and 'x' in data._reduced
        assert data.value_at('x', 150) == values[150]                  # read from the run, not from the envelope
        for step, value in enumerate(values[200:], start=200):
            recorder.log(x=value, step=step)
        recorder.flush()
        assert 'scalars' in data.refresh()
        steps, got = data.scalar('x')
        assert steps[-1] == 299 and len(steps) <= 2 * 20 + 2 and got.max() == 9.0
        recorder.close()

    def test_a_run_moved_away_and_back_is_read_again(self, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.data import RunData
        path = _copy(recorded, tmp_path)
        data = RunData(path)
        moved = path.with_name('moved')
        path.rename(moved)
        assert data.refresh() == {'error'} and data.error
        assert data.scalar('not read yet') [0].size == 0
        moved.rename(path)
        assert 'error' in data.refresh() and data.error is None

    def test_a_window_readable_later_is_read_then(self, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.data import RunData
        path = _copy(recorded, tmp_path)
        file = path / 'windows' / 'activity' / '000000.npz'
        file.rename(file.with_suffix('.away'))                          # as a file not visible yet on a network file system
        data = RunData(path)
        assert data.arrays('activity', 0, ('span_t0',)) == {}
        file.with_suffix('.away').rename(file)
        assert list(data.arrays('activity', 0, ('span_t0',))['span_t0']) == [0, 20, 40]

    def test_spans_lost_are_hidden_once_the_run_is_resumed(self, tmp_path) -> None:
        from spark.graph_editor.runs.data import RunData
        # Four calls, the last one in a window never written: the process ends without closing.
        path = pathlib.Path(_python(f'''
            import os, sys, time
            sys.path.insert(0, {RECORDING_TESTS!r})
            import numpy as np, spark
            from cases import CASES
            R = spark.recording
            brain, _, _ = CASES['brain']()
            probe = R.SummaryProbe('first_pool.__call__:in_spikes', reduce=('active_fraction',))
            recorder = R.Recorder({str(tmp_path)!r}, brain, [R.Measurements('a', (probe,), trigger=R.Always(), group=5)], flush_steps=10)
            runner = R.Runner(brain, recorder)
            for _ in range(3):
                runner.run(5, {{'signal': np.ones(8)}})
            recorder.flush()
            runner.run(5, {{'signal': np.ones(8)}})
            recorder._queue.join()
            time.sleep(1.0)
            print(recorder.path, flush=True)
            os._exit(1)
        '''))
        data = RunData(path)
        assert data.status == 'running' and data.segments('a') == [(0, 20)]     # listed spans count while written
        resumed = R.Recorder.resume(path, _brain())
        data.refresh()
        assert data.status == 'running' and data.segments('a') == [(0, 15)]
        resumed.close()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRunViewer:

    def test_the_graph_of_the_model_is_shown(self, viewer) -> None:
        names = {node.name for node in viewer.scene.model.nodes}
        assert {'spiker', 'first_pool', 'second_pool', 'integrator', 'signal', 'action'} <= names
        assert not viewer.scene.selectedItems()

    def test_the_graph_is_shown_without_the_styles_of_the_editor(self, qapp, recorded, monkeypatch) -> None:
        from spark.graph_editor.styles.manager import STYLES
        from spark.graph_editor.runs.viewer import RunViewerWindow
        monkeypatch.setattr(STYLES, '_config', {})
        window = RunViewerWindow(recorded[1])
        assert window.badges and {'first_pool', 'second_pool'} <= set(window.badges)
        window.close()
        qapp.processEvents()

    def test_the_graph_is_read_only(self, viewer, qapp) -> None:
        from PySide6.QtGui import QKeyEvent
        from spark.graph_editor.view.node_item import NodeItem, PortItem
        from spark.graph_editor.view.pipe_item import PipeItem
        nodes = [item for item in viewer.scene.items() if isinstance(item, NodeItem)]
        assert nodes and not any(item.flags() & item.GraphicsItemFlag.ItemIsMovable for item in nodes)
        _select(viewer, qapp, 'first_pool')
        viewer.view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier))
        qapp.processEvents()
        assert 'first_pool' in {node.name for node in viewer.scene.model.nodes}
        # Pipes and ports take no clicks, which would reshape or disconnect them.
        items = [item for item in viewer.scene.items() if isinstance(item, (PipeItem, PortItem))]
        assert any(isinstance(item, PipeItem) for item in items)
        assert all(item.acceptedMouseButtons() == Qt.MouseButton.NoButton for item in items)

    def test_the_graph_stays_fitted_as_the_window_is_laid_out(self, qapp, recorded) -> None:
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QWheelEvent
        from spark.graph_editor.runs.viewer import RunViewerWindow
        viewer = RunViewerWindow(recorded[1])
        # Fitted first before the window has its size, as when a window is built and shown later.
        qapp.processEvents()
        viewer.show()
        qapp.processEvents()
        whole = lambda: viewer.view.mapToScene(viewer.view.viewport().rect()).boundingRect().contains(viewer.scene.itemsBoundingRect())
        assert whole()
        viewer.resize(viewer.width() - 300, viewer.height() - 200)
        qapp.processEvents()
        assert whole()
        # Zoomed by hand, it is left as it is.
        position = QPointF(50, 50)
        viewer.view.wheelEvent(QWheelEvent(position, position, QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton,
                                           Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.NoScrollPhase, False))
        zoom = viewer.view.transform().m11()
        viewer.resize(viewer.width() + 300, viewer.height() + 200)
        qapp.processEvents()
        assert viewer.view.transform().m11() == zoom and not viewer.view.fitting
        viewer.close()
        qapp.processEvents()

    def test_the_panels_of_the_left_are_runs_and_details(self, viewer) -> None:
        assert (viewer.dock_runs.windowTitle(), viewer.dock_run.windowTitle()) == ('Runs', 'Details')
        # The layouts explorations kept before name it as before.
        assert viewer.dock_run.objectName() == 'dockRun'

    def test_badges_show_the_rate_at_the_cursor(self, viewer) -> None:
        key = 'summary/first_pool.soma:spikes/active_fraction'
        viewer.data.dt = 0.25
        for step in (35, 55):
            viewer.set_cursor(step)
            assert viewer.badges['first_pool'].value == pytest.approx(viewer.data.value_at(key, step) * 4000.0)
        assert viewer.badges['spiker'].value is None
        viewer.data.dt = 1.0

    def test_selecting_a_node_shows_what_was_recorded_for_it(self, viewer, qapp) -> None:
        from spark.graph_editor.runs.plots import SeriesPlot, ImagePlot, MatrixPlot
        _select(viewer, qapp, 'first_pool')
        assert viewer.probe_panel.node == 'first_pool'
        plots = _plots(viewer.probe_panel)
        assert {SeriesPlot, ImagePlot, MatrixPlot} <= {type(p) for p in plots}
        viewer.set_cursor(25)
        series = [p for p in plots if isinstance(p, SeriesPlot) and p.title.endswith('active_fraction')][0]
        assert series.cursor == 25 and len(series.series[0][1]) == 6
        _select(viewer, qapp, None)
        assert viewer.probe_panel.node is None
        assert [p.title for p in _plots(viewer.probe_panel, SeriesPlot)] == ['reward']

    def test_plots_at_the_cursor_follow_the_span(self, viewer, qapp) -> None:
        from spark.graph_editor.runs.plots import ImagePlot, MatrixPlot
        _select(viewer, qapp, 'first_pool')
        raster = [p for p in _plots(viewer.probe_panel, ImagePlot) if p.title.endswith('spikes · raster')][0]
        # The recorded spans of "activity" share one window; each is shown on its own.
        for cursor, span in ((25, (20.0, 30.0)), (45, (40.0, 50.0)), (5, (0.0, 10.0))):
            viewer.set_cursor(cursor)
            assert raster.x_range == span
        weights = [p for p in _plots(viewer.probe_panel, MatrixPlot) if 'snapshot' in p.title][0]
        viewer.set_cursor(45)
        assert weights._image is not None and weights.subtitle

    def test_the_inputs_at_the_cursor(self, viewer) -> None:
        from spark.graph_editor.runs.plots import BarPlot
        viewer.set_cursor(55)
        plots = viewer.inputs_panel.findChildren(BarPlot)
        observation = [p for p in plots if p.title.startswith('env/observation')][0]
        assert list(observation.values) == [4.0, 8.0, 12.0] and observation.labels == ['a', 'b', 'c']
        assert observation.subtitle.endswith('at step 40, 15 steps earlier')
        signal = [p for p in plots if p.title.startswith('signal')][0]
        np.testing.assert_array_equal(signal.values, SIGNAL.astype(np.float64))

    def test_single_values_are_listed(self, qapp, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        brain = _brain()
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('env', (), trigger=R.Always(), raw=('env/reward', 'env/state'))])
        recorder.raw('env/reward', np.float32(1.5))
        recorder.raw('env/state', np.arange(3, dtype=np.float32))
        R.Runner(brain, recorder).run(5, {'signal': SIGNAL})
        recorder.close()
        viewer = RunViewerWindow(recorder.path)
        viewer.set_cursor(3)
        reward, state = viewer.inputs_panel._entries
        assert not reward.name.isHidden() and reward.plot.isHidden() and reward.value.text() == '1.5'
        assert reward.step.text() == 'at step 0, 3 steps earlier'
        assert reward.name.text() == 'env/reward (env)' and state.name.isHidden() and not state.plot.isHidden()
        viewer.close()
        qapp.processEvents()

    def test_the_warnings_of_the_run_are_listed(self, qapp, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        brain = _brain()
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)])
        runner = R.Runner(brain, recorder)
        runner.run(5, {'signal': SIGNAL})
        with pytest.warns(R.RecordingWarning):
            recorder.log(loss='high')
        runner.run(5, {'signal': SIGNAL})
        recorder.close()
        viewer = RunViewerWindow(recorder.path)
        labels = [viewer.run_panel._notable.itemAt(i).widget() for i in range(viewer.run_panel._notable.count())]
        assert viewer.run_panel._notable_heading.text() == 'Warnings (1)' and len(labels) == 1
        assert 'step 5' in labels[0].text() and 'takes a number' in labels[0].text()
        labels[0].linkActivated.emit('5')
        assert viewer.cursor == 5
        viewer.close()
        qapp.processEvents()

    def test_an_image_frame_is_drawn_with_its_shape(self, qapp, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        brain = _brain()
        frames = R.Measurements('frames', (), trigger=R.Always(), raw=('env/frame', 'env/gray'),
                             views={'env/frame': {'kind': 'image', 'shape': [4, 6, 3]}, 'env/gray': {'kind': 'image'}})
        recorder = R.Recorder(tmp_path, brain, [frames])
        recorder.raw('env/frame', np.arange(72, dtype=np.float32))
        recorder.raw('env/gray', np.ones((4, 6, 1), dtype=np.uint8))
        R.Runner(brain, recorder).run(5, {'signal': SIGNAL})
        recorder.close()
        viewer = RunViewerWindow(recorder.path)
        viewer.set_cursor(3)
        pixmaps = [label.pixmap() for label in viewer.inputs_panel.findChildren(QLabel) if not label.pixmap().isNull()]
        assert len(pixmaps) == 2 and all(abs(p.width() / p.height() - 1.5) < 0.05 for p in pixmaps)
        viewer.close()
        qapp.processEvents()

    def test_the_timeline_shows_all_measurements(self, viewer) -> None:
        rows = {name: np.asarray(spans).tolist() for name, spans in viewer.timeline.rows}
        assert rows['activity'] == [[0, 10], [20, 30], [40, 50]] and rows['weights'] == [[0, 10], [30, 40]]
        assert viewer.timeline.total == 60
        viewer.timeline.cursor_moved.emit(33)
        assert viewer.cursor == 33 and viewer.probe_panel.cursor == 33

    def test_a_summary_spread_shares_one_plot(self, viewer, qapp) -> None:
        from spark.graph_editor.runs.plots import SeriesPlot, format_value
        _select(viewer, qapp, 'first_pool')
        titles = [p.title for p in _plots(viewer.probe_panel, SeriesPlot)]
        assert 'potential · mean ± std, min to max' in titles and not any(t.endswith('· std') for t in titles)
        spread = [p for p in _plots(viewer.probe_panel, SeriesPlot) if p.title.startswith('potential')][0]
        assert [s[0] for s in spread.series] == ['mean'] and [b[0] for b in spread.bands] == [('min', 'max'), 'std']
        viewer.set_cursor(25)
        mean = viewer.data.value_at('summary/first_pool.soma.potential/mean', 25)
        assert spread.readout.startswith(f'mean {format_value(mean)}') and 'max' in spread.readout and 'std' in spread.readout

    def test_scalar_plots_show_the_steps_of_the_timeline(self, viewer, qapp) -> None:
        from spark.graph_editor.runs.plots import SeriesPlot
        _select(viewer, qapp, 'first_pool')
        viewer.timeline.show_range(10, 30)
        plots = [p for p in _plots(viewer.probe_panel, SeriesPlot) if not p.title.endswith('trace')]
        assert plots and all(p.x_range == (10.0, 30.0) for p in plots)
        viewer.timeline.show_range(0, 60)
        assert all(p.x_range == (0.0, 60.0) for p in plots)

    def test_the_value_under_the_mouse(self, viewer, qapp) -> None:
        from spark.graph_editor.runs.plots import SeriesPlot, format_value
        _select(viewer, qapp, None)
        (reward,) = _plots(viewer.probe_panel, SeriesPlot)
        assert reward.hover_text(35.0, 0.0) == 'step 35\nreward: ' + format_value(viewer.data.value_at('reward', 35))
        viewer.set_cursor(35)
        assert reward.readout == format_value(viewer.data.value_at('reward', 35))

    def test_the_cursor_moves_by_keys_and_by_recorded_spans(self, viewer) -> None:
        viewer._go(0)
        viewer._jump(1)
        # Recorded spans start at steps 0 ("summary", one joined span), 20 and 40 ("activity"), and 30 ("weights").
        assert viewer.cursor == 20
        viewer._jump(1)
        assert viewer.cursor == 30
        viewer._jump(-1)
        assert viewer.cursor == 20
        viewer._step_by(1, True)
        assert viewer.cursor == 21
        viewer.timeline.show_range(0, 60)
        viewer._step_by(-1, False)
        assert viewer.cursor == 20
        viewer._go(viewer.data.step)
        assert viewer.cursor == 60

    def test_events_are_described_in_the_timeline(self, viewer) -> None:
        from spark.graph_editor.runs.timeline import _describe
        assert _describe((1200, 'warning'), {'message': 'No frames.'}) == 'warning at step 1,200: No frames.'
        assert _describe((5, 'episode_end'), {'t': 5, 'kind': 'episode_end', 'wall': 0.0, 'outcome': 'fell'}) == 'episode_end at step 5: "outcome": "fell"'
        from spark.graph_editor.styles.run_viewer import THEME
        rows = viewer.timeline.rows
        top = THEME.timeline.ruler_height
        assert viewer.timeline.hover_text(viewer.timeline._x(25), top + 1).startswith(rows[0][0])

    def test_the_plots_of_a_node_are_in_tabs_by_measurements(self, viewer, qapp) -> None:
        tabs = viewer.probe_panel.tabs
        _select(viewer, qapp, 'first_pool')
        assert [tabs.tabText(i) for i in range(tabs.count())] == ['summary', 'activity', 'weights', 'Inputs']
        tabs.setCurrentIndex(2)
        _select(viewer, qapp, 'second_pool')
        assert tabs.tabText(tabs.currentIndex()) == 'weights'
        # The inputs are a tab too, kept from node to node.
        tabs.setCurrentIndex(3)
        _select(viewer, qapp, 'first_pool')
        assert tabs.tabText(tabs.currentIndex()) == 'Inputs' and tabs.currentWidget().widget() is viewer.inputs_panel
        _select(viewer, qapp, None)
        assert [tabs.tabText(i) for i in range(tabs.count())] == ['logged', 'Inputs']

    def test_a_box_zooms_every_plot_to_its_steps(self, viewer, qapp) -> None:
        from spark.graph_editor.runs.plots import SeriesPlot
        _select(viewer, qapp, 'first_pool')
        scalars = [p for p in _plots(viewer.probe_panel, SeriesPlot) if not p.title.endswith('trace')]
        norm = [p for p in scalars if p.title == 'kernel · change norm'][0]
        norm.select(20.0, 40.0, 0.0, 1.0)
        # The steps go to the timeline and every scalar plot; the values to the plot zoomed only.
        assert viewer.timeline.view == (20.0, 40.0) and all(p.x_range == (20.0, 40.0) for p in scalars)
        assert norm.zoomed == (0.0, 1.0) and all(p.zoomed is None for p in scalars if p is not norm)
        norm.show_every_step()
        norm.fit()
        assert viewer.timeline.view == (0.0, 60.0) and norm.zoomed is None

    def test_the_menu_bar_opens_runs_and_quits(self, qapp, recorded, tmp_path, monkeypatch) -> None:
        from PySide6.QtWidgets import QMessageBox
        from spark.graph_editor.runs.viewer import RunViewerWindow
        viewer = RunViewerWindow(recorded[1])
        viewer.show()
        assert [action.text() for action in viewer.menuBar().actions()] == ['&File', '&Edit', '&Window']
        assert [action.text() for action in viewer._file_menu.actions() if action.text()] == ['Open...', 'Open Exploration...', 'Save Exploration As...', 'Close Window', 'Quit']
        warned = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warned.append(args))
        assert viewer.open_run(tmp_path / 'no_run') is None and warned
        other = viewer.open_run(recorded[1])
        assert other is not None and other.isVisible()
        viewer.quit()
        qapp.processEvents()
        assert not viewer.isVisible() and not other.isVisible()

    def test_the_preferences_change_the_look_of_the_viewer(self, viewer, monkeypatch) -> None:
        from spark.graph_editor.widgets.preferences_dialog import PreferencesDialog
        from spark.graph_editor.runs.viewer import PREFERENCES
        from spark.graph_editor.styles.run_viewer import THEME
        from spark.graph_editor.styles.manager import STYLES
        dialog = PreferencesDialog(viewer, sections=PREFERENCES, library=False)
        assert [dialog.sidebar.item(i).text() for i in range(dialog.sidebar.count())] == list(PREFERENCES)
        dialog.close()
        monkeypatch.setitem(STYLES._config, 'run_viewer', {**STYLES._config.get('run_viewer', {}), 'area_color': '#102030'})
        STYLES.reloaded.emit()
        assert THEME.area.name() == '#102030'
        monkeypatch.undo()
        THEME.read()

    def test_a_style_saved_before_a_setting_takes_its_default(self, viewer, tmp_path) -> None:
        import json
        from PySide6.QtCore import QSettings
        from spark.graph_editor.styles.manager import STYLES
        from spark.graph_editor.styles.run_viewer import THEME
        chosen = json.loads(json.dumps(STYLES.defaults()))
        chosen['run_viewer'] = {'area_color': '#102030'}                # none of its other settings
        del chosen['run_viewer_timeline']
        path = tmp_path / 'style.json'
        path.write_text(json.dumps(chosen))
        QSettings().setValue('style_config_path', str(path))
        try:
            STYLES.reload()
            assert THEME.area.name() == '#102030' and THEME.text.name() == STYLES.defaults()['run_viewer']['text_color']
            assert THEME.timeline.row_height == STYLES.defaults()['run_viewer_timeline']['row_height']
        finally:
            QSettings().remove('style_config_path')
            STYLES.reload()
        assert THEME.area.name() == STYLES.defaults()['run_viewer']['area_color']

    def test_the_status_of_a_run_takes_its_colour_from_the_style(self, viewer, qapp) -> None:
        from PySide6.QtGui import QPalette
        from spark.graph_editor.styles.manager import STYLES
        status = viewer.run_panel._status
        viewer.show()
        qapp.processEvents()
        assert status.property('status') == 'finished'
        assert status.palette().color(QPalette.ColorRole.WindowText).name() == STYLES.get_val('run_viewer_status', 'finished_color')

    def test_a_finished_run_cannot_be_recorded(self, viewer) -> None:
        assert viewer.run_panel.record_rows and not any(row.button.isEnabled() for row in viewer.run_panel.record_rows.values())

    def test_the_window_paints(self, viewer, qapp) -> None:
        _select(viewer, qapp, 'second_pool')
        viewer.set_cursor(25)
        qapp.processEvents()
        image = viewer.grab().toImage()
        assert image.width() > 0 and image.height() > 0

    def test_a_closed_window_drops_what_it_read(self, qapp, recorded) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        window = RunViewerWindow(recorded[1])
        window.show()
        window.set_cursor(25)
        assert window.data.scalars and window.data._cache
        window.close()
        assert not window.data.scalars and not window.data._cache and not window._timer.isActive()

    def test_closed_windows_are_freed(self, qapp, recorded) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        freed = []
        for _ in range(3):
            window = RunViewerWindow(recorded[1])
            window.show()
            qapp.processEvents()
            freed.append(weakref.ref(window))
            window.close()
            qapp.processEvents()
            del window
        last = RunViewerWindow(recorded[1])
        last.show()                                                     # drops the windows closed before
        gc.collect()
        qapp.processEvents()
        assert [ref() for ref in freed] == [None] * 3
        last.close()

    def test_a_run_that_cannot_be_read_any_more_is_reported(self, qapp, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        path = _copy(recorded, tmp_path)
        viewer = RunViewerWindow(path)
        viewer.show()
        moved = path.with_name('moved')
        path.rename(moved)
        viewer.follow.setChecked(True)
        viewer.refresh()
        assert 'Reading the run failed' in viewer.statusBar().currentMessage()
        moved.rename(path)
        viewer.refresh()
        assert 'failed' not in viewer.statusBar().currentMessage()
        viewer.close()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestLive:

    def test_a_running_run_is_followed_and_recorded_from_the_viewer(self, qapp, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        from spark.graph_editor.runs.plots import SeriesPlot
        brain = _brain()
        probe = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),)
        measurements = [R.Measurements('summary', probe, group=5, trigger=R.Always()), R.Measurements('manual', probe, trigger=R.Manual(), group=5)]
        recorder = R.Recorder(tmp_path, brain, measurements)
        runner = R.Runner(brain, recorder)
        runner.run(5, {'signal': SIGNAL})
        recorder.flush()
        viewer = RunViewerWindow(recorder.path, refresh_every=100000)
        viewer.show()
        qapp.processEvents()
        assert viewer.data.status == 'running' and viewer.follow.isChecked()
        assert _plots(viewer.probe_panel, SeriesPlot) == []
        row = viewer.run_panel.record_rows['manual']
        assert row.button.isEnabled()
        row.steps.setValue(10)
        row.button.click()
        assert 'pending' in row.note.text()
        _wait_for(lambda: viewer.data.run.requests()[0]['status'] == 'received')
        for _ in range(4):
            runner.run(5, {'signal': SIGNAL})
        recorder.log(reward=1.0)
        recorder.flush()
        viewer.refresh()
        assert viewer.timeline.total == 25 and viewer.cursor == 25
        assert 'applied' in row.note.text()
        assert viewer.data.segments('manual') == [(5, 15)]
        # Logged scalars that appear while the run is written are shown, and given a panel in the workspace.
        assert [p.title for p in _plots(viewer.probe_panel, SeriesPlot)] == ['reward']
        assert 'reward' in viewer.workspace.sections['logged'].keys
        runner.close()
        viewer.refresh()
        assert viewer.data.status == 'finished'
        viewer.close()
        qapp.processEvents()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestCompare:

    @staticmethod
    def _runs(recorded, tmp_path, *hparams, experiments=None):
        paths = []
        for index, params in enumerate(hparams):
            path = tmp_path / f'20260101-00000{index}_{recorded[1].name.split("_", 1)[1]}'
            shutil.copytree(recorded[1], path)
            R.store.write_json(path / 'hparams.json', params)
            if experiments is not None:
                R.store.write_json(path / 'run.json', {**json.loads((path / 'run.json').read_text()), 'experiment': experiments[index]})
            paths.append(path)
        return paths

    def test_a_directory_opens_on_its_workspace(self, qapp, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        paths = self._runs(recorded, tmp_path, {'lr': 0.1}, {'lr': 0.5})
        viewer = RunViewerWindow(tmp_path)
        viewer.show()
        qapp.processEvents()
        assert [viewer.center.tabText(i) for i in range(viewer.center.count())] == ['Workspace', 'Graph']
        assert viewer.center.currentWidget() is viewer.workspace
        # The newest run is shown on the Graph tab; every run is drawn in the workspace.
        assert viewer.data.path == paths[-1]
        assert viewer.selection.visible == {str(path) for path in paths}
        assert viewer.runs_table.tree.topLevelItemCount() == 2
        (reward,) = viewer.workspace.sections['logged'].panels.values()
        assert [s[0] for s in reward.plot.series] == [viewer.selection.label(run) for run in viewer.project.runs]
        viewer.close()
        qapp.processEvents()

    def test_a_directory_without_runs_is_refused(self, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        with pytest.raises(FileNotFoundError):
            RunViewerWindow(tmp_path)

    def test_runs_shown_are_drawn_in_the_probe_panel(self, qapp, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.plots import SeriesPlot
        from spark.graph_editor.runs.viewer import RunViewerWindow
        first, second = self._runs(recorded, tmp_path, {'lr': 0.1}, {'lr': 0.5})
        viewer = RunViewerWindow(first)
        # A run opened on its own is compared with nothing.
        assert viewer.selection.visible == {str(first)}
        assert viewer.center.tabText(viewer.center.currentIndex()) == 'Graph'
        _select(viewer, qapp, 'first_pool')
        rate = [p for p in _plots(viewer.probe_panel, SeriesPlot) if p.title.endswith('active_fraction')][0]
        assert [s[0] for s in rate.series] == ['active_fraction']
        viewer.selection.set_visible([second], True)
        labels = [viewer.selection.label(run) for run in viewer.project.runs]
        assert [s[0] for s in rate.series] == labels
        assert rate.series[1][3] == viewer.selection.color(str(second))
        spread = [p for p in _plots(viewer.probe_panel, SeriesPlot) if p.title.startswith('potential')][0]
        assert [s[0] for s in spread.series] == labels
        assert labels[1] in viewer.probe_panel._legend.text()
        # Hidden, the run shown is drawn still, and named so.
        viewer.selection.set_visible([first], False)
        assert [s[0] for s in rate.series] == [f'{labels[0]} · shown', labels[1]]
        viewer.selection.show_only([first])
        assert [s[0] for s in rate.series] == ['active_fraction']
        viewer.close()
        qapp.processEvents()

    def test_a_run_shown_alone_is_drawn_in_its_color(self, qapp, recorded, tmp_path) -> None:
        from PySide6.QtGui import QColor
        from spark.graph_editor.runs.plots import SeriesPlot, BarPlot
        from spark.graph_editor.runs.viewer import RunViewerWindow
        (first,) = self._runs(recorded, tmp_path, {'lr': 0.1})
        viewer = RunViewerWindow(first)
        viewer.selection.set_color(str(first), QColor('#d88fbf'))
        _select(viewer, qapp, 'first_pool')
        plots = _plots(viewer.probe_panel, SeriesPlot)
        rate = [p for p in plots if p.title.endswith('active_fraction')][0]
        spread = [p for p in plots if p.title.startswith('potential')][0]
        assert [s[3].name() for s in rate.series] == ['#d88fbf'] and [s[3].name() for s in spread.series] == ['#d88fbf']
        assert {band[4].rgb() for band in spread.bands} == {QColor('#d88fbf').rgb()}
        bars = BarPlot('bars')
        bars.set_values(np.arange(3.0), color=QColor('#d88fbf'))
        assert bars.color.name() == '#d88fbf'
        viewer.close()
        qapp.processEvents()

    def test_runs_of_an_experiment_are_drawn_as_one(self, qapp, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.plots import SeriesPlot
        from spark.graph_editor.runs.viewer import RunViewerWindow
        paths = self._runs(recorded, tmp_path, {'seed': 1}, {'seed': 2}, {'seed': 1}, {'seed': 2}, experiments=('X', 'X', 'Y', 'Y'))
        viewer = RunViewerWindow(tmp_path)
        assert viewer.selection.group_by == ('experiment',)
        tree = viewer.runs_table.tree
        assert sorted(tree.topLevelItem(i).text(0) for i in range(2)) == ['X  (2)', 'Y  (2)']
        assert [line.label for line in viewer.selection.lines()] == ['X · 2 runs', 'Y · 2 runs']
        (reward,) = viewer.workspace.sections['logged'].panels.values()
        # The copies of one run: a mean equal to the run, and no spread.
        assert [s[0] for s in reward.plot.series] == ['X · 2 runs', 'Y · 2 runs'] and len(reward.plot.bands) == 2
        _, x, mean, _ = reward.plot.series[0]
        steps, own = viewer.data.scalar('reward')
        np.testing.assert_allclose(mean, own)
        _, _, low, high, _ = reward.plot.bands[0]
        np.testing.assert_allclose(low, mean)
        np.testing.assert_allclose(high, mean)
        viewer.show_run(paths[0])
        _select(viewer, qapp, 'first_pool')
        rate = [p for p in _plots(viewer.probe_panel, SeriesPlot) if p.title.endswith('active_fraction')][0]
        assert [s[0] for s in rate.series] == ['X · 2 runs', 'Y · 2 runs'] and len(rate.bands) == 2
        assert not rate.faint
        # The runs of groups, drawn faintly.
        viewer.selection.set_settings(members=True)
        assert len(rate.faint) == 4 and len(reward.plot.faint) == 4
        # A run hidden leaves its group.
        viewer.selection.set_visible([paths[3]], False)
        assert [s[0] for s in rate.series] == ['X · 2 runs', 'Y · 1 run'] and len(rate.bands) == 1
        viewer.close()
        qapp.processEvents()

    def test_another_run_is_shown_on_a_double_click(self, qapp, recorded, tmp_path, monkeypatch) -> None:
        from PySide6.QtWidgets import QMessageBox
        from spark.graph_editor.runs.viewer import RunViewerWindow
        first, second = self._runs(recorded, tmp_path, {'lr': 0.1}, {'lr': 0.5})
        viewer = RunViewerWindow(tmp_path)
        viewer.show()
        qapp.processEvents()
        assert viewer.data.path == second
        old_data, old_panel = viewer.data, viewer.probe_panel
        item = next(viewer.runs_table.tree.topLevelItem(i) for i in range(2)
                    if viewer.runs_table.tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) == str(first))
        viewer.runs_table.tree.itemDoubleClicked.emit(item, 0)
        qapp.processEvents()
        assert viewer.data.path == first and not old_data.scalars
        assert viewer.probe_panel is not old_panel and viewer.run_panel.data is viewer.data
        assert viewer.center.tabText(viewer.center.currentIndex()) == 'Graph' and viewer.scene is not None
        assert viewer.windowTitle().endswith(first.name)
        # Its selection moves the cursor of its own plots.
        _select(viewer, qapp, 'first_pool')
        viewer.set_cursor(25)
        assert viewer.probe_panel.cursor == 25
        warned = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warned.append(args))
        viewer.show_run(tmp_path / 'no_run')
        assert warned and viewer.data.path == first
        viewer.close()
        qapp.processEvents()

    def test_the_timeline_shows_the_values_of_tags(self, qapp, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        viewer = RunViewerWindow(_copy(recorded, tmp_path))
        (name, steps, values), = viewer.timeline.tags
        assert name == 'episode' and list(values) == [0, 1, 2] and list(steps) == [0, 20, 40]
        viewer.close()
        qapp.processEvents()

    def test_runs_added_to_the_directory_are_listed(self, qapp, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        (first,) = self._runs(recorded, tmp_path, {'lr': 0.1})
        viewer = RunViewerWindow(tmp_path)
        viewer.show()
        qapp.processEvents()
        added = tmp_path / f'20260101-000009_{recorded[1].name.split("_", 1)[1]}'
        shutil.copytree(recorded[1], added)
        viewer._refresh_runs()
        qapp.processEvents()
        assert [run.path for run in viewer.project.runs] == [first, added]
        assert viewer.runs_table.tree.topLevelItemCount() == 2 and str(added) in viewer.selection.visible
        viewer.close()
        qapp.processEvents()

    def test_runs_that_cannot_be_read_are_left_out(self, qapp, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        first, second, third = self._runs(recorded, tmp_path, {'lr': 0.1}, {'lr': 0.5}, {'lr': 0.9})
        viewer = RunViewerWindow(first)
        shutil.rmtree(second)                                           # deleted while the viewer is open
        (third / 'index.sqlite').write_bytes(b'not a database')
        viewer.selection.show_only([first, second, third])
        viewer.refresh()
        qapp.processEvents()
        (reward,) = viewer.workspace.sections['logged'].panels.values()
        assert [len(x) for _, x, _, _ in reward.plot.series] == [6]
        viewer.close()
        qapp.processEvents()

    def test_the_exploration_is_kept_and_resumed(self, qapp, recorded, tmp_path, _explorations) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        from spark.graph_editor.runs.workspace import Exploration
        paths = self._runs(recorded, tmp_path, {'seed': 1}, {'seed': 2}, {'seed': 3})
        viewer = RunViewerWindow(tmp_path)
        viewer.show()
        qapp.processEvents()
        viewer.runs_table.set_experiment([paths[0], paths[1]], 'X')
        viewer.selection.set_group_by(['experiment'])
        viewer.selection.set_visible([paths[2]], False)
        viewer.selection.set_settings(smoothing=0.3)
        viewer.runs_table.set_columns(['seed'])
        viewer.workspace.set_columns(1)
        pinned = viewer.workspace.add_panel(['reward'])
        pinned.set_zoom([10.0, 40.0], [0.0, 3.0])
        viewer.show_run(paths[0])
        _select(viewer, qapp, 'second_pool')
        viewer.set_cursor(35)
        viewer.close()
        qapp.processEvents()
        kept = Exploration.of(tmp_path)
        assert kept.path.parent == _explorations and kept.read()['shown'] == paths[0].name
        # Nothing is written within the directory of runs.
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(p.name for p in paths)
        again = RunViewerWindow(tmp_path)
        assert again.data.path == paths[0] and again.cursor == 35 and again.probe_panel.node == 'second_pool'
        assert again.center.tabText(again.center.currentIndex()) == 'Graph'
        assert again.project.experiment(again.project.run(paths[1])) == 'X' and again.project.run(paths[1]).experiment is None
        assert again.selection.group_by == ('experiment',) and again.selection.smoothing == 0.3
        assert again.selection.visible == {str(paths[0]), str(paths[1])}
        assert again.runs_table.columns == ['seed'] and again.workspace.columns == 1
        assert again.workspace.pinned.panels['reward'].zoom() == {'x': [10.0, 40.0], 'y': [0.0, 3.0]}
        # A run opened on its own is shown, in the exploration of its directory.
        again.close()
        alone = RunViewerWindow(paths[2])
        assert alone.data.path == paths[2] and alone.selection.group_by == ('experiment',)
        alone.close()
        qapp.processEvents()

    def test_the_panels_of_the_graph_are_collapsed_on_the_workspace(self, qapp, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        first, _ = self._runs(recorded, tmp_path, {'lr': 0.1}, {'lr': 0.5})
        viewer = RunViewerWindow(tmp_path)
        viewer.show()
        qapp.processEvents()
        docks = (viewer.dock_probes, viewer.dock_timeline)
        hidden = lambda: [dock.isHidden() for dock in docks]
        graph, workspace = viewer._graph_tab(), viewer.center.indexOf(viewer.workspace)
        assert viewer.center.currentIndex() == workspace and hidden() == [True, True]
        # The left panels reach the bottom of the window whether the timeline is shown or not; the timeline reaches its right.
        assert viewer.corner(Qt.Corner.BottomLeftCorner) == Qt.DockWidgetArea.LeftDockWidgetArea
        assert viewer.corner(Qt.Corner.BottomRightCorner) == Qt.DockWidgetArea.BottomDockWidgetArea
        viewer.show_run(first)
        assert viewer.center.currentIndex() == graph and hidden() == [False, False]
        # On the Graph tab, the Probes panel can take all but a sliver of the graph.
        viewer.resize(1700, 900)
        viewer.resizeDocks([viewer.dock_runs], [300], Qt.Orientation.Horizontal)
        viewer.resizeDocks([viewer.dock_probes], [1250], Qt.Orientation.Horizontal)
        qapp.processEvents()
        assert viewer.dock_probes.width() > 1200 and viewer.center.width() < 200
        # Closed on the Graph tab, a panel stays closed there.
        viewer.dock_timeline.close()
        viewer.center.setCurrentIndex(workspace)
        viewer.center.setCurrentIndex(graph)
        assert hidden() == [False, True]
        viewer.dock_probes.close()
        viewer.dock_timeline.show()
        viewer.center.setCurrentIndex(workspace)
        assert hidden() == [True, True]
        viewer.close()
        # Resumed on the Workspace, they open on the Graph tab as they were left there.
        again = RunViewerWindow(tmp_path)
        again.show()
        qapp.processEvents()
        assert again.center.currentWidget() is again.workspace and [d.isHidden() for d in (again.dock_probes, again.dock_timeline)] == [True, True]
        again.center.setCurrentIndex(again._graph_tab())
        # A layout kept with other corners, as by an earlier viewer, resumes with the corners of the window.
        again.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        again.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        state = again.state()
        again.resume(state)
        assert again.corner(Qt.Corner.BottomLeftCorner) == Qt.DockWidgetArea.LeftDockWidgetArea
        assert again.corner(Qt.Corner.BottomRightCorner) == Qt.DockWidgetArea.BottomDockWidgetArea
        again.center.setCurrentIndex(again.center.indexOf(again.workspace))
        again.center.setCurrentIndex(again._graph_tab())
        assert [d.isHidden() for d in (again.dock_probes, again.dock_timeline)] == [True, False]
        again.close()
        qapp.processEvents()

    def test_runs_added_since_the_exploration_are_shown(self, qapp, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        first, second = self._runs(recorded, tmp_path, {'seed': 1}, {'seed': 2})
        viewer = RunViewerWindow(tmp_path)
        viewer.show()
        viewer.selection.show_only([first])
        viewer.close()
        added = tmp_path / f'20260101-000009_{recorded[1].name.split("_", 1)[1]}'
        shutil.copytree(recorded[1], added)
        shutil.rmtree(first)
        again = RunViewerWindow(tmp_path)
        assert again.selection.visible == {str(added)}
        again.close()
        qapp.processEvents()

    def test_an_exploration_is_saved_and_opened_by_hand(self, qapp, recorded, tmp_path, monkeypatch) -> None:
        from PySide6.QtWidgets import QMessageBox
        from spark.graph_editor.runs.viewer import RunViewerWindow
        runs = tmp_path / 'runs'
        runs.mkdir()
        first, second = self._runs(recorded, runs, {'seed': 1}, {'seed': 2})
        warned = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warned.append(args[-1]))
        viewer = RunViewerWindow(runs)
        viewer.selection.show_only([second])
        # Never within a directory of runs, nor within a run.
        for refused in (runs / 'view', first / 'view', runs / 'notes' / 'view'):
            assert not viewer.save_exploration(refused)
        assert len(warned) == 3 and 'directory of runs' in warned[0]
        assert not list(runs.rglob('*.exploration.json'))
        assert viewer.save_exploration(tmp_path / 'view')
        saved = tmp_path / 'view.exploration.json'
        assert saved.exists()
        viewer.selection.show_only([first, second])
        assert viewer.open_exploration(saved) is viewer and viewer.selection.visible == {str(second)}
        assert viewer.open_exploration(tmp_path / 'missing.exploration.json') is None and len(warned) == 4
        viewer.close()
        other = RunViewerWindow(first)
        window = other.open_exploration(saved)
        assert window is other and other.selection.visible == {str(second)}
        other.close()
        qapp.processEvents()

    @pytest.mark.skipif(sys.platform == 'win32' or os.geteuid() == 0, reason='permissions of POSIX, not enforced for root')
    def test_a_read_only_directory_with_a_crashed_run_opens(self, qapp, recorded, tmp_path) -> None:
        from spark.graph_editor.runs.viewer import RunViewerWindow
        first, crashed = self._runs(recorded, tmp_path, {'lr': 0.1}, {'lr': 0.5})
        info = json.loads((crashed / 'run.json').read_text())
        R.store.write_json(crashed / 'run.json', {**info, 'status': 'running', 'heartbeat': '2000-01-01T00:00:00+00:00'})
        items = [tmp_path, *tmp_path.rglob('*')]
        modes = {item: item.stat().st_mode for item in items}
        try:
            for item in items:
                item.chmod(0o555 if item.is_dir() else 0o444)
            viewer = RunViewerWindow(first)
            tree = viewer.runs_table.tree
            statuses = {tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole): tree.topLevelItem(i).toolTip(3) for i in range(2)}
            assert statuses[str(crashed.resolve())] == 'crashed' and viewer.data.status == 'finished'
            viewer.set_cursor(25)
            viewer.close()
        finally:
            for item in reversed(items):
                item.chmod(modes[item])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestOpening:

    def test_a_run_opens_from_the_editor(self, editor, recorded, qapp) -> None:
        viewer = editor.open_run(recorded[1])
        assert viewer is not None and viewer.data.path == recorded[1]
        viewer.close()
        qapp.processEvents()

    def test_a_directory_without_a_run_is_reported(self, editor, answers, tmp_path) -> None:
        assert editor.open_run(tmp_path) is None

    def test_a_run_opens_from_a_script(self, qapp, recorded) -> None:
        from spark.graph_editor.runs.viewer import SparkRunViewer
        assert spark.RunViewer is SparkRunViewer
        viewer = SparkRunViewer()
        viewer._is_interactive = True                                   # returns at once instead of running the event loop
        window = viewer.open(recorded[1])
        assert viewer.windows == [window] and window.isVisible()
        window.close()
        assert viewer.open(recorded[1]) is not window and len(viewer.windows) == 1
        viewer.windows[0].close()
        qapp.processEvents()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestPlots:

    def test_a_long_series_draws_in_proportion_to_the_width(self, qapp) -> None:
        from spark.graph_editor.runs.plots import SeriesPlot
        plot = SeriesPlot('long')
        plot.resize(800, 150)
        x = np.arange(2_000_000)
        plot.set_series([('a', x, np.sin(x / 1000.0)), ('b', x, np.cos(x / 777.0))])
        rect = plot.plot_rect()
        shapes = plot.shapes(rect)
        assert len(shapes) == 2 and all(shape.elementCount() <= 4 * rect.width() for shape, _ in shapes)
        plot.grab()

    def test_points_outside_the_range_are_not_drawn(self, qapp) -> None:
        from spark.graph_editor.runs.plots import SeriesPlot
        plot = SeriesPlot('clipped')
        plot.resize(300, 150)
        x = np.arange(10_000, dtype=np.float64)
        y = np.where(x < 5_000, 0.0, 1e6)
        plot.set_series([('a', x, y)], x_range=(0.0, 5_000.0), y_range=(-1.0, 1.0))
        rect = plot.plot_rect()
        (path, _), = plot.shapes(rect)
        assert path.boundingRect().height() < 2

    def test_tick_labels_tell_ticks_apart(self) -> None:
        from spark.graph_editor.runs.plots import nice_ticks, tick_labels
        for lo, hi in ((12_000, 12_400), (1_600_000, 1_650_000), (0, 1), (0, 12.5), (1e-5, 2e-5), (1e10, 1.4e10), (-3, 3),
                       (1_000_000, 1_000_010), (5, 5 + 1e-7)):
            labels = tick_labels(nice_ticks(lo, hi, count=5))
            assert len(set(labels)) == len(labels) and max(map(len, labels)) <= 12, (lo, hi, labels)
        assert tick_labels([12_000.0, 12_250.0, 12_500.0]) == ['12,000', '12,250', '12,500']

    def test_empty_plots_say_so(self, qapp) -> None:
        from spark.graph_editor.runs.plots import MatrixPlot, BarPlot, ImagePlot
        matrix = MatrixPlot('m')
        matrix.set_matrix(np.zeros(0))
        assert matrix.message and matrix._image is None and not matrix.subtitle
        matrix.set_matrix(np.float32(3.0))
        assert not matrix.message and matrix._image is not None
        bars = BarPlot('b')
        bars.set_values(np.zeros(0))
        assert bars.message
        image = ImagePlot('i')
        image.set_image(np.zeros(0), np.zeros(0))
        assert image.message and image._image is None

    def test_row_0_of_a_matrix_is_at_the_bottom(self, qapp) -> None:
        from spark.graph_editor.runs.plots import MatrixPlot
        plot = MatrixPlot('m')
        matrix = np.zeros((10, 10))
        matrix[0] = 1.0
        plot.set_matrix(matrix)
        assert plot._image.pixelColor(0, 9) != plot._image.pixelColor(0, 0)
        assert plot._image.pixelColor(0, 9) == plot._image.pixelColor(5, 9)

    def test_an_image_of_a_raster(self, qapp) -> None:
        from spark.graph_editor.runs.plots import ImagePlot
        plot = ImagePlot('raster')
        plot.resize(600, 150)
        bits = np.random.default_rng(0).random((5000, 256)) < 0.05
        plot.set_image(np.arange(5000), bits)
        # Pairs of steps are drawn as one column, spiking when either step spikes.
        assert plot._image is not None and plot._image.width() == 2500 and plot._image.height() == 256
        assert plot.x_range == (0.0, 5000.0) and plot.y_range == (0.0, 256.0)
        plot.grab()

    def test_many_bars_draw_one_bar_per_column(self, qapp) -> None:
        from spark.graph_editor.runs.plots import BarPlot
        class Painter:
            def __init__(self):
                self.rects = 0
            def drawRect(self, *args):
                self.rects += 1
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        plot = BarPlot('bars')
        plot.resize(600, 120)
        plot.set_values(np.random.default_rng(0).random(1_000_000))
        painter, rect = Painter(), plot.plot_rect()
        plot.draw(painter, rect)
        assert 0 < painter.rects <= rect.width()
        plot.grab()

    def test_a_box_zooms_and_a_double_click_fits(self, qapp) -> None:
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QMouseEvent
        from spark.graph_editor.runs.plots import SeriesPlot
        plot = SeriesPlot('zoom')
        plot.resize(300, 150)
        plot.set_series([('a', np.arange(100), np.linspace(0.0, 100.0, 100))])
        rect = plot.plot_rect()
        press = lambda kind, x, y: QMouseEvent(kind, QPointF(x, y), QPointF(x, y), Qt.MouseButton.RightButton,
                                               Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
        plot.mousePressEvent(press(QEvent.Type.MouseButtonPress, rect.left() + 10, rect.top() + 10))
        plot.mouseMoveEvent(press(QEvent.Type.MouseMove, rect.center().x(), rect.center().y()))
        plot.grab()                                                     # draws the box
        plot.mouseReleaseEvent(press(QEvent.Type.MouseButtonRelease, rect.center().x(), rect.center().y()))
        (x0, x1), (y0, y1) = plot.x_range, plot.y_range
        assert 0 < x0 < x1 < 60 and 40 < y0 < y1 < 100 and 'zoomed' in plot.title_text()
        # A double click fits the values of the steps shown.
        plot.mouseDoubleClickEvent(None)
        assert plot.zoomed is None and plot.x_range == (x0, x1) and plot.y_range[0] < y0
        plot.show_every_step()
        assert plot.x_range == (0.0, 99.0)
        plot.zoom(0.5)
        assert plot.zoomed is not None

    def test_images_of_numbers_have_a_colour_bar(self, qapp) -> None:
        from spark.graph_editor.runs.plots import MatrixPlot, ImagePlot, SeriesPlot
        from spark.graph_editor.styles.run_viewer import THEME
        matrix = MatrixPlot('m')
        matrix.set_matrix(np.arange(6.0).reshape(2, 3))
        trace = ImagePlot('t')
        trace.set_image(np.arange(4), np.arange(8.0).reshape(4, 2))
        raster = ImagePlot('r')
        raster.set_image(np.arange(4), np.eye(4, dtype=bool))
        assert matrix.colorbar == (0.0, 5.0) and trace.colorbar == (0.0, 7.0) and raster.colorbar is None
        # A raster: events on the background of the plots.
        assert raster._image.pixelColor(0, 3) == THEME.raster_event and raster._image.pixelColor(1, 3) == THEME.area
        series = SeriesPlot('s')
        series.set_series([('a', np.arange(4), np.arange(4.0))])
        for plot in (matrix, trace, raster, series):
            plot.resize(300, 150)
            assert not plot.grab().isNull()
        # The colour bar is in the title line: the steps of every plot line up.
        assert trace.plot_rect() == series.plot_rect() == raster.plot_rect()

    def test_values_past_the_float_range_paint(self, qapp) -> None:
        from spark.graph_editor.runs.plots import BarPlot, SeriesPlot, nice_ticks
        assert nice_ticks(-1e308, 1e308) == [-1e308] and nice_ticks(float('nan'), 1.0) == []
        bars = BarPlot('extreme')
        bars.resize(200, 100)
        bars.set_values(np.array([-1e308, 1e308, np.inf, np.nan]))
        series = SeriesPlot('extreme')
        series.resize(200, 100)
        series.set_series([('a', np.arange(3), np.array([-1e308, 1e308, np.nan]))])
        for plot in (bars, series):
            assert not plot.grab().isNull()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
