#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import os
import sys
import json
import time
import signal
import socket
import sqlite3
import textwrap
import warnings
import threading
import subprocess
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import spark
from cases import CASES, STEPS

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

R = spark.recording
RN = spark.recording.runner
SIGNAL = np.full((8,), 1.0, dtype=np.float16)
HERE = os.path.dirname(os.path.abspath(__file__))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

_WORKER = """
import os, sys, json, time, signal, pathlib
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_num_cpu_devices', 4)
mode, index, port, root = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), pathlib.Path(sys.argv[4])
jax.distributed.initialize(f'localhost:{port}', num_processes=2, process_id=index, initialization_timeout=120)
import flax.nnx as nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import spark
R = spark.recording
R.SETTINGS.open_timeout = 20.0
# A mesh in another order than the ids of its devices, as the meshes of clusters often are.
mesh = Mesh(np.array(jax.devices()).reshape(2, 4).T.reshape(-1), ('x',))
neuron = spark.nn.neurons.LIFNeuron(units=(64,), seed=5)
neuron(in_spikes=spark.SpikeArray((jax.random.uniform(jax.random.key(0), (32,)) < 0.3).astype(jnp.uint8)))
def shard(leaf):
    host = np.asarray(leaf)
    sharding = NamedSharding(mesh, P('x') if host.ndim and host.shape[0] % 8 == 0 else P())
    return jax.make_array_from_callback(host.shape, sharding, lambda i: host[i])
nnx.update(neuron, jax.tree.map(shard, nnx.state(neuron)))
measurements = [
    R.Measurements('weights', R.presets.weights(neuron), trigger=R.Always(), group=8),
    R.Measurements('summary', R.presets.summary(neuron), trigger=R.Every(2, tag='episode'), group='episode'),
    R.Measurements('manual', R.presets.activity(neuron), trigger=R.Manual()),
]
options = {'flush_steps': 1} if mode == 'fail' else {}
if mode == 'fail' and index == 0:
    def full(path, arrays):
        raise OSError(28, 'No space left on device')
    R.store.write_arrays = full
try:
    recorder = R.Recorder(root, neuron, measurements, signals=(signal.SIGUSR1, signal.SIGTERM), **options) if mode != 'alone' or index == 0 else None
except RuntimeError as error:
    print(json.dumps({'raised': type(error).__name__, 'message': str(error), 'chunk': 0, 'path': None}), flush=True)
    raise
if recorder is None:
    print(json.dumps({'raised': None, 'chunk': 0, 'path': None}), flush=True)
    sys.exit(0)
runner = R.Runner(neuron, recorder)
per_step = {'in_spikes': (np.random.default_rng(1).random((8, 32)) < 0.3).astype(np.uint8)}
points = []
try:
    for chunk in range(8):
        recorder.tag(episode=chunk if index == 0 else 0)                # the tags of process 0 count
        if chunk == 2:
            recorder.record('manual')
        if mode == 'preempt' and chunk == 4 and index == 0:
            os.kill(os.getpid(), signal.SIGUSR1)
        if mode in ('signal1', 'sigterm1', 'sigterm_user') and chunk == 3 and index == 1:
            os.kill(os.getpid(), signal.SIGUSR1 if mode == 'signal1' else signal.SIGTERM)
        if mode in ('signal1', 'sigterm1', 'sigterm_user'):
            time.sleep(0.4)                                             # the signal reaches process 0 meanwhile
        if mode == 'sigterm_user':
            # The sync point of the preemption service used by other code, as by the checkpoint managers of orbax.
            from jax.experimental import multihost_utils
            points.append(bool(multihost_utils.reached_preemption_sync_point(1000 + chunk)))
        runner.run(8, per_step=per_step)
        if chunk == 5:
            runner.checkpoint()
    runner.close()
except BaseException as error:
    print(json.dumps({'raised': type(error).__name__, 'chunk': recorder._calls, 'path': str(recorder.path), 'points': points}), flush=True)
    raise
print(json.dumps({'raised': None, 'chunk': recorder._calls, 'path': str(recorder.path), 'points': points}), flush=True)
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture
def brain():
    model, _, _ = CASES['brain']()
    return model

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture(autouse=True)
def _close_recorders():
    """
        Closes the recorders a test left open, whatever its outcome.
    """
    yield
    for recorder in list(spark.recording.recorder._OPEN):
        try:
            recorder.close()
        except Exception:
            pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _recorded_t0(run, measurements):
    return [int(t) for t in run.timeline(measurements)['span_t0']] if run.windows(measurements) else []

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _python(script, returncode=0):
    """
        Runs ``script`` in a new interpreter, on the processor, and returns the last line it printed.
    """
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(script)], capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == returncode, result.stderr
    return result.stdout.strip().splitlines()[-1]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _wait_for(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, 'timed out'
        time.sleep(0.05)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestTriggers:

    def _pattern(self, trigger, count=12):
        return [c for c in range(count) if trigger.recorded({'step': c})]

    def test_every(self) -> None:
        assert self._pattern(R.Every(3)) == [0, 3, 6, 9]
        assert self._pattern(R.Every(5, length=2, offset=1)) == [1, 2, 6, 7, 11]

    def test_at_and_between(self) -> None:
        assert self._pattern(R.At((2, 7), length=2)) == [2, 3, 7, 8]
        assert self._pattern(R.Between(4, 6)) == [4, 5]
        assert self._pattern(R.Between(10)) == [10, 11]
        with pytest.raises(ValueError, match='after'):
            R.Between(5, 5)

    def test_a_call_is_recorded_when_any_step_it_covers_is(self) -> None:
        calls = lambda trigger, steps=64, count=200: [c * steps for c in range(count) if trigger.recorded({'step': c * steps}, {'step': steps})]
        assert calls(R.At((5000,))) == [4992]
        assert calls(R.At((5000,), length=100)) == [4992, 5056]
        assert calls(R.Between(1000, 1100)) == [960, 1024, 1088]
        assert calls(R.Every(5000, length=10, offset=30)) == [0, 4992, 9984]
        # Without the span, the first step of the call decides.
        assert not R.At((5000,)).recorded({'step': 4992})

    def test_always_and_manual(self) -> None:
        assert self._pattern(R.Always()) == list(range(12))
        assert self._pattern(R.Manual()) == []

    def test_a_tag(self) -> None:
        trigger = R.Every(2, tag='episode')
        assert trigger.recorded({'step': 5, 'episode': 4}) and not trigger.recorded({'step': 4, 'episode': 3})
        assert not trigger.recorded({'step': 0})

    def test_steps_are_counted_without_a_tag(self) -> None:
        assert R.Every(2).tag is None
        for tag in ('step', '', 3):
            with pytest.raises(ValueError, match='tag'):
                R.Every(2, tag=tag)

    def test_counts_are_integers(self) -> None:
        for make in (lambda: R.Every(2.5), lambda: R.Every(2, offset=0.5), lambda: R.At((1.5,)), lambda: R.Between(0, 2.5),
                     lambda: R.When(bool, watch='a', length=1.5), lambda: R.Every(True), lambda: R.Every(float('inf')),
                     lambda: R.Every(float('nan'))):
            with pytest.raises(ValueError, match='integer'):
                make()
        assert R.Every(np.int64(4)).n == 4 and type(R.At((np.int32(3),)).points[0]) is int
        # Whole floats, as step counts are often written.
        assert R.Every(1e5) == R.Every(100_000) and type(R.Every(1e5).n) is int
        assert R.Between(0, 2.0).stop == 2 and R.Every(np.float32(4.0), length=2.0).length == 2

    def test_round_trip(self) -> None:
        for trigger in (R.Every(3, length=2, tag='episode', offset=1), R.At((1, 5)), R.Between(2, 9), R.Always(), R.Manual()):
            assert R.Trigger.from_dict(json.loads(json.dumps(trigger.to_dict()))) == trigger
            assert type(trigger).from_dict(trigger.to_dict()) == trigger

    def test_a_trigger_of_its_own_comes_back_manual(self) -> None:
        class Odd(R.Trigger):
            def covers(self, start, stop):
                return any(c % 2 for c in range(start, stop))
        assert Odd().recorded({'step': 3}) and not Odd().recorded({'step': 4})
        assert R.Trigger.from_dict(Odd(tag='episode').to_dict()) == R.Manual(tag='episode')

    def test_a_subclass_rebuilds_its_own_kind(self) -> None:
        with pytest.raises(ValueError, match='Every rebuilds triggers of kind "Every", got "At"'):
            R.Every.from_dict(R.At((1, 5)).to_dict())

    def test_the_unit_of_earlier_runs_is_read_as_a_tag(self) -> None:
        from_dict = R.Trigger.from_dict
        assert from_dict({'kind': 'Every', 'n': 2, 'length': 1, 'offset': 0, 'unit': 'episode'}) == R.Every(2, tag='episode')
        assert from_dict({'kind': 'Every', 'n': 2, 'length': 1, 'offset': 0, 'unit': 'step'}) == R.Every(2)
        assert from_dict({'kind': 'When', 'unit': 'episode', 'watch': 'a', 'length': 1, 'condition': 'f'}) == R.Manual(tag='episode')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestMeasurements:

    def test_probes_of_one_key_merge(self) -> None:
        merged = R.measurements.merge_probes([R.SummaryProbe('a.b', reduce=('mean',)), R.SummaryProbe('a.b', reduce=('max', 'mean')), R.RasterProbe('a:c')])
        assert [p.key for p in merged] == ['a.b@summary', 'a:c@raster']
        assert merged[0].reduce == ('mean', 'max')

    def test_a_histogram_merges_with_other_reductions(self) -> None:
        histogram = R.SummaryProbe('a.b', reduce=('hist',), bins=8, range=(-80.0, 0.0))
        for probes in ([R.SummaryProbe('a.b'), histogram], [histogram, R.SummaryProbe('a.b')]):
            (merged,) = R.measurements.merge_probes(probes)
            assert set(merged.reduce) == {'mean', 'std', 'min', 'max', 'hist'} and (merged.bins, merged.range) == (8, (-80.0, 0.0))
        # Without a histogram, bins and range play no part.
        assert R.SummaryProbe('a.b', bins=8, range=(0, 1)) == R.SummaryProbe('a.b')

    def test_conflicting_probes_are_refused(self) -> None:
        with pytest.raises(ValueError, match='differ'):
            R.measurements.merge_probes([R.TraceProbe('a.b', units=(0,)), R.TraceProbe('a.b', units=(1,))])
        with pytest.raises(ValueError, match='differ'):
            R.measurements.merge_probes([R.SummaryProbe('a.b', reduce=('hist',), range=(0, 1)), R.SummaryProbe('a.b', reduce=('hist',), range=(0, 2))])

    def test_round_trip(self) -> None:
        measurements = R.Measurements('r', (R.TraceProbe('a.b', units=(1, 2), stride=2),), trigger=R.Every(3), raw=('m',), views={'m': {'kind': 'image'}})
        assert R.Measurements.from_dict(json.loads(json.dumps(measurements.to_dict()))) == measurements

    def test_measurements_are_recorded_by_hand_unless_given_a_trigger(self) -> None:
        trace = R.TraceProbe('a.c')
        assert R.Measurements('r', (trace,)).trigger == R.Manual()
        with pytest.raises(TypeError):
            R.Measurements('r', (trace,), R.Always())

    @pytest.mark.parametrize('name', ['', 'a/b', 'a b', '.', '..', '-a'])
    def test_a_bad_name_is_refused(self, name) -> None:
        with pytest.raises(ValueError):
            R.Measurements(name, ())

    def test_inconsistent_measurements_are_refused(self) -> None:
        probe = R.SummaryProbe('a.b')
        with pytest.raises(ValueError, match='negative'):
            R.Measurements('r', (probe,), lookback=-1, group=5)
        for lookback in (2.5, True, '4'):
            with pytest.raises(ValueError, match='lookback'):
                R.Measurements('r', (probe,), lookback=lookback, group=5)
        with pytest.raises(TypeError, match='Probe'):
            R.Measurements('r', ('a.b',))
        with pytest.raises(ValueError, match='share'):
            R.Measurements('r', (probe, R.SummaryProbe('a.b', reduce=('max',))), group=5)
        with pytest.raises(TypeError, match='Trigger'):
            R.Measurements('r', (probe,), trigger='always', group=5)

    def test_measurements_that_reduce_have_a_group(self) -> None:
        summary, trace = R.SummaryProbe('a.b'), R.TraceProbe('a.c')
        for probes in ((summary,), (trace, R.SnapshotProbe('a.b')), (R.DeltaProbe('a.b'),)):
            with pytest.raises(ValueError, match='give "group"'):
                R.Measurements('r', probes)
        for group in (0, -3, 2.5, True, ''):
            with pytest.raises(ValueError, match='group'):
                R.Measurements('r', (summary,), group=group)
        # Whole floats, as step counts are often written.
        whole = R.Measurements('r', (summary, R.TraceProbe('a.c', stride=2.0)), group=1e3, lookback=1e4)
        assert (whole.group, whole.probes[0].group, whole.probes[1].stride, whole.lookback) == (1000, 1000, 2, 10_000)
        assert all(type(value) is int for value in (whole.group, whole.probes[1].stride, whole.lookback))
        with pytest.raises(ValueError, match='group of their own'):
            R.Measurements('r', (R.SummaryProbe('a.b', group=7),), group=5)
        with pytest.raises(TypeError, match='group'):
            R.TraceProbe('a.c', group=5)
        # The measurements set the group of the probes that reduce, by steps or by tag.
        measurements = R.Measurements('r', (summary, trace), group='episode')
        assert measurements.probes[0].group == 'episode' and measurements.probes[1] == trace and R.Measurements('r', (trace,)).group is None
        assert R.Measurements.from_dict(json.loads(json.dumps(measurements.to_dict()))) == measurements
        # Measurements sharing a probe agree on its group.
        with pytest.raises(ValueError, match='group'):
            R.measurements.merge_probes([R.Measurements('a', (summary,), group=5).probes[0], R.Measurements('b', (summary,), group=6).probes[0]])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRunner:

    def test_arrays_and_payloads_give_the_same_run(self, brain) -> None:
        outputs = []
        for given in (SIGNAL, SIGNAL.astype(np.float64).tolist(), jnp.asarray(SIGNAL), spark.FloatArray(jnp.asarray(SIGNAL))):
            outputs.append(np.asarray(R.Runner(brain).run(10, {'signal': given})['action'].value))
        for output in outputs[1:]:
            np.testing.assert_array_equal(output, outputs[0])

    def test_negative_spikes_are_inhibitory(self) -> None:
        runner = R.Runner(CASES['alif']()[0])
        for spikes in (np.array([0, 1, -1, 0.5] + [0] * 16, dtype=np.float32), jnp.array([0, 1, -1, 0.5] + [0] * 16)):
            payload = runner._payload('in_spikes', spikes)
            np.testing.assert_array_equal(np.asarray(payload.spikes)[:4], [False, True, True, True])
            np.testing.assert_array_equal(np.asarray(payload.inhibition_mask)[:4], [False, False, True, False])

    def test_spike_inputs_from_arrays(self) -> None:
        model, _, per_step = CASES['alif']()
        spikes = np.asarray(per_step['in_spikes'].spikes)
        a = R.Runner(model, outputs='all').run(STEPS, per_step={'in_spikes': spikes})
        b = R.Runner(model, outputs='all').run(STEPS, per_step={'in_spikes': spikes.astype(np.float32)})
        c = R.Runner(model, outputs='all').run(STEPS, per_step=per_step)
        np.testing.assert_array_equal(np.asarray(a['out_spikes'].spikes), np.asarray(c['out_spikes'].spikes))
        np.testing.assert_array_equal(np.asarray(b['out_spikes'].spikes), np.asarray(c['out_spikes'].spikes))

    def test_outputs(self, brain) -> None:
        assert R.Runner(brain, outputs='all').run(6, {'signal': SIGNAL})['action'].value.shape == (6, 2)
        assert R.Runner(brain, outputs='last').run(6, {'signal': SIGNAL})['action'].value.shape == (2,)
        assert R.Runner(brain, outputs='none').run(6, {'signal': SIGNAL}) is None
        with pytest.raises(ValueError, match='Unknown outputs'):
            R.Runner(brain, outputs='first').run(6, {'signal': SIGNAL})

    def test_per_step_inputs(self) -> None:
        model, held, per_step = CASES['lif']()
        runner = R.Runner(model, outputs='all')
        raw = {'in_spikes': np.asarray(per_step['in_spikes'].spikes)}
        assert runner.run(STEPS, per_step=raw)['out_spikes'].shape == (STEPS, 12)

    def test_inputs_are_checked(self, brain) -> None:
        runner = R.Runner(brain)
        # A whole float, as step counts are often written.
        outputs = R.Runner(brain, outputs='all').run(4.0, {'signal': SIGNAL})
        assert outputs and all(np.shape(leaf)[0] == 4 for leaf in jax.tree.leaves(outputs))
        with pytest.raises(ValueError, match='Missing inputs: signal'):
            runner.run(4)
        with pytest.raises(ValueError, match='not an input'):
            runner.run(4, {'signal': SIGNAL, 'sginal': SIGNAL})
        with pytest.raises(ValueError, match='first axis'):
            runner.run(4, per_step={'signal': np.zeros((3, 8))})
        # One value broadcast over the inputs of a model built with eight is not accepted.
        for held in (np.array([0.3]), 0.3, np.zeros((2, 8)), spark.FloatArray(jnp.zeros((4,), jnp.float16))):
            with pytest.raises(ValueError, match=r'shape \(8,\)'):
                runner.run(4, {'signal': held})
        with pytest.raises(ValueError, match='on every step'):
            runner.run(4, per_step={'signal': np.zeros((4, 1))})
        with pytest.raises(ValueError, match='takes a FloatArray'):
            runner.run(4, {'signal': spark.SpikeArray(jnp.zeros((8,), jnp.uint8))})
        with pytest.raises(ValueError, match='positive integer'):
            runner.run(0, {'signal': SIGNAL})

    def test_the_model_given_stays_usable(self, brain) -> None:
        before = np.asarray(brain.first_pool.soma.potential.value)
        for donate in (True, False):
            runner = R.Runner(brain, donate=donate)
            for _ in range(3):
                runner.run(8, {'signal': SIGNAL})
            np.testing.assert_array_equal(np.asarray(brain.first_pool.soma.potential.value), before)
            assert not np.array_equal(np.asarray(runner.model.first_pool.soma.potential.value), before)

    def test_an_unbuilt_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match='not built'):
            R.Runner(spark.nn.neurons.LIFNeuron(units=(4,)))

    def test_a_path_gives_a_recorder_of_the_default_measurements(self, brain, tmp_path) -> None:
        with R.Runner(brain, tmp_path) as runner:
            runner.run(5, {'signal': SIGNAL})
        run = R.load(runner.recorder.path)
        assert set(run.measurements) == {'summary', 'activity', 'weights'} and run.status == 'finished'

    def test_probes_are_traced_before_the_first_call(self, brain, tmp_path) -> None:
        # The size of an output port of a module is known from its specification.
        with pytest.raises(ValueError, match='has 8 units'):
            R.Recorder(tmp_path, brain, [R.Measurements('a', (R.RasterProbe('spiker:spikes', units=(8,)),), trigger=R.Manual())])
        # The size of an input of the model is known once the model is traced, before the first call runs,
        # however late the measurements are recorded.
        late = R.Measurements('late', (R.TraceProbe('__call__:signal', units=(3, 8)),), trigger=R.At((5000,)))
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [late]))
        for _ in range(2):                                              # a retry is checked again
            with pytest.raises(ValueError, match='has 8 units') as raised:
                runner.run(4, {'signal': SIGNAL})
        assert 'before the first call' in str(raised.value.__notes__)
        assert runner.recorder.step == 0
        runner.close()

    def test_inputs_of_a_neuron_with_other_inputs(self) -> None:
        pool = lambda name, units, origins: spark.ModuleSpecs(
            name=name, module_cls=spark.nn.neurons.ALIFNeuron,
            inputs={'in_spikes': [spark.PortMap(origin=o, port=p) for o, p in origins]},
            config=spark.nn.neurons.ALIFNeuronConfig(_s_units=units, inhibitory_rate=0.3),
        )
        spikes = lambda n: spark.SpikeArray(jnp.zeros((n,), jnp.uint8))
        topologies = [
            ([pool('p', (8,), [('__call__', 'a'), ('__call__', 'b')])], {'a': 5, 'b': 3}),
            ([pool('p', (8,), [('__call__', 'ext'), ('p', 'out_spikes')])], {'ext': 6}),
            ([pool('q', (4,), [('__call__', 'ext'), ('q', 'out_spikes')]), pool('p', (8,), [('__call__', 'ext')])], {'ext': 6}),
        ]
        for specs, sizes in topologies:
            brain = spark.nn.Brain(config=spark.nn.BrainConfig(modules_specs=specs, seed=3))
            brain(**{k: spikes(n) for k, n in sizes.items()})
            runner = R.Runner(brain)
            runner.run(4, {k: np.zeros(n) for k, n in sizes.items()})
        # A module fed by the input alone still gives its shape, found after one fed by several maps.
        with pytest.raises(ValueError, match=r'shape \(6,\)'):
            runner.run(4, {'ext': np.zeros(5)})

    def test_warmup_compiles_the_sets_the_triggers_arm(self, brain, tmp_path) -> None:
        # Calls of 5 steps: summary on every call, activity on one call in 4, weights on the call ending step 39.
        recorder = R.Recorder(tmp_path, brain, R.presets.default(brain, summary_group=10, activity_every=20, activity_length=5, weights_every=40))
        sets = recorder.warmup_sets(5)
        names = {probes: recorded for recorded, probes in recorder._variants.items()}
        warmed = sorted(sorted(names[p]) for p in sets)
        assert warmed == [['activity', 'summary'], ['activity', 'summary', 'weights'], ['summary'], ['summary', 'weights']]
        handed = []
        probes = recorder.probes
        recorder.probes = lambda steps=None: handed.append(probes(steps)) or handed[-1]
        runner = R.Runner(brain, recorder)
        for _ in range(9):
            runner.run(5, {'signal': SIGNAL})
        assert set(handed) <= set(sets)
        # Measurements recorded by hand or by a tag are added to every set the triggers record.
        manual = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Every(10, length=5), group=5),
                                              R.Measurements('b', R.presets.weights(brain), trigger=R.Manual(), group=5)])
        assert len(manual.warmup_sets(5)) == 4
        runner.close()
        manual.close()

    def test_warmup_covers_the_calls_after_a_window_and_the_ends_of_groups(self, brain, tmp_path) -> None:
        probe = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),)
        # Steps 0 to 249 are recorded: calls 0 to 2 of 100 steps, and call 3 after them is not.
        window = R.Recorder(tmp_path, brain, [R.Measurements('a', probe, trigger=R.Between(0, 250), group=100)])
        assert set(window.warmup_sets(100)) == {(), window._merged(frozenset({'a'}))}
        # Recorded for step 0 only, a group of 150 steps keeps it recorded for the calls up to step 150.
        long = R.Recorder(tmp_path, brain, [R.Measurements('a', probe, trigger=R.At((0,)), group=150)])
        assert set(long.warmup_sets(100)) == {(), long._merged(frozenset({'a'}))}
        # All measurements together, the set needing the most memory, even when the triggers never record it.
        other = (R.SummaryProbe('second_pool.soma:spikes', reduce=('active_fraction',)),)
        apart = R.Recorder(tmp_path, brain, [R.Measurements('a', probe, trigger=R.Every(10), group=5), R.Measurements('b', other, trigger=R.Every(10, offset=5), group=5)])
        sets = apart.warmup_sets(5)
        # Compiled ahead, the sets are not counted as sets handed out for calls.
        assert apart._variants[frozenset({'a', 'b'})] in sets and apart._handed == {frozenset()}
        window.close()
        long.close()
        apart.close()

    def test_warmup_compiles_ahead(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Every(14, length=7), group=7)]))
        assert runner.warmup(7, {'signal': SIGNAL}) == 2
        compiled = []
        def listener(name, duration, **_):
            if name == '/jax/core/compile/backend_compile_duration':
                compiled.append(name)
        jax.monitoring.register_event_duration_secs_listener(listener)
        try:
            runner.run(7, {'signal': SIGNAL})
            runner.run(7, {'signal': SIGNAL})
        finally:
            jax.monitoring.unregister_event_duration_listener(listener)
        assert compiled == []
        runner.close()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRecorder:

    def test_calls_are_recorded_as_the_triggers_say(self, brain, tmp_path) -> None:
        probe = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),)
        measurements = [
            R.Measurements('a', probe, trigger=R.Every(8, length=4), group=4), R.Measurements('b', probe, trigger=R.Every(12, length=4, offset=4), group=4),
            R.Measurements('c', probe, trigger=R.Manual(), group=4),
        ]
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, measurements))
        for call in range(12):
            if call == 5:
                runner.recorder.record('c', steps=8)
            runner.run(4, {'signal': SIGNAL})
        runner.close()
        run = R.load(runner.recorder.path)
        assert _recorded_t0(run, 'a') == [4 * c for c in range(0, 12, 2)]
        assert _recorded_t0(run, 'b') == [4 * c for c in (1, 4, 7, 10)]
        assert _recorded_t0(run, 'c') == [20, 24]
        assert run.status == 'finished' and run.step == 48
        with pytest.warns(R.RecordingWarning, match='No measurements "d"'):
            runner.recorder.record('d')

    def test_each_recorded_set_compiles_once(self, brain, tmp_path) -> None:
        measurements = [R.Measurements('a', R.presets.summary(brain), trigger=R.Every(10, length=5), group=5), R.Measurements('b', R.presets.activity(brain), trigger=R.Every(15, length=5))]
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, measurements))
        before = runner._fn._cache_size()
        for _ in range(12):
            runner.run(5, {'signal': SIGNAL})
        runner.close()
        assert runner._fn._cache_size() - before == 4
        assert len(runner.recorder._variants) == 4

    def test_the_same_tuple_is_handed_out(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)])
        probes = recorder.probes()
        with pytest.warns(UserWarning, match='called again before `push`'):
            assert recorder.probes() is probes
        with pytest.raises(ValueError, match='no records'):
            recorder.push({}, 5)
        recorder.close()

    def test_a_recorder_takes_the_configuration_of_the_model(self, brain, tmp_path) -> None:
        measurements = [R.Measurements('a', R.presets.summary(brain), group=5)]
        with R.Recorder(tmp_path, brain.config, measurements) as recorder:
            assert recorder.path.name.split('_')[-2] == 'Brain'
        run = R.load(recorder.path)
        assert run.info['model'] == 'Brain' and (recorder.path / 'model.scfg').exists()
        # A model gives its configuration too, and the default measurements.
        with R.Recorder(tmp_path, brain) as recorder:
            assert {r.name for r in recorder.measurements} == {'summary', 'activity', 'weights'}
        given, taken = (spark.nn.BrainConfig.from_file(str(path / 'model.scfg')).to_dict() for path in (run.path, recorder.path))
        assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)), given, taken))
        with pytest.raises(ValueError, match='Expected measurements'):
            R.Recorder(tmp_path, brain.config)

    def test_stored_records_agree_with_their_raw_values(self, brain, tmp_path) -> None:
        """
            Summaries and deltas written to the run agree with the same metrics computed from the traces
            and snapshots written beside them.
        """
        probes = (
            R.TraceProbe('first_pool.soma.potential'),
            R.SummaryProbe('first_pool.soma.potential', reduce=('mean', 'std', 'min', 'max')),
            R.TraceProbe('first_pool.soma:spikes'),
            R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction', 'active_fraction_per_unit', 'inactive_unit_fraction')),
            R.SnapshotProbe('first_pool.synapses.kernel'),
            R.DeltaProbe('first_pool.synapses.kernel', reduce=('full', 'norm', 'mean_abs')),
        )
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('all', probes, trigger=R.Always(), group=10)], flush_steps=30))
        for _ in range(7):
            runner.run(10, {'signal': SIGNAL})
        runner.close()
        run = R.load(runner.recorder.path)
        assert len(run.windows('all')) == 3
        data = run.timeline('all')
        potential = data['first_pool.soma.potential@trace'].reshape(7, 10, -1).astype(np.float64)
        spikes = data['first_pool.soma:spikes@trace'].reshape(7, 10, -1)
        np.testing.assert_array_equal(data['first_pool.soma.potential@trace#t'], np.arange(70))
        np.testing.assert_allclose(data['first_pool.soma.potential@summary#mean'], potential.mean(axis=(1, 2)), rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(data['first_pool.soma.potential@summary#std'], potential.std(axis=(1, 2)), rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(data['first_pool.soma.potential@summary#min'], potential.min(axis=(1, 2)))
        np.testing.assert_array_equal(data['first_pool.soma.potential@summary#max'], potential.max(axis=(1, 2)))
        np.testing.assert_allclose(data['first_pool.soma:spikes@summary#active_fraction'], spikes.mean(axis=(1, 2)), rtol=1e-6)
        np.testing.assert_allclose(data['first_pool.soma:spikes@summary#active_fraction_per_unit'], spikes.mean(axis=1), rtol=1e-6)
        np.testing.assert_allclose(data['first_pool.soma:spikes@summary#inactive_unit_fraction'], (~spikes.any(axis=1)).mean(axis=1), rtol=1e-6)
        kernels = data['first_pool.synapses.kernel@snapshot'].astype(np.float32)
        change = data['first_pool.synapses.kernel@delta#full']
        np.testing.assert_array_equal(change[1:], kernels[1:] - kernels[:-1])
        np.testing.assert_allclose(data['first_pool.synapses.kernel@delta#norm'], np.sqrt((change.astype(np.float64) ** 2).sum(axis=(1, 2))), rtol=1e-5)
        t, rate = run.scalar('all/first_pool.soma:spikes/active_fraction')
        np.testing.assert_array_equal(t, np.arange(0, 70, 10))
        np.testing.assert_allclose(rate, data['first_pool.soma:spikes@summary#active_fraction'], rtol=1e-6)

    def test_host_values_round_trip(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Every(4), group=10)], hparams={'lr': 0.5})
        runner = R.Runner(brain, recorder)
        for episode in range(3):
            recorder.tag(episode=episode)
            runner.run(10, {'signal': SIGNAL})
            recorder.log(reward=episode * 1.5, steps=10)
            recorder.event('episode_end', length=10, outcome='fell')
        recorder.log(late=1.0, step=100)
        runner.close()
        run = R.load(recorder.path)
        np.testing.assert_array_equal(run.scalar('reward')[1], [0.0, 1.5, 3.0])
        np.testing.assert_array_equal(run.scalar('reward')[0], [10, 20, 30])
        assert run.scalar('late')[0].tolist() == [100]
        assert set(run.scalars('epi')) == set() and set(run.scalars('re')) == {'reward'}
        assert [e['outcome'] for e in run.events('episode_end')] == ['fell'] * 3
        assert [(t['t'], t['value']) for t in run.tags('episode')] == [(0, 0), (10, 1), (20, 2)]
        assert run.hparams == {'lr': 0.5}
        assert type(run.config()).__name__ == 'BrainConfig'
        assert run.info['environment']['versions']['jax'] == jax.__version__

    def test_hparams_tags_and_events_keep_their_numbers(self, brain, tmp_path) -> None:
        hparams = {'lr': np.float32(0.5), 'n': np.int64(5), 'shape': np.arange(3), 'scale': jnp.float32(2.0)}
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())], hparams=hparams)
        recorder.tag(episode=np.int64(3))
        recorder.event('end', reward=np.float32(1.5))
        with pytest.warns(R.RecordingWarning, match='fields of the event'):
            recorder.event('lap', t=12.5, lap=2)                          # written without the key it takes no value for
        recorder.close()
        with pytest.warns(R.RecordingWarning):
            run = R.load(recorder.path)
        assert [(e['t'], e['lap']) for e in run.events('lap')] == [(0, 2)]
        assert run.hparams == {'lr': 0.5, 'n': 5, 'shape': [0, 1, 2], 'scale': 2.0}
        assert run.tags('episode')[0]['value'] == 3 and run.events('end')[0]['reward'] == 1.5

    def test_only_integer_tags_drive_triggers(self, brain, tmp_path) -> None:
        measurements = [R.Measurements('a', R.presets.summary(brain), trigger=R.Every(2, tag='episode'), group=5)]
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, measurements))
        runner.recorder.tag(best=float('inf'), label='3')
        runner.run(5, {'signal': SIGNAL})                               # no episode yet
        runner.recorder.tag(episode=np.int32(2))
        runner.run(5, {'signal': SIGNAL})
        runner.close()
        assert _recorded_t0(R.load(runner.recorder.path), 'a') == [5]

    def test_raw_is_kept_for_the_calls_recorded(self, brain, tmp_path) -> None:
        measurements = R.Measurements('frames', (R.TraceProbe('__call__:signal'),), trigger=R.Every(6, length=3), raw=('env/frame',))
        recorder = R.Recorder(tmp_path, brain, [measurements])
        runner = R.Runner(brain, recorder)
        # A frame logged between calls is the input of the next one.
        for call in range(6):
            recorder.raw('env/frame', np.full((4, 4), call, dtype=np.uint8))
            assert recorder.is_recording('frames') == (call % 2 == 0)
            runner.run(3, {'signal': SIGNAL})
        # A frame the recorder cannot keep is dropped with a warning, and the run goes on.
        with pytest.warns(R.RecordingWarning, match='No measurements declare'):
            recorder.raw('env/other', np.zeros(2))
        with pytest.warns(R.RecordingWarning, match='shape'):
            recorder.raw('env/frame', np.zeros((2, 2), dtype=np.uint8))
        runner.close()
        with pytest.warns(R.RecordingWarning, match='recorded with warnings'):
            run = R.load(recorder.path)
        assert [w['raw'] for w in run.warnings()] == ['env/other', 'env/frame']
        data = run.timeline('frames')
        np.testing.assert_array_equal(data['raw:env/frame'][:, 0, 0], [0, 2, 4])
        np.testing.assert_array_equal(data['raw:env/frame#t'], [0, 6, 12])

    @pytest.mark.parametrize('option', [
        {'queue_size': 0}, {'queue_bytes': 0}, {'flush_steps': 0}, {'flush_seconds': -1.0}, {'flush_bytes': 0},
        {'heartbeat': 0}, {'heartbeat': float('nan')}, {'on_error': 'ignore'}, {'signals': (999,)},
    ])
    def test_options_are_checked(self, brain, tmp_path, option) -> None:
        with pytest.raises(ValueError):
            R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())], **option)
        assert not list(tmp_path.iterdir())

    def test_integer_tags_held_in_arrays_drive_triggers(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),), trigger=R.Every(2, tag='episode'), group=3)])
        runner = R.Runner(brain, recorder)
        for episode in (jnp.int32(0), np.int64(1), jnp.asarray(2, jnp.int32)):
            recorder.tag(episode=episode)
            runner.run(3, {'signal': SIGNAL})
        runner.close()
        assert _recorded_t0(R.load(recorder.path), 'a') == [0, 6]

    def test_a_tag_never_set_is_reported(self, brain, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(R.SETTINGS, 'tag_warning_calls', 2)
        probe = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),)
        recorder = R.Recorder(tmp_path, brain, [
            R.Measurements('a', probe, trigger=R.Every(2, tag='epsiode'), group=3), R.Measurements('b', probe, trigger=R.Every(2, tag='trial'), group=3),
            R.Measurements('c', (R.SummaryProbe('second_pool.soma:spikes', reduce=('active_fraction',)),), group='lap', trigger=R.Always()),
        ])
        runner = R.Runner(brain, recorder)
        recorder.tag(trial='first')
        with pytest.warns(UserWarning) as caught:
            for _ in range(3):
                runner.run(3, {'signal': SIGNAL})
        messages = [str(w.message) for w in caught]
        assert any('"epsiode", which no call to `tag` has set' in m for m in messages)
        assert any('"trial", a tag set to \'first\', which is not an integer' in m for m in messages)
        assert any('grouped by "lap", which no call to `tag` has set' in m for m in messages)
        runner.close()

    def test_raw_is_not_copied_unless_recorded(self, brain, tmp_path) -> None:
        class Frame:
            shape, dtype = (3,), np.dtype(np.float32)
            def __array__(self, *args, **kwargs):
                raise AssertionError('copied')
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', (), trigger=R.Every(6, length=3), raw=('env/frame',))])
        runner = R.Runner(brain, recorder)
        runner.run(3, {'signal': SIGNAL})                                # the call from step 3 is not recorded
        recorder.raw('env/frame', Frame())
        with pytest.warns(R.RecordingWarning, match='shape'):
            recorder.raw('env/frame', np.zeros(4, np.float32))
        with pytest.warns(R.RecordingWarning, match='without a frame'):                   # steps 0 to 3 were recorded without one
            runner.close()

    def test_the_queue_is_bounded_in_bytes(self, brain, tmp_path, monkeypatch) -> None:
        handle = R.recorder._Writer._on_call
        def slow(writer, *args):
            time.sleep(0.2)
            return handle(writer, *args)
        monkeypatch.setattr(R.recorder._Writer, '_on_call', slow)
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.activity(brain), trigger=R.Always())], queue_bytes=1)
        runner = R.Runner(brain, recorder)
        runner.run(5, {'signal': SIGNAL})
        waited = []
        for _ in range(3):
            start = time.monotonic()
            runner.run(5, {'signal': SIGNAL})
            waited.append(time.monotonic() - start)
        # Each call waits for the writer to take the records of the one before.
        assert recorder._stalls >= 2 and min(waited[1:]) > 0.1
        recorder.flush()
        assert recorder._queued_bytes == 0
        runner.close()

    def test_the_index_is_committed_about_once_a_second(self, brain, tmp_path, monkeypatch) -> None:
        commits = []
        commit = R.recorder._Writer._commit
        monkeypatch.setattr(R.recorder._Writer, '_commit', lambda writer: (commits.append(time.monotonic()), commit(writer)))
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=3)])
        runner = R.Runner(brain, recorder)
        runner.run(3, {'signal': SIGNAL})
        recorder.flush()
        start, before = time.monotonic(), len(commits)
        for call in range(60):
            runner.run(3, {'signal': SIGNAL})
            recorder.log(loss=float(call))
        elapsed = time.monotonic() - start
        # Once a second while busy, and once per wait for the queue while idle, as when training is slow.
        assert len(commits) - before <= elapsed / min(R.SETTINGS.commit_every, R.SETTINGS.requests_every) + 2
        runner.close()
        assert len(R.load(recorder.path).scalar('loss')[0]) == 60

    def test_push_does_not_wait_for_the_writer(self, brain, tmp_path, monkeypatch) -> None:
        original = R.Packed.unpack
        release, unpacked = threading.Event(), []
        def held(self, completed=True):
            release.wait(60)
            unpacked.append(True)
            return original(self, completed)
        monkeypatch.setattr(R.Packed, 'unpack', held)
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)]))
        for _ in range(5):
            runner.run(5, {'signal': SIGNAL})
        # Every chunk was run while the writer waited on the first one.
        assert not unpacked
        release.set()
        runner.close()
        assert len(R.load(runner.recorder.path).scalar('a/first_pool.soma:spikes/active_fraction')[0]) == 5

    def test_windows_are_written_by_size_too(self, brain, tmp_path) -> None:
        probes = (R.TraceProbe('first_pool.soma.potential'),)
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', probes, trigger=R.Always())], flush_bytes=1))
        for _ in range(3):
            runner.run(5, {'signal': SIGNAL})
        runner.recorder._queue.join()
        assert [w.spans for w in R.load(runner.recorder.path).windows('a')] == [1, 1, 1]
        runner.close()

    def test_a_run_reads_back_under_any_path(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path / 'exp #1?%', brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())], name='a/b c'))
        runner.run(5, {'signal': SIGNAL})
        runner.close()
        run = R.load(runner.recorder.path)
        assert runner.recorder.path.parent == tmp_path / 'exp #1?%' and '_a-b-c_' in run.path.name
        assert _recorded_t0(run, 'a') == [0] and len(run.scalar_keys()) > 0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRecordingWarnings:
    """
        Mistakes in what the loop hands over: warned of, written to the run, and the run goes on.
    """

    @staticmethod
    def _measurements(brain):
        return [
            R.Measurements('frames', (R.TraceProbe('__call__:signal'),), trigger=R.Every(6, length=3), raw=('env/frame',)),
            R.Measurements('a', R.presets.summary(brain), group=3, trigger=R.Always()),
        ]

    def test_a_declared_stream_without_frames_is_warned_of(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, self._measurements(brain))
        runner = R.Runner(brain, recorder)
        with pytest.warns(R.RecordingWarning, match='No measurements declare the raw stream "env/fram"'):
            recorder.raw('env/fram', np.zeros((4, 4), dtype=np.uint8))                 # a typo of the name
        with pytest.warns(R.RecordingWarning, match='without a frame of their raw stream "env/frame"'):
            for _ in range(4):
                runner.run(3, {'signal': SIGNAL})
            runner.close()
        with pytest.warns(R.RecordingWarning, match='recorded with warnings'):
            run = R.load(recorder.path)
        # The run went on, with everything else.
        assert run.status == 'finished' and run.step == 12 and _recorded_t0(run, 'a') == [0, 3, 6, 9]
        assert [w['raw'] for w in run.warnings()] == ['env/fram', 'env/frame'] and '2 warnings' in repr(run)

    def test_a_stream_fed_once_per_group_is_not_warned_of(self, brain, tmp_path) -> None:
        # A frame at the start of every episode. The group of an episode ends when the next one recorded starts.
        measurements = [R.Measurements('episode', (R.TraceProbe('__call__:signal'),), group='episode', trigger=R.Every(2, tag='episode'), raw=('env/frame',))]
        recorder = R.Recorder(tmp_path, brain, measurements)
        runner = R.Runner(brain, recorder)
        with warnings.catch_warnings():
            warnings.simplefilter('error', R.RecordingWarning)
            for episode in range(6):
                recorder.tag(episode=episode)
                recorder.raw('env/frame', np.full((2,), episode, dtype=np.uint8))
                for _ in range(3):
                    runner.run(2, {'signal': SIGNAL})
            runner.close()
            run = R.load(recorder.path)
        assert run.warnings() == [] and sorted(run.read('episode')) == [0, 2, 4]

    def test_a_short_run_can_raise_each_warning_where_it_happens(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, self._measurements(brain))
        runner = R.Runner(brain, recorder)
        with warnings.catch_warnings():
            warnings.simplefilter('error', R.RecordingWarning)
            with pytest.raises(R.RecordingWarning, match='No measurements "franes"'):
                recorder.record('franes')
            # Found by the writer, raised by the next call of the loop at the latest by `close`.
            with pytest.raises(R.RecordingWarning, match='without a frame'):
                for _ in range(4):
                    runner.run(3, {'signal': SIGNAL})
                runner.close()
        runner.close()
        with pytest.warns(R.RecordingWarning):
            assert R.load(recorder.path).status == 'finished'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestFailures:
    """
        Runs that end badly are marked, and lose nothing written.
    """

    def test_a_failed_run_is_marked(self, brain, tmp_path) -> None:
        with pytest.raises(KeyError):
            with R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]) as recorder:
                raise KeyError('broken')
        run = R.load(recorder.path)
        assert run.status == 'failed' and 'KeyError' in run.info['error']

    def test_a_writer_failure_reaches_the_caller(self, brain, tmp_path, monkeypatch) -> None:
        def broken(path, arrays):
            raise OSError('disk full')
        monkeypatch.setattr(spark.recording.store, 'write_arrays', broken)
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=3, trigger=R.Always())], flush_steps=1))
        with pytest.raises(RuntimeError, match='writer'):
            for _ in range(5):
                runner.run(3, {'signal': SIGNAL})
                runner.recorder.flush()
        with pytest.raises(RuntimeError, match='writer'):
            runner.close()
        assert R.load(runner.recorder.path).status == 'failed'

    def test_a_failed_final_write_does_not_hang_close(self, brain, tmp_path, monkeypatch) -> None:
        def broken(path, arrays):
            raise OSError('disk full')
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=3, trigger=R.Always())]))
        runner.run(3, {'signal': SIGNAL})                               # the window stays open until close
        monkeypatch.setattr(spark.recording.store, 'write_arrays', broken)
        raised = []
        def close():
            try:
                runner.close()
            except RuntimeError as error:
                raised.append(error)
        closing = threading.Thread(target=close, daemon=True)
        closing.start()
        closing.join(timeout=30)
        assert not closing.is_alive() and raised and 'writer' in str(raised[0])
        run = R.load(runner.recorder.path)
        assert run.status == 'failed' and 'disk full' in run.info['error']

    @staticmethod
    def _break_events(monkeypatch):
        def broken(writer, *args):
            raise sqlite3.OperationalError('disk I/O error')
        monkeypatch.setattr(R.recorder._Writer, '_on_event', broken)

    def test_a_writer_failure_writes_the_open_windows(self, brain, tmp_path, monkeypatch) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)])
        runner = R.Runner(brain, recorder)
        for _ in range(3):
            runner.run(5, {'signal': SIGNAL})
        self._break_events(monkeypatch)
        recorder.event('boom')
        _wait_for(lambda: recorder._writer.error is not None)
        with pytest.raises(RuntimeError, match='writer'):
            runner.run(5, {'signal': SIGNAL})
        with pytest.raises(RuntimeError, match='writer'):
            runner.close()
        run = R.load(recorder.path)
        assert run.status == 'failed' and 'disk I/O error' in run.info['error']
        assert _recorded_t0(run, 'a') == [0, 5, 10]

    def test_a_writer_failure_can_leave_training_going(self, brain, tmp_path, monkeypatch) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)], on_error='continue')
        runner = R.Runner(brain, recorder)
        runner.run(5, {'signal': SIGNAL})
        self._break_events(monkeypatch)
        recorder.event('boom')
        _wait_for(lambda: recorder._writer.error is not None)
        with pytest.warns(UserWarning, match='no longer recorded') as caught:
            for _ in range(3):
                runner.run(5, {'signal': SIGNAL})
                recorder.log(loss=1.0)
        assert len(caught) == 1 and recorder.step == 20
        runner.close()
        run = R.load(recorder.path)
        assert run.status == 'failed' and _recorded_t0(run, 'a') == [0]

    def test_a_writer_failure_and_recording_warnings_are_warned_of_apart(self, brain, tmp_path, monkeypatch) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)], on_error='continue')
        runner = R.Runner(brain, recorder)
        with pytest.warns(R.RecordingWarning, match='"before" takes a number'):
            recorder.log(before='none')
        runner.run(5, {'signal': SIGNAL})
        self._break_events(monkeypatch)
        recorder.event('boom')
        _wait_for(lambda: recorder._writer.error is not None)
        with pytest.warns(UserWarning, match='no longer recorded'):
            runner.run(5, {'signal': SIGNAL})
        with pytest.warns(R.RecordingWarning, match='"after" takes a number'):
            recorder.log(after='none')
        runner.close()

    @pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='SIGUSR1 is POSIX')
    def test_a_signal_stops_the_run_before_the_next_chunk(self, brain, tmp_path) -> None:
        previous = signal.getsignal(signal.SIGUSR1)
        with pytest.raises(R.Preempted) as raised:
            with R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)], signals=(signal.SIGUSR1,)) as recorder:
                runner = R.Runner(brain, recorder)
                runner.run(5, {'signal': SIGNAL})
                os.kill(os.getpid(), signal.SIGUSR1)
                runner.run(5, {'signal': SIGNAL})
        assert raised.value.code == 128 + signal.SIGUSR1 and recorder.step == 5
        run = R.load(recorder.path)
        assert run.status == 'preempted' and 'SIGUSR1' in run.info['error'] and _recorded_t0(run, 'a') == [0]
        assert signal.getsignal(signal.SIGUSR1) == previous

    @pytest.mark.skipif(not hasattr(signal, 'SIGTERM') or sys.platform == 'win32', reason='signals of POSIX')
    def test_a_script_stopped_by_its_scheduler_leaves_its_run_preempted(self, tmp_path) -> None:
        path = _python(f'''
            import os, signal, sys
            sys.path.insert(0, {HERE!r})
            import numpy as np, spark
            from cases import CASES
            R = spark.recording
            brain, _, _ = CASES['brain']()
            recorder = R.Recorder({str(tmp_path)!r}, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)], signals=(signal.SIGTERM,))
            print(recorder.path, flush=True)
            runner = R.Runner(brain, recorder)
            for chunk in range(10):
                if chunk == 3:
                    os.kill(os.getpid(), signal.SIGTERM)
                runner.run(5, {{'signal': np.ones(8)}})
            print('not stopped', flush=True)
        ''', returncode=128 + signal.SIGTERM)
        run = R.load(path)
        assert run.status == 'preempted' and _recorded_t0(run, 'a') == [0, 5, 10] and run.step == 15

    def test_exiting_cleanly_finishes_the_run(self, brain, tmp_path) -> None:
        with pytest.raises(SystemExit):
            with R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]) as recorder:
                sys.exit(0)
        assert R.load(recorder.path).status == 'finished'

    def test_a_write_failing_for_a_moment_is_tried_again(self, brain, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(R.SETTINGS, 'retry_after', (0.01, 0.01))
        write, failures = R.store.write_arrays, [OSError(5, 'Input/output error'), OSError(116, 'Stale file handle')]
        def flaky(path, arrays):
            if failures:
                raise failures.pop(0)
            write(path, arrays)
        monkeypatch.setattr(R.store, 'write_arrays', flaky)
        with R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)]) as recorder:
            R.Runner(brain, recorder).run(5, {'signal': SIGNAL})
        run = R.load(recorder.path)
        assert run.status == 'finished' and _recorded_t0(run, 'a') == [0] and not failures

    def test_a_reader_holding_the_index_delays_commits_without_failing(self, brain, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(R.SETTINGS, 'busy_timeout', 0.2)
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)])
        runner = R.Runner(brain, recorder)
        runner.run(5, {'signal': SIGNAL})
        recorder.flush()
        # A reader in the middle of a transaction, as a cursor not read to its end, for a second.
        reader = sqlite3.connect(recorder.path / 'index.sqlite', isolation_level=None, check_same_thread=False)
        reader.execute('BEGIN')
        reader.execute('SELECT COUNT(*) FROM scalars').fetchall()
        release = threading.Timer(1.0, reader.execute, ('COMMIT',))
        release.start()
        with pytest.warns(UserWarning, match='held by another process'):
            runner.run(5, {'signal': SIGNAL})
            recorder.flush()
        release.join()
        reader.close()
        runner.close()
        run = R.load(recorder.path)
        assert run.status == 'finished' and len(run.scalar('a/first_pool.soma:spikes/active_fraction')[0]) == 2

    def test_a_failed_write_leaves_no_partial_file(self, tmp_path, monkeypatch) -> None:
        def broken(file, **arrays):
            file.write(b'half')
            raise OSError(28, 'No space left on device')
        monkeypatch.setattr(np, 'savez', broken)
        with pytest.raises(OSError, match='No space'):
            R.store.write_arrays(tmp_path / 'windows' / 'x.npz', {'a': np.zeros(3)})
        assert not list((tmp_path / 'windows').iterdir())

    def test_a_recorder_left_open_is_closed_at_exit(self, tmp_path) -> None:
        path = _python(f'''
            import sys
            sys.path.insert(0, {HERE!r})
            import numpy as np, spark
            from cases import CASES
            R = spark.recording
            def train():
                brain, _, _ = CASES['brain']()
                runner = R.Runner(brain, R.Recorder({str(tmp_path)!r}, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]))
                for _ in range(3):
                    runner.run(5, {{'signal': np.ones(8)}})
                return runner.recorder.path
            print(train(), flush=True)
            import gc; gc.collect()
        ''')
        run = R.load(path)
        assert run.status == 'finished' and run.step == 15 and len(run.timeline('a')['span_t0']) == 3

    def test_a_script_that_raises_leaves_its_run_failed(self, tmp_path) -> None:
        path = _python(f'''
            import sys
            sys.path.insert(0, {HERE!r})
            import numpy as np, spark
            from cases import CASES
            R = spark.recording
            brain, _, _ = CASES['brain']()
            runner = R.Runner(brain, R.Recorder({str(tmp_path)!r}, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]))
            runner.run(5, {{'signal': np.ones(8)}})
            print(runner.recorder.path, flush=True)
            raise ValueError('training diverged')
        ''', returncode=1)
        run = R.load(path)
        assert run.status == 'failed' and 'training diverged' in run.info['error'] and _recorded_t0(run, 'a') == [0]

    def test_a_crash_leaves_complete_windows_and_resumes_after_them(self, tmp_path) -> None:
        path = _python(f'''
            import os, sys, time
            sys.path.insert(0, {HERE!r})
            import numpy as np, spark
            from cases import CASES
            brain, _, _ = CASES['brain']()
            R = spark.recording
            recorder = R.Recorder({str(tmp_path)!r}, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)],
                                  flush_steps=10, heartbeat=0.1)
            runner = R.Runner(brain, recorder)
            for _ in range(7):
                runner.run(5, {{'signal': np.ones(8)}})
            recorder.flush()
            for _ in range(3):
                runner.run(5, {{'signal': np.ones(8)}})
            recorder._queue.join()                                      # handled, the last chunk not yet written
            time.sleep(1.0)                                             # the writer commits the index once idle
            print(recorder.path, flush=True)
            os._exit(1)
        ''', returncode=1)
        run = R.load(path)
        _wait_for(lambda: run.refresh().status == 'crashed')
        windows = run.windows('a')
        assert [w.spans for w in windows] == [2, 2, 2, 1, 2]
        for window in windows:
            assert len(window.read()['span_t0']) == window.spans
        assert len(run.scalar('a/first_pool.soma:spikes/active_fraction')[0]) == 10
        assert not list((run.path / 'windows').rglob('*.partial'))
        # Resuming marks the chunk whose window was lost and continues after it.
        brain, _, _ = CASES['brain']()
        resumed = R.Runner(brain, R.Recorder.resume(path, brain, flush_steps=10))
        assert resumed.recorder.step == 50
        resumed.run(5, {'signal': SIGNAL})
        resumed.close()
        run = R.load(path)
        assert [w.number for w in run.windows('a')] == [0, 1, 2, 3, 4, 6]
        assert _recorded_t0(run, 'a') == [*range(0, 45, 5), 50]
        chunks = run.rows('spans')
        assert [row[3] for row in chunks if row[2] >= 0] == _recorded_t0(run, 'a')
        assert [(row[2], row[3]) for row in chunks if row[2] < 0] == [(-1, 45)]            # listed, then lost
        # The number of the lost window is not used again by a later resume either.
        again = R.Runner(brain, R.Recorder.resume(path, brain, flush_steps=10))
        again.run(5, {'signal': SIGNAL})
        again.close()
        assert [w.number for w in R.load(path).windows('a')] == [0, 1, 2, 3, 4, 6, 7]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestResume:

    def test_resume_continues_the_run(self, brain, tmp_path) -> None:
        measurements = [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)]
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, measurements, flush_steps=10))
        for _ in range(4):
            runner.run(5, {'signal': SIGNAL})
        runner.close()
        path = runner.recorder.path
        resumed = R.Runner(brain, R.Recorder.resume(path, brain, flush_steps=10))
        assert resumed.recorder.step == 20
        for _ in range(3):
            resumed.run(5, {'signal': SIGNAL})
        resumed.close()
        run = R.load(path)
        assert run.status == 'finished' and run.step == 35 and len(run.info['resumed']) == 1
        assert [w.number for w in run.windows('a')] == [0, 1, 2, 3]
        assert _recorded_t0(run, 'a') == list(range(0, 35, 5))

    def test_a_run_being_written_is_not_resumed(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())])
        with pytest.raises(RuntimeError, match='being written'):
            R.Recorder.resume(recorder.path, brain)
        recorder.close()
        R.Recorder.resume(recorder.path, brain).close()

    def test_a_window_written_but_not_listed_is_kept(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)], flush_steps=10))
        for _ in range(4):
            runner.run(5, {'signal': SIGNAL})
        runner.close()
        path = runner.recorder.path
        # As when the process ends between writing a window and listing it.
        with sqlite3.connect(path / 'index.sqlite') as connection:
            connection.execute("DELETE FROM windows WHERE measurements = 'a' AND window = 1")
        R.Recorder.resume(path, brain).close()
        run = R.load(path)
        assert [(w.number, w.t0, w.t1, w.spans) for w in run.windows('a')] == [(0, 0, 10, 2), (1, 10, 20, 2)]
        assert all(row[2] >= 0 for row in run.rows('spans'))

    def test_requests_left_pending_expire(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Manual(), group=5)])
        run = R.load(recorder.path)
        first = run.record('a')                                            # no chunk follows to apply it
        recorder.close()
        with pytest.raises(ValueError, match='not being written'):
            run.record('a')
        # As a request written when the recorder ended.
        second = R.store.write_request(recorder.path, 'record', {'measurements': 'a', 'steps': 1})
        assert [r['status'] for r in R.load(recorder.path).requests()] == ['expired', 'pending']
        R.Recorder.resume(recorder.path, brain).close()
        assert [(r['id'], r['status']) for r in R.load(recorder.path).requests()] == [(first, 'expired'), (second, 'expired')]
        assert not list((recorder.path / 'requests').glob('*.json'))

    def test_resuming_waits_for_a_lock_held_briefly(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())])
        recorder.close()
        held = R.store.lock(recorder.path)                              # as the viewer asking whether the run is written
        assert held is not None and R.store.locked(recorder.path)
        release = threading.Timer(0.5, R.store.unlock, (held,))
        release.start()
        R.Recorder.resume(recorder.path, brain).close()
        release.join()

    def test_measurements_recorded_by_conditions_are_reported_on_resume(self, brain, tmp_path) -> None:
        probe = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),)
        measurements = [R.Measurements('w', probe, trigger=R.Always(), group=5), R.Measurements('c', probe, trigger=R.When(bool, watch='w'), group=5)]
        recorder = R.Recorder(tmp_path, brain, measurements)
        recorder.close()
        with pytest.warns(UserWarning, match='recorded by conditions'):
            resumed = R.Recorder.resume(recorder.path, brain)
        assert resumed.measurements[1].trigger == R.Manual()
        resumed.close()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            R.Recorder.resume(recorder.path, brain, measurements).close()

    def test_a_run_id_names_the_run(self, brain, tmp_path) -> None:
        measurements = [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]
        with R.Recorder(tmp_path, brain, measurements, run_id='job-42') as recorder:
            R.Runner(brain, recorder).run(5, {'signal': SIGNAL})
        assert recorder.path == tmp_path / 'job-42'
        with pytest.raises(FileExistsError):
            R.Recorder(tmp_path, brain, measurements, run_id='job-42')
        with pytest.raises(ValueError, match='Invalid run id'):
            R.Recorder(tmp_path, brain, measurements, run_id='jobs/42')
        with R.Recorder.resume(tmp_path / 'job-42', brain) as resumed:
            assert resumed.step == 5
        assert sorted(p.name for p in tmp_path.iterdir()) == ['job-42']

    def test_runs_of_an_experiment(self, brain, tmp_path) -> None:
        measurements = [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]
        for experiment in ('X', 'X', 'Y', None):
            with R.Recorder(tmp_path, brain, measurements, experiment=experiment) as recorder:
                R.Runner(brain, recorder).run(5, {'signal': SIGNAL})
        # Created within a second: their order is that of their ids.
        assert sorted(str(run.experiment) for run in R.runs(tmp_path)) == ['None', 'X', 'X', 'Y']
        assert len(R.runs(tmp_path, experiment='X')) == 2 and [run.experiment for run in R.runs(tmp_path, experiment='Y')] == ['Y']
        # Kept when the run is resumed.
        with R.Recorder.resume(R.runs(tmp_path, experiment='Y')[0].path, brain):
            pass
        assert R.runs(tmp_path, experiment='Y')[0].experiment == 'Y'
        for experiment in ('', '  ', 3):
            with pytest.raises(ValueError, match='experiment'):
                R.Recorder(tmp_path, brain, measurements, experiment=experiment)

    def test_the_tag_scalars_are_logged_per(self, brain, tmp_path) -> None:
        rate = lambda pool: (R.SummaryProbe(f'{pool}.soma:spikes', reduce=('active_fraction',)),)
        recorder = R.Recorder(tmp_path, brain, [
            R.Measurements('a', rate('first_pool'), group='episode', trigger=R.Always()),
            R.Measurements('b', rate('second_pool'), group=5, trigger=R.Always()),
        ])
        runner = R.Runner(brain, recorder)
        with pytest.warns(R.RecordingWarning, match='not set to an integer yet'):
            recorder.log({'early': 1.0}, tag='episode')
        for episode in range(3):
            recorder.tag(episode=episode)
            runner.run(5, {'signal': SIGNAL})
            recorder.log({'episode/steps': 5.0}, tag='episode')
            recorder.log(loss=0.1)
        with pytest.warns(R.RecordingWarning, match='logged per step'):
            recorder.log({'other': 1.0}, tag='step')
        # Logged per another tag: it keeps the first, warned of by the writer.
        recorder.log({'episode/steps': 5.0})
        with pytest.warns(R.RecordingWarning, match='stays per the tag'):
            runner.close()
        run = R.Run(recorder.path)
        assert (run.tag_of('episode/steps'), run.tag_of('early'), run.tag_of('loss'), run.tag_of('other')) == ('episode', 'episode', None, None)
        assert run.tag_of('a/first_pool.soma:spikes/active_fraction') == 'episode' and run.tag_of('b/second_pool.soma:spikes/active_fraction') is None
        assert run.tag_of('missing') is None and len(run.scalar('episode/steps')[0]) == 4

    def test_runs_written_before_tags_were_kept(self, brain, tmp_path) -> None:
        rate = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),)
        with R.Recorder(tmp_path, brain, [R.Measurements('episode', rate, group='episode', trigger=R.Always())]) as recorder:
            recorder.tag(episode=0)
            R.Runner(brain, recorder).run(5, {'signal': SIGNAL})
            recorder.log({'episode/steps': 5.0}, tag='episode')
        # The index as it was written before.
        connection = sqlite3.connect(recorder.path / 'index.sqlite')
        connection.execute('ALTER TABLE keys DROP COLUMN tag')
        connection.commit()
        connection.close()
        run = R.Run(recorder.path)
        # Summaries have the tag of their measurements; a logged series, even one named after them, is per step.
        assert run.tag_of('episode/first_pool.soma:spikes/active_fraction') == 'episode' and run.tag_of('episode/steps') is None
        with R.Recorder.resume(recorder.path, brain) as resumed:
            resumed.log({'loss': 1.0}, tag='episode')
        assert R.Run(recorder.path).tag_of('loss') == 'episode'

    def test_windows_of_other_probes_are_not_joined(self, brain, tmp_path) -> None:
        with R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)]) as recorder:
            R.Runner(brain, recorder).run(5, {'signal': SIGNAL})
        other = [R.Measurements('a', (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),), trigger=R.Always(), group=5)]
        with R.Recorder.resume(recorder.path, brain, other) as resumed:
            R.Runner(brain, resumed).run(5, {'signal': SIGNAL})
        run = R.load(recorder.path)
        assert [w.number for w in run.windows('a')] == [0, 1]
        with pytest.raises(ValueError, match='different probes'):
            run.timeline('a')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRecord:
    """
        Measurements recorded from outside the loop: requests, conditions, and the calls kept before.
    """

    PROBE = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),)

    def test_a_request_from_another_handle_records_measurements(self, brain, tmp_path) -> None:
        measurements = [R.Measurements('manual', self.PROBE, trigger=R.Manual(), group=4)]
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, measurements))
        runner.run(4, {'signal': SIGNAL})
        run = R.load(runner.recorder.path)                              # as the viewer or another process would
        request = run.record('manual', steps=8)
        _wait_for(lambda: run.requests()[0]['status'] == 'received')
        for _ in range(4):
            runner.run(4, {'signal': SIGNAL})
        runner.close()
        run = R.load(runner.recorder.path)
        assert _recorded_t0(run, 'manual') == [4, 8]
        assert [(r['id'], r['status']) for r in run.requests()] == [(request, 'applied')]
        recorded = run.events('record')
        assert len(recorded) == 1 and recorded[0]['by'] == 'request' and recorded[0]['measurements'] == 'manual'
        with pytest.raises(ValueError, match='No measurements'):
            run.record('nope')

    @pytest.mark.skipif(sys.platform == 'win32' or os.geteuid() == 0, reason='permissions of POSIX, not enforced for root')
    def test_a_request_that_cannot_be_removed_is_applied_once(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('m', self.PROBE, trigger=R.Manual(), group=4)]))
        runner.run(4, {'signal': SIGNAL})
        path = runner.recorder.path
        run = R.load(path)
        run.record('m')
        (path / 'requests').chmod(0o555)                                # as a directory whose files others cannot remove
        try:
            _wait_for(lambda: run.requests()[0]['status'] == 'received')
            for _ in range(3):
                time.sleep(R.SETTINGS.requests_every + 0.1)             # read again, were it read twice
                runner.run(4, {'signal': SIGNAL})
            runner.close()
            R.Recorder.resume(path, brain).close()
        finally:
            (path / 'requests').chmod(0o755)
        assert _recorded_t0(R.load(path), 'm') == [4] and len(R.load(path).requests()) == 1

    def test_malformed_requests_are_rejected(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('m', self.PROBE, trigger=R.Manual(), group=4)]))
        runner.run(4, {'signal': SIGNAL})
        requests = runner.recorder.path / 'requests'
        requests.mkdir(exist_ok=True)
        bad = [
            '{"kind": "record", "payload": {"measurements": "m", "steps": "two"}}', 'null', '[]', 'not json',
            '{"kind": "record", "payload": {"measurements": ["m"]}}', '{"kind": "record", "payload": {"measurements": "m", "steps": 1e400}}',
            '{"kind": "record", "payload": {"measurements": "m", "steps": true}}', '{"kind": "stop", "payload": {"measurements": "m"}}',
        ]
        for index, text in enumerate(bad):
            (requests / f'{index:020d}-bad.json').write_text(text)
        run = R.load(runner.recorder.path)
        _wait_for(lambda: all(r['status'] != 'pending' for r in run.requests()))
        runner.run(4, {'signal': SIGNAL})
        runner.close()
        run = R.load(runner.recorder.path)
        assert [r['status'] for r in run.requests()] == ['rejected'] * len(bad) and run.status == 'finished'
        assert _recorded_t0(run, 'm') == []
        with pytest.raises(ValueError, match='positive integer'):
            R.load(runner.recorder.path).record('m', steps=0)

    def test_a_condition_records_measurements(self, brain, tmp_path) -> None:
        fired = []
        def once(record):
            fired.append(record.summaries['first_pool.soma:spikes'].active_fraction)
            return len(fired) == 2
        measurements = [
            R.Measurements('watched', self.PROBE, trigger=R.Always(), group=4),
            R.Measurements('hit', self.PROBE, trigger=R.When(once, watch='watched', length=8), group=4),
        ]
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, measurements))
        for _ in range(6):
            runner.run(4, {'signal': SIGNAL})
            runner.recorder.flush()                                     # the condition runs once the group is written
        runner.close()
        run = R.load(runner.recorder.path)
        # Met on the second group (t0 = 4), recorded for the next 8 steps.
        assert _recorded_t0(run, 'hit') == [8, 12]
        assert run.events('record')[0]['by'] == 'condition' and run.events('record')[0]['at'] == 4
        assert run.measurements['hit'].trigger == R.Manual()

    def test_a_failing_condition_is_dropped_and_recording_goes_on(self, brain, tmp_path) -> None:
        measurements = [
            R.Measurements('watched', self.PROBE, trigger=R.Always(), group=4),
            R.Measurements('hit', self.PROBE, trigger=R.When(lambda record: record.summaries['nope'], watch='watched'), group=4),
        ]
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, measurements))
        with pytest.warns(UserWarning, match='no longer tested'):
            for _ in range(3):
                runner.run(4, {'signal': SIGNAL})
                runner.recorder.flush()
        runner.close()
        run = R.load(runner.recorder.path)
        assert _recorded_t0(run, 'watched') == [0, 4, 8] and _recorded_t0(run, 'hit') == []
        (error,) = run.events('error')
        assert error['measurements'] == 'hit' and 'KeyError' in error['error']

    def test_a_when_trigger_must_watch_other_measurements(self, brain, tmp_path) -> None:
        with pytest.raises(ValueError, match='names no other measurements'):
            R.Recorder(tmp_path, brain, [R.Measurements('a', self.PROBE, trigger=R.When(bool, watch='a'), group=5)])

    def test_steps_of_the_lookback_are_kept(self, brain, tmp_path) -> None:
        probes = (R.TraceProbe('first_pool.soma.potential'),)
        measurements = [
            R.Measurements('everything', probes, trigger=R.Always()),
            R.Measurements('late', probes, trigger=R.Manual(), lookback=15),
        ]
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, measurements))
        for call in range(8):
            if call == 6:
                runner.recorder.record('late')
            runner.run(5, {'signal': SIGNAL})
        runner.close()
        run = R.load(runner.recorder.path)
        # The calls of the 15 steps before kept, then the call from step 30, which recorded it.
        assert _recorded_t0(run, 'late') == [15, 20, 25, 30]
        late, everything = run.timeline('late'), run.timeline('everything')
        key = 'first_pool.soma.potential@trace'
        np.testing.assert_array_equal(late[f'{key}#t'], np.arange(15, 35))
        np.testing.assert_array_equal(late[key], everything[key][15:35])

    def test_kept_calls_are_not_handed_over_until_recorded(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('late', self.PROBE, trigger=R.Manual(), lookback=4, group=2)])
        runner = R.Runner(brain, recorder)
        handed = []
        put = recorder._put
        recorder._put = lambda item: (handed.append(item), put(item))
        for _ in range(3):
            runner.run(2, {'signal': SIGNAL})
        assert [item[2] for item in handed if item[0] == 'call'] == [None, None, None]
        assert len(recorder._rings['late']) == 2
        runner.close()

    def test_pushing_without_asking_for_probes_leaves_the_kept_calls_alone(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('late', self.PROBE, trigger=R.Manual(), lookback=2, group=2)])
        runner = R.Runner(brain, recorder)
        runner.run(2, {'signal': SIGNAL})
        recorder.push({}, 2)                                            # a call run by hand, nothing captured
        recorder.record('late')
        runner.run(2, {'signal': SIGNAL})
        runner.close()
        assert _recorded_t0(R.load(recorder.path), 'late') == [0, 4]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestCheckpoints:

    def test_a_checkpoint_restores_the_state_it_was_taken_at(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Every(10), group=5)]))
        for _ in range(3):
            runner.run(5, {'signal': SIGNAL})
        path = runner.checkpoint()
        expected = jax.tree.map(np.asarray, runner.state)
        for _ in range(3):
            runner.run(5, {'signal': SIGNAL})                           # the state is donated again and again
        runner.close()
        run = R.load(runner.recorder.path)
        assert run.checkpoints() == [15] and path.name == f'{15:012d}.spark'
        restored = run.restore()
        for got, want in zip(jax.tree.leaves(spark.split((restored))[1]), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(np.asarray(got), want)
        assert run.events('checkpoint')[0]['t'] == 15
        with pytest.raises(FileNotFoundError):
            run.restore(step=3)

    def test_a_checkpoint_is_written_by_the_model(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path / 'runs', brain, [R.Measurements('a', R.presets.summary(brain), group=5)]))
        runner.run(5, {'signal': SIGNAL})
        path = runner.checkpoint()
        own = runner.model.checkpoint(tmp_path / 'own', verbose=False)          # the same state, saved by the model itself
        runner.close()
        from_run, from_model = type(brain).from_checkpoint(path, verbose=False), type(brain).from_checkpoint(own, verbose=False)
        assert type(from_run) is type(from_model) is type(brain)
        for got, want in zip(jax.tree.leaves(spark.split((from_run))[1]), jax.tree.leaves(spark.split((from_model))[1])):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_a_checkpoint_that_cannot_be_written_raises_when_the_recorder_closes(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5)])
        recorder.checkpoint(type(brain)(config=brain.config))                  # never built
        with pytest.raises(RuntimeError, match='not yet built'):
            recorder.close()
        run = R.load(recorder.path)
        assert run.checkpoints() == [] and list((run.path / 'checkpoints').iterdir()) == []

    def test_training_resumes_from_a_checkpoint(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]))
        for _ in range(2):
            runner.run(5, {'signal': SIGNAL})
        runner.checkpoint()
        expected = jax.tree.map(np.asarray, runner.state)
        runner.close()
        run = R.load(runner.recorder.path)
        resumed = R.Runner(run.restore(), R.Recorder.resume(run.path, brain))
        for got, want in zip(jax.tree.leaves(resumed.state), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(np.asarray(got), want)
        for _ in range(2):
            resumed.run(5, {'signal': SIGNAL})
        resumed.close()
        run = R.load(run.path)
        assert run.step == 20 and run.status == 'finished' and _recorded_t0(run, 'a') == [0, 5, 10, 15]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestReading:

    def test_records_by_tag_hold_the_episodes_recorded(self, brain, tmp_path) -> None:
        probes = (
            R.TraceProbe('first_pool.soma.potential'), R.TraceProbe('second_pool.soma.potential', stride=3),
            R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),
        )
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('m', probes, group='episode', raw=('obs',))]))
        calls = (3, 2, 4, 2, 3)
        for episode, count in enumerate(calls):
            runner.recorder.tag(episode=episode)
            if episode % 2 == 0:
                runner.recorder.record('m')
            for _ in range(count):
                runner.recorder.raw('obs', np.full(2, episode, np.float32))
                runner.run(4, {'signal': SIGNAL})
        runner.close()
        run = R.load(runner.recorder.path)
        episodes, flat = run.read('m'), run.timeline('m')
        starts = np.cumsum((0, *calls)) * 4
        assert list(episodes) == [0, 2, 4]
        assert repr(episodes[0]) == 'Record(0, t0=0, steps=12, 2 traces, 1 raw, 1 summaries)'
        assert '    first_pool.soma.potential' in str(episodes[0]) and '    first_pool.soma:spikes  active_fraction' in str(episodes[0])
        for number, episode in episodes.items():
            assert episode.key == number and episode.t0 == starts[number] and episode.steps == calls[number] * 4
            np.testing.assert_array_equal(episode.t, np.arange(calls[number] * 4))
            t, potential = episode.traces['first_pool.soma.potential']
            np.testing.assert_array_equal(t, episode.t)
            steps = flat['first_pool.soma.potential@trace#t']
            within = (steps >= episode.t0) & (steps < episode.t0 + episode.steps)
            np.testing.assert_array_equal(potential, flat['first_pool.soma.potential@trace'][within])
            assert np.all((episode.traces['second_pool.soma.potential'].t + episode.t0) % 3 == 0)
            t, frames = episode.raw['obs']
            np.testing.assert_array_equal(frames[:, 0], number)
            np.testing.assert_array_equal(t, np.arange(calls[number]) * 4)
            rate = episode.summaries['first_pool.soma:spikes']
            assert rate.active_fraction.shape == () and rate['active_fraction'] == rate.active_fraction
            assert rate.active_fraction == flat['first_pool.soma:spikes@summary#active_fraction'][list(episodes).index(number)]
            with pytest.raises(AttributeError, match='Recorded: active_fraction'):
                rate.mean

    def test_a_group_recorded_from_its_middle_starts_past_its_first_step(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('m', (R.TraceProbe('first_pool.soma.potential'),), group='episode')]))
        runner.recorder.tag(episode=0)
        for call in range(4):
            if call == 2:
                runner.recorder.record('m')
            runner.run(4, {'signal': SIGNAL})
        runner.recorder.tag(episode=1)
        runner.run(4, {'signal': SIGNAL})
        runner.close()
        episodes = R.load(runner.recorder.path).read('m')
        assert list(episodes) == [0] and episodes[0].t0 == 0
        np.testing.assert_array_equal(episodes[0].t, np.arange(8, 16))
        np.testing.assert_array_equal(episodes[0].traces['first_pool.soma.potential'].t, np.arange(8, 16))

    def test_records_by_groups_of_steps_and_by_stretches(self, brain, tmp_path) -> None:
        trace = (R.TraceProbe('first_pool.soma.potential'),)
        rate = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),)
        measurements = [
            R.Measurements('groups', trace + rate, group=6, trigger=R.Always()),
            R.Measurements('stretches', trace, trigger=R.At((0, 16), length=8)),
        ]
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, measurements))
        for _ in range(6):
            runner.run(4, {'signal': SIGNAL})
        runner.close()
        run = R.load(runner.recorder.path)
        groups = run.read('groups')
        assert list(groups) == [0, 1, 2, 3] and [g.t0 for g in groups.values()] == [0, 6, 12, 18]
        for group in groups.values():
            np.testing.assert_array_equal(group.traces['first_pool.soma.potential'].t, np.arange(6))
            assert group.summaries['first_pool.soma:spikes'].active_fraction.shape == ()
        stretches = run.read('stretches')
        assert list(stretches) == [0, 16] and [s.steps for s in stretches.values()] == [8, 8]

    def test_a_tag_coming_back_to_a_value_starts_another_record(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('m', (R.TraceProbe('first_pool.soma.potential'),), group='phase')]))
        for phase in ('a', 'b', 'a'):
            runner.recorder.tag(phase=phase)
            runner.recorder.record('m')
            runner.run(4, {'signal': SIGNAL})
        runner.close()
        records = R.load(runner.recorder.path).read('m')
        assert list(records) == ['a', 'b', ('a', 1)] and [r.t0 for r in records.values()] == [0, 4, 8]

    def test_snapshots_deltas_and_scalar_names(self, brain, tmp_path) -> None:
        probes = (
            R.SnapshotProbe('first_pool.synapses.kernel'), R.DeltaProbe('first_pool.synapses.kernel', reduce=('norm', 'full')),
            R.SummaryProbe('first_pool.synapses.kernel', reduce=('mean',)),
        )
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('w', probes, group=8, trigger=R.Always())]))
        for _ in range(4):
            runner.run(4, {'signal': SIGNAL})
        runner.close()
        run = R.load(runner.recorder.path)
        group = run.read('w')[1]
        kernel = run.timeline('w')['first_pool.synapses.kernel@snapshot'][1]
        np.testing.assert_array_equal(group.snapshots['first_pool.synapses.kernel'], kernel)
        change = group.deltas['first_pool.synapses.kernel']
        assert change.full.shape == kernel.shape and float(change.norm) == pytest.approx(float(np.linalg.norm(change.full.astype(np.float64))), rel=1e-4)
        # One address, three kinds, and scalar series named without the mode of the probe.
        assert set(run.scalar_keys()) == {'w/first_pool.synapses.kernel/norm', 'w/first_pool.synapses.kernel/mean'}
        t, norm = run.scalar('w/first_pool.synapses.kernel/norm')
        assert t.tolist() == [0, 8] and norm[1] == pytest.approx(float(change.norm))

    def test_rows_are_read_incrementally(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())])
        runner = R.Runner(brain, recorder)
        runner.run(5, {'signal': SIGNAL})
        recorder.flush()
        run = R.load(recorder.path)
        first = run.rows('scalars')
        runner.run(5, {'signal': SIGNAL})
        recorder.flush()
        more = run.rows('scalars', after=first[-1][0])
        assert first and more and more[0][0] > first[-1][0] and all(row[1] == 5 for row in more)
        last, grouped = run.scalar_rows(after=first[-1][0])
        assert last == more[-1][0] and sum(len(rows) for rows in grouped.values()) == len(more)
        assert run.refresh().step == 10
        with pytest.raises(ValueError, match='Unknown table'):
            run.rows('nope')
        runner.close()

    def test_a_long_series_reads_as_its_envelope(self, brain, tmp_path) -> None:
        values = np.sin(np.arange(1000) / 50.0)
        values[500], values[700] = 7.0, np.nan
        with R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Manual(), group=5)]) as recorder:
            for step, value in enumerate(values):
                recorder.log(x=value, step=step)
        run = R.load(recorder.path)
        steps, got = run.scalar('x')
        np.testing.assert_array_equal(steps, np.arange(1000))
        np.testing.assert_array_equal(got, values)
        steps, got = run.scalar('x', points=40)
        assert len(steps) <= 60 and np.all(np.diff(steps) >= 0)
        assert np.nanmax(got) == 7.0 and np.nanmin(got) == np.nanmin(values) and 700 in steps[np.isnan(got)]
        assert run.scalar_at('x', 500) == 7.0 and run.scalar_at('x', 650) == values[650] and np.isnan(run.scalar_at('x', 700))
        assert run.scalar_at('x', -1) is None and len(run.scalar('nope')[0]) == 0

    @pytest.mark.skipif(sys.platform == 'win32' or os.geteuid() == 0, reason='permissions of POSIX, not enforced for root')
    def test_runs_read_from_a_read_only_directory(self, brain, tmp_path) -> None:
        paths = []
        for _ in range(2):
            with R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]) as recorder:
                R.Runner(brain, recorder).run(5, {'signal': SIGNAL})
            paths.append(recorder.path)
        # The second run crashed: marked running, with an old heartbeat.
        info = json.loads((paths[1] / 'run.json').read_text())
        R.store.write_json(paths[1] / 'run.json', {**info, 'status': 'running', 'heartbeat': '2000-01-01T00:00:00+00:00'})
        assert sorted(p.name for p in paths[0].iterdir() if p.name.startswith('index')) == ['index.sqlite']
        items = [tmp_path, *tmp_path.rglob('*')]
        modes = {item: item.stat().st_mode for item in items}
        try:
            for item in items:
                item.chmod(0o555 if item.is_dir() else 0o444)
            finished, crashed = R.load(paths[0]), R.load(paths[1])
            assert finished.status == 'finished' and crashed.status == 'crashed'
            assert len(finished.scalar('a/first_pool.soma:spikes/active_fraction')[0]) == 1
            assert _recorded_t0(finished, 'a') == [0] and len(R.runs(tmp_path)) == 2
        finally:
            for item in reversed(items):
                item.chmod(modes[item])

    def test_the_version_of_the_index_is_checked(self, brain, tmp_path) -> None:
        with R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]) as recorder:
            pass
        connection = sqlite3.connect(recorder.path / 'index.sqlite')
        connection.execute('PRAGMA user_version = 99')
        connection.close()
        with pytest.raises(ValueError, match='version 99'):
            R.load(recorder.path)
        with pytest.raises(ValueError, match='version 99'):
            R.Recorder.resume(recorder.path, brain)
        assert not R.store.locked(recorder.path)

    def test_unreadable_runs_are_skipped_with_a_warning(self, brain, tmp_path) -> None:
        R.Recorder(tmp_path, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]).close()
        (tmp_path / 'broken').mkdir()
        (tmp_path / 'broken' / 'run.json').write_text('{')
        with pytest.warns(UserWarning, match='not a readable run'):
            assert len(R.runs(tmp_path)) == 1

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Store:
    """
        The key-value store of jax.distributed, in memory, shared by the recorders of processes simulated in
        this one.
    """
    data: dict[str, str] = {}

    def __init__(self) -> None:
        pass

    def publish(self, name, value) -> None:
        _Store.data[name] = json.dumps(value)

    def receive(self, name, seconds=None):
        if name.startswith('open/') and name not in _Store.data:
            return {'error': None}                                      # the simulated processes open one after another
        return json.loads(_Store.data[name])

    def listed(self, directory):
        return [json.loads(v) for k, v in _Store.data.items() if k.startswith(f'{directory}/')]

    def forget(self, name) -> None:
        for key in [k for k in _Store.data if k == name or k.startswith(f'{name}/')]:
            _Store.data.pop(key)

    def meet(self, name, seconds=None) -> None:
        pass

    def noticed(self) -> bool:
        return R.recorder.PREEMPTION_NOTICE in _Store.data

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestProcesses:
    """
        With several processes, process 0 writes the run and decides for all of them.
    """

    @pytest.fixture
    def processes(self, monkeypatch):
        """
            Simulated processes: sets the index of the process that recorders made next belong to.
        """
        _Store.data = {}
        monkeypatch.setattr(R.recorder, '_Sync', _Store)
        monkeypatch.setattr(jax, 'process_count', lambda: 2)
        index = [0]
        monkeypatch.setattr(jax, 'process_index', lambda: index[0])
        return index

    def test_process_zero_decides_for_every_process(self, brain, tmp_path, processes, monkeypatch) -> None:
        monkeypatch.setattr(R.SETTINGS, 'decision_block', 2)
        probe = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),)
        measurements = [R.Measurements('a', probe, trigger=R.Every(2, tag='episode'), group='episode'), R.Measurements('m', probe, trigger=R.Manual(), group='episode')]
        first = R.Recorder(tmp_path / 'runs', brain, measurements)
        processes[0] = 1
        other = R.Recorder(tmp_path / 'elsewhere', brain, measurements)
        assert other.path == first.path and not (tmp_path / 'elsewhere').exists()
        handed = []
        probes = first.probes
        first.probes = lambda steps=None: handed.append(probes(steps)) or handed[-1]
        runner = R.Runner(brain, first)
        for call in range(5):
            first.tag(episode=call)
            other.tag(episode=1)                                        # not read
            if call == 1:
                first.record('m')
            runner.run(3, {'signal': SIGNAL})
            assert other.probes(3) == handed[-1]
            other.push(R.Packed(np.zeros(0, np.uint8), (), handed[-1]) if handed[-1] else {}, 3)      # not written
        assert (other.step, other._calls) == (first.step, first._calls) == (15, 5)
        # Every process met after calls 1 and 3; the decisions of the block before the last are gone.
        decided = sorted(k for k in _Store.data if k.startswith('call/'))
        assert decided == ['call/1/2', 'call/1/3', 'call/2/4']
        first.close()
        other.close()
        run = R.load(first.path)
        assert _recorded_t0(run, 'a') == [0, 6, 12] and _recorded_t0(run, 'm') == [3]
        assert run.timeline('a')['group_t0'].tolist() == [0, 6, 12] and run.timeline('a')['group_steps'].tolist() == [3, 3, 3]
        # A resumed run continues at the same step on every process, with the measurements of process 0.
        processes[0] = 0
        resumed = R.Recorder.resume(first.path, brain)
        processes[0] = 1
        follower = R.Recorder.resume(tmp_path / 'not read', None)
        assert (follower.step, follower.path) == (15, first.path)
        assert [r.name for r in follower.measurements] == ['a', 'm']
        resumed.close()
        follower.close()

    @pytest.mark.parametrize('stop', ['writer', 'signal'])
    def test_every_process_stops_at_the_same_call(self, brain, tmp_path, processes, stop) -> None:
        measurements = [R.Measurements('a', R.presets.summary(brain), trigger=R.Manual(), group=5)]
        first = R.Recorder(tmp_path, brain, measurements)
        processes[0] = 1
        other = R.Recorder(tmp_path, brain, measurements)
        assert first.probes(3) == other.probes(3) == ()
        if stop == 'writer':
            first._writer.error = OSError('No space left on device')
            expected, raised = (RuntimeError, 'writer'), (RuntimeError, 'on process 0 failed: OSError: No space left')
        else:
            first._flag(signal.SIGTERM)
            expected, raised = (R.Preempted, 'SIGTERM'), (R.Preempted, 'SIGTERM')
        first.push({}, 3), other.push({}, 3)
        # Raised again, the same way, when asked again for the same call.
        for _ in range(2):
            with pytest.raises(expected[0], match=expected[1]):
                first.probes(3)
            with pytest.raises(raised[0], match=raised[1]):
                other.probes(3)
        first._writer.error = None
        first.close()
        other.close()

    @pytest.mark.parametrize('writer', ['working', 'failed'])
    def test_signals_of_other_processes_stop_every_process(self, brain, tmp_path, processes, monkeypatch, writer) -> None:
        # A SIGTERM noted by the preemption service of JAX, whose handler stays, and a signal of process 1.
        monkeypatch.setattr(R.recorder, '_preemption_service', lambda: True)
        monkeypatch.setattr(R.SETTINGS, 'signals_every', 0.0)
        handler = signal.getsignal(signal.SIGTERM)
        measurements = [R.Measurements('a', R.presets.summary(brain), trigger=R.Manual(), group=5)]
        options = dict(signals=(signal.SIGTERM, signal.SIGUSR1), on_error='continue')
        first = R.Recorder(tmp_path, brain, measurements, **options)
        processes[0] = 1
        other = R.Recorder(tmp_path, brain, measurements, **options)
        assert signal.getsignal(signal.SIGTERM) == handler
        if writer == 'failed':
            first._writer.error = OSError('No space left on device')
        for key, signum in [(R.recorder.PREEMPTION_NOTICE, signal.SIGTERM), ('signal/1', signal.SIGUSR1)]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                assert first.probes(3) == other.probes(3) == ()
            first.push({}, 3), other.push({}, 3)
            _Store.data[key] = json.dumps({'signal': int(signum)})
            with pytest.raises(R.Preempted, match=signal.Signals(signum).name):
                first.probes(3)
            with pytest.raises(R.Preempted, match=signal.Signals(signum).name):
                other.probes(3)
            first._signal = first._decided = other._decided = None
            _Store.data.pop(key)
        first._writer.error = None
        first.close()
        other.close()

    def test_a_process_resuming_the_run_process_zero_creates(self, brain, tmp_path, processes) -> None:
        # The pattern of a restarted job, where another process finds the run already created by process 0.
        measurements = [R.Measurements('a', R.presets.summary(brain), trigger=R.Every(3), group=5)]
        first = R.Recorder(tmp_path, brain, measurements, run_id='job')
        processes[0] = 1
        other = R.Recorder.resume(tmp_path / 'job', brain)
        assert other.path == first.path and [r.name for r in other.measurements] == ['a'] and other.measurements[0].trigger == R.Every(3)
        first.close()
        other.close()

    def test_a_failure_of_process_zero_while_opening_raises_everywhere(self, brain, tmp_path, processes) -> None:
        measurements = [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]
        with pytest.raises(RuntimeError, match='being written'):
            locked = R.Recorder(tmp_path, brain, measurements)
            R.Recorder.resume(locked.path, brain)
        processes[0] = 1
        with pytest.raises(RuntimeError, match='could not open the run: RuntimeError'):
            R.Recorder.resume(tmp_path / 'any', brain)
        locked.close()

    @staticmethod
    def _launch(mode, root):
        with socket.socket() as sock:
            sock.bind(('localhost', 0))
            port = sock.getsockname()[1]
        env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
        workers = [
            subprocess.Popen([sys.executable, '-c', _WORKER, mode, str(index), str(port), str(root)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for index in range(2)
        ]
        results = []
        try:
            for worker in workers:
                out, err = worker.communicate(timeout=300)
                lines = [line for line in out.splitlines() if line.startswith('{')]
                assert lines, err[-3000:]
                results.append((worker.returncode, json.loads(lines[-1]), err))
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
        return results

    def test_two_processes_record_one_run(self, tmp_path) -> None:
        results = self._launch('ok', tmp_path)
        assert [(code, result['raised'], result['chunk']) for code, result, _ in results] == [(0, None, 8)] * 2, results[0][2][-3000:]
        run = R.load(results[0][1]['path'])
        assert run.status == 'finished' and run.step == 64
        # Sharded weights are read whole; the episodes and the record of process 0 count.
        assert _recorded_t0(run, 'weights') == list(range(0, 64, 8))
        assert run.timeline('weights')['synapses.kernel@snapshot'].shape[1:] == (64, 32)
        assert _recorded_t0(run, 'summary') == [0, 16, 32, 48] and _recorded_t0(run, 'manual') == [16]
        assert run.checkpoints() == [48]
        # The sharded state is gathered whole into the checkpoint, as the snapshot at the end of its group.
        restored = run.restore()
        np.testing.assert_array_equal(np.asarray(restored.synapses.kernel.value), run.read('weights')[5].snapshots['synapses.kernel'])

    def test_a_full_disk_on_process_zero_stops_both(self, tmp_path) -> None:
        results = self._launch('fail', tmp_path)
        assert [(result['raised'], code != 0) for code, result, _ in results] == [('RuntimeError', True)] * 2
        assert results[0][1]['chunk'] == results[1][1]['chunk'] < 8
        assert R.load(results[0][1]['path']).status == 'failed'

    def test_a_recorder_on_process_zero_alone_is_reported(self, tmp_path) -> None:
        results = self._launch('alone', tmp_path)
        assert results[0][1]['raised'] == 'RuntimeError' and 'process 1 did not within' in results[0][1]['message']
        assert results[1][0] == 0

    @pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='SIGUSR1 is POSIX')
    @pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='SIGUSR1 is POSIX')
    @pytest.mark.parametrize('mode, signum', [
        ('signal1', getattr(signal, 'SIGUSR1', 0)), ('sigterm1', signal.SIGTERM), ('sigterm_user', signal.SIGTERM),
    ])
    def test_a_signal_to_another_process_stops_both(self, tmp_path, mode, signum) -> None:
        # SIGUSR1 reaches process 0 through the key-value store, SIGTERM through the preemption service of JAX,
        # whose sync point other code may use meanwhile.
        results = self._launch(mode, tmp_path)
        (code0, result0, _), (code1, result1, _) = results
        assert (code0, code1) == (128 + signum, 128 + signum) and result0['raised'] == result1['raised'] == 'Preempted'
        assert result0['chunk'] == result1['chunk'] and 3 <= result0['chunk'] < 8
        assert R.load(result0['path']).status == 'preempted'

    def test_a_signal_to_process_zero_stops_both(self, tmp_path) -> None:
        results = self._launch('preempt', tmp_path)
        assert [(code, result['raised'], result['chunk']) for code, result, _ in results] == [(128 + signal.SIGUSR1, 'Preempted', 4)] * 2
        assert R.load(results[0][1]['path']).status == 'preempted'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestGroups:
    """
        Summaries, snapshots and deltas give one record per group of steps, whatever the calls of the model.
    """

    KERNEL = 'first_pool.synapses.kernel'

    def _probes(self, trace_kernel=False):
        probes = [
            R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction', 'active_fraction_per_unit', 'inactive_unit_fraction')),
            R.SummaryProbe('first_pool.soma.potential', reduce=('mean', 'std', 'min', 'max', 'hist'), bins=8, range=(-80.0, 0.0)),
            R.DeltaProbe(self.KERNEL, reduce=('norm', 'mean_abs', 'full')),
            R.SnapshotProbe(self.KERNEL),
        ]
        return tuple(probes + [R.TraceProbe(self.KERNEL)] * trace_kernel)

    def _run(self, brain, root, calls, measurements, tags=None):
        recorder = R.Recorder(root, brain, measurements)
        runner = R.Runner(brain, recorder)
        for index, steps in enumerate(calls):
            if tags is not None:
                recorder.tag(**tags(index))
            runner.run(steps, {'signal': SIGNAL})
        runner.close()
        return R.load(recorder.path), runner

    def test_groups_do_not_depend_on_how_the_steps_are_split(self, brain, tmp_path) -> None:
        measurements = lambda: [R.Measurements('a', self._probes(), group=20, trigger=R.Always()), R.Measurements('t', (R.TraceProbe('first_pool.soma.potential', stride=3),), trigger=R.Always())]
        reference, _ = self._run(brain, tmp_path, [84], measurements())
        expected, rows = reference.timeline('a'), reference.timeline('t')
        assert expected['group_t0'].tolist() == [0, 20, 40, 60, 80] and expected['group_steps'].tolist() == [20, 20, 20, 20, 4]
        for calls in ([13, 50, 21], [7] * 12, [1] * 84, [19, 1, 64]):
            run, _ = self._run(brain, tmp_path, calls, measurements())
            got = run.timeline('a')
            assert got['group_t0'].tolist() == expected['group_t0'].tolist() and got['group_steps'].tolist() == expected['group_steps'].tolist()
            for key, value in expected.items():
                if key.startswith(('span_', 'group_')):
                    continue
                if key.endswith(('#mean', '#std', '#norm', '#mean_abs')):
                    np.testing.assert_allclose(got[key], value, rtol=1e-5, atol=1e-6, err_msg=key)
                else:
                    np.testing.assert_array_equal(got[key], value, err_msg=key)
            # Strides are aligned on the steps of the run.
            np.testing.assert_array_equal(run.timeline('t')['first_pool.soma.potential@trace#t'], np.arange(0, 84, 3))
            np.testing.assert_array_equal(run.timeline('t')['first_pool.soma.potential@trace'], rows['first_pool.soma.potential@trace'])
            assert run.scalar('a/first_pool.soma:spikes/active_fraction')[0].tolist() == [0, 20, 40, 60, 80]

    def test_groups_within_one_call_record_the_steps_that_end_them(self, brain, tmp_path) -> None:
        run, _ = self._run(brain, tmp_path, [20, 20], [R.Measurements('a', self._probes(trace_kernel=True), group=5, trigger=R.Always())])
        data = run.timeline('a')
        assert data['group_t0'].tolist() == list(range(0, 40, 5)) and set(data['group_steps'].tolist()) == {5}
        kernels = data[f'{self.KERNEL}@trace']
        np.testing.assert_array_equal(data[f'{self.KERNEL}@snapshot'], kernels[4::5])
        change = data[f'{self.KERNEL}@delta#full']
        np.testing.assert_array_equal(change[1:], (kernels[9::5].astype(np.float32) - kernels[4:-5:5].astype(np.float32)))

    def test_a_group_holding_a_step_asked_for_is_recorded_to_its_end(self, brain, tmp_path) -> None:
        run, runner = self._run(brain, tmp_path, [4] * 10, [R.Measurements('a', self._probes(), trigger=R.At((3,)), group=10)])
        data = run.timeline('a')
        # Recorded for step 3: the calls from steps 0 to 8, until group 0 ends; steps 10 and 11 are in no group.
        assert data['span_t0'].tolist() == [0, 4, 8] and data['group_t0'].tolist() == [0] and data['group_steps'].tolist() == [10]
        assert len(runner.recorder.warmup_sets(4)) == 2

    def test_a_group_recorded_from_its_middle_covers_the_steps_recorded(self, brain, tmp_path) -> None:
        run, _ = self._run(brain, tmp_path, [5] * 6, [R.Measurements('a', self._probes(), trigger=R.Between(5, 12), group=10)])
        data = run.timeline('a')
        assert data['span_t0'].tolist() == [5, 10, 15]
        assert data['group_t0'].tolist() == [5, 10] and data['group_steps'].tolist() == [5, 10]
        assert run.scalar('a/first_pool.soma:spikes/active_fraction')[0].tolist() == [5, 10]

    def test_groups_by_tag(self, brain, tmp_path) -> None:
        # Episodes of 2, 3 and 1 calls of 4 steps; the last one ends with the run.
        episodes = [0, 0, 1, 1, 1, 2]
        run, runner = self._run(brain, tmp_path, [4] * 6, [R.Measurements('a', self._probes(trace_kernel=True), group='episode', trigger=R.Always())],
                                tags=lambda index: {'episode': episodes[index]})
        data = run.timeline('a')
        assert data['group_t0'].tolist() == [0, 8, 20] and data['group_steps'].tolist() == [8, 12, 4]
        kernels = data[f'{self.KERNEL}@trace']
        np.testing.assert_array_equal(data[f'{self.KERNEL}@snapshot'], kernels[[7, 19, 23]])
        np.testing.assert_array_equal(data[f'{self.KERNEL}@snapshot'][-1], np.asarray(runner.model.first_pool.synapses.kernel.value))
        assert data['first_pool.soma:spikes@summary#active_fraction'].shape == (3,)

    def test_a_group_cut_short_by_the_end_of_the_run_is_written(self, brain, tmp_path) -> None:
        run, runner = self._run(brain, tmp_path, [7, 7], [R.Measurements('a', self._probes(), group=10, trigger=R.Always())])
        data = run.timeline('a')
        assert data['group_t0'].tolist() == [0, 10] and data['group_steps'].tolist() == [10, 4]
        np.testing.assert_array_equal(data[f'{self.KERNEL}@snapshot'][-1], np.asarray(runner.model.first_pool.synapses.kernel.value))
        assert data[f'{self.KERNEL}@delta#norm'].shape == (2,)

    def test_only_calls_crossing_the_end_of_a_group_run_another_program(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', self._probes(), group=20, trigger=R.Always())])
        runner, flags, start = R.Runner(brain, recorder), [], recorder.start
        recorder.start = lambda probes: flags.append(start(probes).split) or start(probes)
        for steps in (10,) * 4 + (15,) * 4:
            runner.run(steps, {'signal': SIGNAL})
        # Calls from steps 40, 55, 70 and 85: the second and third hold the end of a group.
        assert flags == [False] * 4 + [False, True, True, False]
        probes = recorder._variants[frozenset({'a'})]
        assert not R.reduce.may_split(probes, 0, 10) and R.reduce.may_split(probes, 0, 15) and not R.reduce.may_split(probes, 5, 5)
        runner.close()

    def test_warmup_compiles_the_calls_crossing_the_end_of_a_group(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', self._probes(), group=10, trigger=R.Always())]))
        runner.warmup(4, {'signal': SIGNAL})
        compiled = []
        listener = lambda name, duration, **_: compiled.append(name) if name == '/jax/core/compile/backend_compile_duration' else None
        jax.monitoring.register_event_duration_secs_listener(listener)
        try:
            for _ in range(8):
                runner.run(4, {'signal': SIGNAL})                       # groups end within the calls from 8, 16 and 28
            runner.recorder.flush()
        finally:
            jax.monitoring.unregister_event_duration_listener(listener)
        assert compiled == []
        runner.close()
        assert R.load(runner.recorder.path).timeline('a')['group_t0'].tolist() == [0, 10, 20, 30]

    def test_the_start_of_a_call_is_checked(self, brain) -> None:
        graph, state = spark.split((brain))
        probes = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',), group=5),)
        with pytest.raises(ValueError, match='phases'):
            RN._scan(graph, state, {'signal': spark.FloatArray(jnp.asarray(SIGNAL))}, steps=4, probes=probes, start=np.zeros(2, np.int32))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestPresets:

    @pytest.mark.parametrize('name', sorted(CASES))
    def test_default_measurements_run_on_every_model(self, name, tmp_path) -> None:
        model, held, per_step = CASES[name]()
        measurements = R.presets.default(model, summary_group=STEPS, activity_every=STEPS, activity_length=STEPS, weights_every=STEPS)
        assert {r.name for r in measurements} == {'summary', 'activity', 'weights'}
        runner = R.Runner(model, R.Recorder(tmp_path, model, measurements))
        for _ in range(2):
            runner.run(STEPS, jax.tree.map(np.asarray, held), jax.tree.map(np.asarray, per_step) or None)
        runner.close()
        run = R.load(runner.recorder.path)
        somas = [p.address for p in run.measurements['summary'].probes if p.address.endswith('.potential')]
        assert somas
        for address in somas:
            assert len(run.scalar(f'summary/{address}/mean')[0]) == 2
        activity = run.timeline('activity')
        for probe in run.measurements['activity'].probes:
            assert activity[probe.key].shape[0] == 2 * STEPS
            if probe.mode == 'raster':
                assert activity[probe.key].dtype == np.bool_
                assert probe.units is None or activity[probe.key].shape[1] == len(probe.units)

    def test_units_are_sampled_the_same_way_in_every_process(self) -> None:
        assert R.presets.sample_units('A_excitatory.soma.potential', 256, 8) == (7, 14, 15, 59, 72, 192, 196, 255)
        assert R.presets.sample_units('a', 8, 8) is None
        model, _, _ = CASES['cartpole']()
        units = {p.key: p.units for p in R.presets.activity(model, trace_units=16)}
        assert units['A_excitatory.soma.potential@trace'] == R.presets.sample_units('A_excitatory.soma.potential', 256, 16)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
