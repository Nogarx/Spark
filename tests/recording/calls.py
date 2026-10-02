#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import signal
import warnings
import threading
import concurrent.futures
from functools import partial
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import spark

from cases import CASES

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

R = spark.recording
SIGNAL = np.full((8,), 1.0, dtype=np.float16)
INPUTS = {'signal': spark.FloatArray(jnp.asarray(SIGNAL))}
PROBES = (
    R.TraceProbe('first_pool.soma.potential'),
    R.RasterProbe('first_pool.soma:spikes'),
    R.TraceProbe('second_pool.soma.potential', stride=3),
    R.TraceProbe('__call__:signal'),
    R.SummaryProbe('first_pool.soma.potential', reduce=('mean', 'std', 'min', 'max', 'hist'), range=(-80.0, 0.0)),
    R.SummaryProbe('second_pool.soma:spikes', reduce=('active_fraction', 'inactive_unit_fraction')),
    R.SnapshotProbe('first_pool.synapses.kernel'),
    R.DeltaProbe('first_pool.synapses.kernel', reduce=('full', 'norm', 'mean_abs')),
)

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

def _unroll(graph, state, steps, unroll=1, **inputs):
    """
        The function of the CartPole tutorial, with `spark.scan`.
    """
    def step_fn(state, _):
        model = spark.merge(graph, state)
        out = model(**inputs)
        _, state = spark.split((model))
        return state, out
    state, outs = spark.scan(step_fn, state, length=steps, unroll=unroll)
    return jax.tree.map(lambda x: x[-1], outs), state

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _unroll_jax(graph, state, steps, unroll=1, **inputs):
    """
        The same function, with `jax.lax.scan`.
    """
    def step_fn(state, _):
        model = spark.merge(graph, state)
        out = model(**inputs)
        _, state = spark.split((model))
        return state, out
    state, outs = jax.lax.scan(step_fn, state, length=steps, unroll=unroll)
    return jax.tree.map(lambda x: x[-1], outs), state

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _jit(fun):
    return spark.jit(fun, static_argnames=['steps', 'unroll'])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _same_leaves(first, second):
    return all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _in_thread(fun, *args, **kwargs):
    """
        Runs ``fun`` in a thread of its own, and returns its result or raises its exception.
    """
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(fun, *args, **kwargs).result()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestWithoutRecorder:

    def test_the_program_is_the_one_of_jax(self, brain) -> None:
        graph, state = spark.split((brain))
        with_spark = jax.make_jaxpr(partial(_unroll, steps=8, **INPUTS))(graph, state)
        with_jax = jax.make_jaxpr(partial(_unroll_jax, steps=8, **INPUTS))(graph, state)
        assert str(with_spark) == str(with_jax)

    def test_a_call_is_a_call_of_jax_jit(self, brain) -> None:
        graph, state = spark.split((brain))
        run, reference = _jit(_unroll), jax.jit(_unroll_jax, static_argnames=['steps', 'unroll'])
        out, after = run(graph, state, steps=8, **INPUTS)
        expected_out, expected = reference(graph, state, steps=8, **INPUTS)
        assert _same_leaves(after, expected) and _same_leaves(out, expected_out)
        assert run._recording is None and run._nnx is None

    def test_modules_as_arguments_are_updated_in_place(self, tmp_path) -> None:
        neuron, _, per_step = CASES['alif']()
        before = jax.tree.map(np.asarray, spark.split((neuron))[1])
        @spark.jit
        def call(model, inputs):
            return model(**inputs), model
        _, called = call(neuron, jax.tree.map(lambda a: a[0], per_step))
        assert called is neuron and not _same_leaves(spark.split((neuron))[1], before)
        # Not a call of an open recorder.
        with R.Recorder(tmp_path, neuron, [R.Measurements('a', (R.TraceProbe('soma.potential'),), trigger=R.Always())]) as recorder:
            call(neuron, jax.tree.map(lambda a: a[0], per_step))
            assert recorder.step == 0

    def test_decorator_forms(self) -> None:
        plain = spark.jit(lambda x: x + 1)
        options = spark.jit(static_argnames=['n'])(lambda x, n: x * n)
        assert float(plain(1.0)) == 2.0 and float(options(2.0, n=3)) == 6.0
        assert options.lower(2.0, n=3) is not None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestWithRecorder:

    def test_calls_record_as_the_runner_does(self, brain, tmp_path) -> None:
        measurements = lambda: [R.Measurements('all', PROBES, trigger=R.Every(3, length=2), group=10)]
        graph, state = spark.split((brain))
        run = _jit(_unroll)
        with R.Recorder(tmp_path, brain, measurements(), name='jit') as recorder:
            for _ in range(9):
                _, state = run(graph, state, steps=4, **INPUTS)
        by_jit = R.load(recorder.path)
        with R.Recorder(tmp_path, brain, measurements(), name='runner') as recorder:
            runner = R.Runner(brain, recorder, donate=False)
            for _ in range(9):
                runner.run(4, {'signal': SIGNAL})
        by_runner = R.load(recorder.path)
        assert by_jit.step == by_runner.step == 36
        first, second = by_jit.timeline('all'), by_runner.timeline('all')
        assert set(first) == set(second)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
        assert _same_leaves(state, runner.state)

    def test_a_call_recording_nothing_runs_the_program_without_recorder(self, brain, tmp_path) -> None:
        graph, state = spark.split((brain))
        run = _jit(_unroll)
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES[:2])]) as recorder:
            for _ in range(3):
                _, state = run(graph, state, steps=5, **INPUTS)
            assert recorder.step == 15 and run._recording.recorded._cache_size() == 0
            recorder.record('a', 5)
            _, state = run(graph, state, steps=5, **INPUTS)
            assert run._recording.recorded._cache_size() == 1
        assert [int(t) for t in R.load(recorder.path).timeline('a')['span_t0']] == [15]

    def test_the_static_arguments_give_the_steps(self, brain, tmp_path) -> None:
        graph, state = spark.split((brain))
        run = _jit(_unroll)
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES[:1], trigger=R.Always())]) as recorder:
            for steps in (4, 6, 4, 6):
                _, state = run(graph, state, steps=steps, **INPUTS)
            assert recorder.step == 20 and len(run._recording.learned) == 2
        np.testing.assert_array_equal(R.load(recorder.path).timeline('a')['first_pool.soma.potential@trace#t'], np.arange(20))

    def test_the_length_of_xs_gives_the_steps(self, tmp_path) -> None:
        neuron, _, per_step = CASES['alif']()
        graph, state = spark.split((neuron))
        @spark.jit
        def run(graph, state, xs):
            def step_fn(state, x):
                model = spark.merge(graph, state)
                model(**x)
                return spark.split((model))[1], None
            return spark.scan(step_fn, state, xs)[0]
        with R.Recorder(tmp_path, neuron, [R.Measurements('a', (R.TraceProbe('soma.potential'),), trigger=R.Always())]) as recorder:
            for length in (5, 8, 5):
                state = run(graph, state, jax.tree.map(lambda a: a[:length], per_step))
            assert recorder.step == 18 and run._recording.shaped
        assert R.load(recorder.path).timeline('a')['soma.potential@trace'].shape[0] == 18

    def test_a_function_running_no_scan_is_not_a_call_of_the_recorder(self, brain, tmp_path) -> None:
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES[:1], trigger=R.Always())]) as recorder:
            assert float(spark.jit(lambda x: x + 1)(1.0)) == 2.0
            assert recorder.step == 0

    def test_a_jit_within_another_is_part_of_it(self, brain, tmp_path) -> None:
        graph, state = spark.split((brain))
        inner = _jit(_unroll)
        @partial(spark.jit, static_argnames=['steps'])
        def outer(graph, state, steps, **inputs):
            out, state = inner(graph, state, steps=steps, **inputs)
            return out['action'].value.sum(), state
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES[:1], trigger=R.Always())]) as recorder:
            _, state = outer(graph, state, steps=4, **INPUTS)
            _, state = outer(graph, state, steps=4, **INPUTS)
            assert recorder.step == 8
        np.testing.assert_array_equal(R.load(recorder.path).timeline('a')['first_pool.soma.potential@trace#t'], np.arange(8))

    def test_donated_state(self, brain, tmp_path) -> None:
        graph, state = spark.split((brain))
        state = jax.tree.map(jnp.copy, state)
        run = spark.jit(_unroll, static_argnames=['steps', 'unroll'], donate_argnames=['state'])
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES, trigger=R.Every(2), group=4)]) as recorder:
            for _ in range(4):
                _, state = run(graph, state, steps=4, **INPUTS)
        assert R.load(recorder.path).step == 16

    def test_a_call_that_fails_is_not_counted(self, brain, tmp_path) -> None:
        graph, state = spark.split((brain))
        run = _jit(_unroll)
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES[:1], trigger=R.Always())]) as recorder:
            _, state = run(graph, state, steps=4, **INPUTS)
            recorded = run._recording.recorded
            def failing(*args, **kwargs):
                raise RuntimeError('failed')
            run._recording.recorded = failing
            with pytest.raises(RuntimeError, match='failed'):
                run(graph, state, steps=4, **INPUTS)
            run._recording.recorded = recorded
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                _, state = run(graph, state, steps=4, **INPUTS)
            assert not [w for w in caught if 'called again' in str(w.message)]
            assert recorder.step == 8

    def test_warmup_compiles_ahead(self, brain, tmp_path) -> None:
        graph, state = spark.split((brain))
        run = _jit(_unroll)
        measurements = [R.Measurements('a', R.presets.summary(brain), trigger=R.Every(14, length=7), group=7)]
        with R.Recorder(tmp_path, brain, measurements):
            assert run.warmup(graph, state, steps=7, **INPUTS) == 2
            compiled = []
            def listener(name, duration, **_):
                if name == '/jax/core/compile/backend_compile_duration':
                    compiled.append(name)
            jax.monitoring.register_event_duration_secs_listener(listener)
            try:
                for _ in range(3):
                    _, state = run(graph, state, steps=7, **INPUTS)
            finally:
                jax.monitoring.unregister_event_duration_listener(listener)
            assert compiled == []

    def test_warmup_needs_an_open_recorder(self, brain) -> None:
        graph, state = spark.split((brain))
        with pytest.raises(RuntimeError, match='no recorder is open'):
            _jit(_unroll).warmup(graph, state, steps=4, **INPUTS)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestCallsToTheOpenRecorder:
    """
        `spark.recording.log`, `event`, `tag`, `raw` and `record`, as a loop calls them.
    """

    @staticmethod
    def _episode(graph, state, run):
        # Written once, and run with a recorder open or without one.
        R.tag(episode=0)
        R.record('a')
        R.raw('env/observation', np.arange(4, dtype=np.float32))
        _, state = run(graph, state, steps=5, **INPUTS)
        R.log({'episode/steps': 5.0})
        R.event('episode_end', outcome='fell')
        return state

    def test_without_a_recorder_they_do_nothing(self, brain) -> None:
        graph, state = spark.split((brain))
        self._episode(graph, state, _jit(_unroll))

    def test_they_reach_the_open_recorder(self, brain, tmp_path) -> None:
        graph, state = spark.split((brain))
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES[:2], raw=('env/observation',))]) as recorder:
            self._episode(graph, state, _jit(_unroll))
        run = R.load(recorder.path)
        assert [tag['value'] for tag in run.tags('episode')] == [0]
        assert list(run.scalar('episode/steps')[1]) == [5.0]
        assert [event['outcome'] for event in run.events('episode_end')] == ['fell']
        (record,) = run.read('a').values()
        assert record.steps == 5
        np.testing.assert_array_equal(record.raw['env/observation'].values, [np.arange(4)])

    def test_a_raw_stream_the_recorder_does_not_declare_is_warned_of(self, brain, tmp_path) -> None:
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES[:2])]):
            with pytest.warns(R.RecordingWarning, match='No measurements declare the raw stream'):
                R.raw('env/observation', np.zeros(4))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestThreads:
    """
        The recorder the calls of a thread go to.
    """

    def test_a_recorder_opened_within_a_run_takes_the_calls_until_it_closes(self, brain, tmp_path) -> None:
        graph, state = spark.split((brain))
        run = _jit(_unroll)
        measurements = [R.Measurements('a', PROBES[:2])]
        with R.Recorder(tmp_path, brain, measurements, name='training') as training:
            R.record('a')
            _, state = run(graph, state, steps=5, **INPUTS)
            with R.Recorder(tmp_path, brain, measurements, name='evaluation') as evaluation:
                R.record('a')
                run(graph, state, steps=3, **INPUTS)
                R.log({'reward': 1.0})
            _, state = run(graph, state, steps=5, **INPUTS)
            R.log({'loss': 0.5})
        assert (training.step, evaluation.step) == (10, 3)
        training, evaluation = R.load(training.path), R.load(evaluation.path)
        assert (training.scalar_keys(), evaluation.scalar_keys()) == (['loss'], ['reward'])
        assert [record.steps for record in training.read('a').values()] == [5]
        assert [record.steps for record in evaluation.read('a').values()] == [3]

    def test_runs_trained_in_threads_are_recorded_apart(self, brain, tmp_path) -> None:
        graph, state = spark.split((brain))
        run = _jit(_unroll)
        # Both recorders are open while either trains.
        together = threading.Barrier(2, timeout=120)

        def train(index, calls):
            try:
                with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES[:2])], name=f'thread{index}') as recorder:
                    together.wait()
                    trained = state
                    for _ in range(calls):
                        R.record('a')
                        _, trained = run(graph, trained, steps=4, **INPUTS)
                        R.log({'index': float(index)})
                    together.wait()
            except BaseException:
                # The other thread stops waiting.
                together.abort()
                raise
            return recorder

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            recorders = list(pool.map(train, (0, 1), (3, 5)))
        for index, (recorder, calls) in enumerate(zip(recorders, (3, 5))):
            written = R.load(recorder.path)
            assert recorder.step == 4 * calls
            assert list(written.scalar('index')[1]) == [float(index)] * calls
            assert sum(record.steps for record in written.read('a').values()) == 4 * calls

    def test_a_loop_in_another_thread_uses_the_only_open_recorder(self, brain, tmp_path) -> None:
        graph, state = spark.split((brain))
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES[:2], raw=('env/observation',))]) as recorder:
            _in_thread(TestCallsToTheOpenRecorder._episode, graph, state, _jit(_unroll))
        assert recorder.step == 5
        (record,) = R.load(recorder.path).read('a').values()
        assert record.steps == 5

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestInterrupts:
    """
        Ctrl-C during a call: the recorder counts the calls whose result the loop received.
    """

    @pytest.fixture
    def setup(self, brain, tmp_path):
        graph, state = spark.split((brain))
        run = _jit(_unroll)
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES[:1], trigger=R.Always())])
        _, state = run(graph, state, steps=4, **INPUTS)
        return graph, state, run, recorder

    @staticmethod
    def _interrupting(function, times=1):
        def interrupted(*args, **kwargs):
            for _ in range(times):
                signal.raise_signal(signal.SIGINT)
            return function(*args, **kwargs)
        return interrupted

    def test_an_interrupt_during_a_call_is_raised_by_the_next_one(self, setup) -> None:
        graph, state, run, recorder = setup
        recorded = run._recording.recorded
        run._recording.recorded = self._interrupting(recorded)
        _, state = run(graph, state, steps=4, **INPUTS)
        run._recording.recorded = recorded
        assert recorder.step == 8
        with pytest.raises(KeyboardInterrupt):
            run(graph, state, steps=4, **INPUTS)
        assert recorder.step == 8
        _, state = run(graph, state, steps=4, **INPUTS)
        recorder.close()
        np.testing.assert_array_equal(R.load(recorder.path).timeline('a')['first_pool.soma.potential@trace#t'], np.arange(12))

    def test_an_interrupt_during_the_hand_over_is_raised_by_the_next_call_of_the_recorder(self, setup) -> None:
        graph, state, run, recorder = setup
        push = recorder.push
        recorder.push = self._interrupting(push)
        _, state = run(graph, state, steps=4, **INPUTS)
        recorder.push = push
        assert recorder.step == 8
        with pytest.raises(KeyboardInterrupt):
            recorder.log(value=1.0)
        recorder.log(value=1.0)
        recorder.close()

    def test_a_second_interrupt_raises_at_once(self, setup) -> None:
        graph, state, run, recorder = setup
        recorded = run._recording.recorded
        run._recording.recorded = self._interrupting(recorded, times=2)
        with pytest.raises(KeyboardInterrupt):
            run(graph, state, steps=4, **INPUTS)
        run._recording.recorded = recorded
        assert recorder.step == 4
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, state = run(graph, state, steps=4, **INPUTS)
        assert recorder.step == 8 and not [w for w in caught if 'called again' in str(w.message)]
        recorder.close()

    def test_closing_drops_a_held_interrupt(self, setup) -> None:
        graph, state, run, recorder = setup
        run._recording.recorded = self._interrupting(run._recording.recorded)
        run(graph, state, steps=4, **INPUTS)
        recorder.close()
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    def test_a_long_compilation_is_not_held(self, setup, monkeypatch) -> None:
        graph, state, run, recorder = setup
        compile_ = R.calls._Calls.compile
        def interrupted(calls, *args, **kwargs):
            signal.raise_signal(signal.SIGINT)
            return compile_(calls, *args, **kwargs)
        monkeypatch.setattr(R.calls._Calls, 'compile', interrupted)
        with pytest.raises(KeyboardInterrupt):
            run(graph, state, steps=6, **INPUTS)
        assert recorder.step == 4
        recorder.close()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestWhatIsRefused:

    @pytest.fixture
    def recorder(self, brain, tmp_path):
        return R.Recorder(tmp_path, brain, [R.Measurements('a', PROBES[:1], trigger=R.Always())])

    def test_the_model_called_outside_scan(self, brain, recorder) -> None:
        graph, state = spark.split((brain))
        @partial(spark.jit, static_argnames=['steps'])
        def loop(graph, state, steps, **inputs):
            model = spark.merge(graph, state)
            for _ in range(steps):
                out = model(**inputs)
            return out, spark.split((model))[1]
        with pytest.raises(RuntimeError, match='outside `spark.scan`'):
            loop(graph, state, steps=3, **INPUTS)
        assert recorder.step == 0

    def test_the_model_called_twice_per_step(self, brain, recorder) -> None:
        graph, state = spark.split((brain))
        @partial(spark.jit, static_argnames=['steps'])
        def twice(graph, state, steps, **inputs):
            def step_fn(state, _):
                model = spark.merge(graph, state)
                model(**inputs)
                model(**inputs)
                return spark.split((model))[1], None
            return spark.scan(step_fn, state, length=steps)[0]
        with pytest.raises(RuntimeError, match='2 times in one step'):
            twice(graph, state, steps=3, **INPUTS)

    def test_a_scan_within_another_transformation(self, brain, recorder) -> None:
        graph, state = spark.split((brain))
        @partial(spark.jit, static_argnames=['steps'])
        def nested(graph, state, steps, **inputs):
            def episode(state, _):
                return _unroll(graph, state, steps, **inputs)[1], None
            return jax.lax.scan(episode, state, length=2)[0]
        with pytest.raises(RuntimeError, match='another transformation'):
            nested(graph, state, steps=3, **INPUTS)

    def test_two_scans_in_one_call(self, brain, recorder) -> None:
        graph, state = spark.split((brain))
        @partial(spark.jit, static_argnames=['steps'])
        def two(graph, state, steps, **inputs):
            _, state = _unroll(graph, state, steps, **inputs)
            return _unroll(graph, state, steps, **inputs)
        with pytest.raises(RuntimeError, match='second `spark.scan`'):
            two(graph, state, steps=3, **INPUTS)

    def test_a_reversed_scan(self, brain, recorder) -> None:
        graph, state = spark.split((brain))
        @partial(spark.jit, static_argnames=['steps'])
        def reversed_(graph, state, steps, **inputs):
            def step_fn(state, _):
                model = spark.merge(graph, state)
                model(**inputs)
                return spark.split((model))[1], None
            return spark.scan(step_fn, state, length=steps, reverse=True)[0]
        with pytest.raises(ValueError, match='reverse'):
            reversed_(graph, state, steps=3, **INPUTS)

    def test_a_thread_that_opened_no_recorder_while_several_are_open(self, brain, recorder, tmp_path) -> None:
        other = R.Recorder(tmp_path, brain, [R.Measurements('b', PROBES[:1])])
        graph, state = spark.split((brain))
        with pytest.raises(RuntimeError, match='This thread opened no recorder, and 2 are open'):
            _in_thread(_jit(_unroll), graph, state, steps=3, **INPUTS)
        with pytest.raises(RuntimeError, match='This thread opened no recorder, and 2 are open'):
            _in_thread(R.log, {'x': 1.0})
        other.close()

    def test_a_probe_that_cannot_record_the_model(self, brain, tmp_path) -> None:
        graph, state = spark.split((brain))
        # Without the model, the recorder does not check its probes when it opens.
        with R.Recorder(tmp_path, measurements=[R.Measurements('a', (R.TraceProbe('first_pool:no_such_port'),))]) as recorder:
            with pytest.raises(RuntimeError, match='was not produced') as error:
                _jit(_unroll)(graph, state, steps=3, **INPUTS)
            assert 'before the first call' in str(error.value.__notes__)
            assert recorder.step == 0

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
