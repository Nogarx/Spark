#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import os
import sys
import json
import time
import signal
import sqlite3
import textwrap
import threading
import warnings
import subprocess
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
RN = spark.recording.runner
SIGNAL = np.full((8,), 1.0, dtype=np.float16)
HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = (R.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction',)),)
RATE = 'a/first_pool.soma:spikes/active_fraction'

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
        except BaseException:
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
    assert result.returncode == returncode, result.stderr[-3000:]
    return result.stdout.strip().splitlines()[-1]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _wait_for(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, 'timed out'
        time.sleep(0.05)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _slow_close(monkeypatch, seconds):
    close = R.recorder._Writer._close
    def slow(writer, *args):
        time.sleep(seconds)
        return close(writer, *args)
    monkeypatch.setattr(R.recorder._Writer, '_close', slow)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestIndex:
    """
        The index of a run through errors that pass, items that fail, and readers.
    """

    def test_an_error_that_passes_is_recovered_without_losing_or_repeating_rows(self, brain, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(R.SETTINGS, 'retry_after', (0.01, 0.01))
        commit, failures = R.store.commit, ['before', 'after']
        def flaky(connection, run_dir):
            # The first commit fails before it is written, the second after: neither loses or repeats rows.
            if failures and failures[0] == 'before':
                failures.pop(0)
                raise sqlite3.OperationalError('disk I/O error')
            commit(connection, run_dir)
            if failures and failures[0] == 'after':
                failures.pop(0)
                raise sqlite3.OperationalError('disk I/O error')
        with pytest.warns(UserWarning, match='opened again'):
            with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)]) as recorder:
                runner = R.Runner(brain, recorder)
                recorder.flush()
                monkeypatch.setattr(R.store, 'commit', flaky)
                for chunk in range(6):
                    runner.run(3, {'signal': SIGNAL})
                    recorder.log(loss=float(chunk))
                    recorder.flush()
        run = R.load(recorder.path)
        assert not failures and run.status == 'finished'
        assert run.scalar('loss')[1].tolist() == [float(c) for c in range(6)] and len(run.scalar(RATE)[0]) == 6
        assert _recorded_t0(run, 'a') == list(range(0, 18, 3))

    def test_rows_applied_before_an_error_between_commits_are_applied_again(self, brain, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(R.SETTINGS, 'retry_after', (0.01, 0.01))
        database, recorded = R.recorder._Writer._database, [False]
        def failing(writer, action):
            # Once rows wait for the next commit, an item fails on its first attempt, as on a page read.
            if recorded[0] and writer.uncommitted and writer.committing is None:
                recorded[0] = False
                attempts = []
                def once():
                    attempts.append(1)
                    if len(attempts) == 1:
                        raise sqlite3.OperationalError('disk I/O error')
                    return action()
                return database(writer, once)
            return database(writer, action)
        monkeypatch.setattr(R.recorder._Writer, '_database', failing)
        with pytest.warns(UserWarning, match='opened again'):
            with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Manual(), group=3)]) as recorder:
                recorder.flush()
                recorded[0] = True
                for step in range(8):
                    recorder.log({f'new/{step % 3}': float(step)}, step=step)
        run = R.load(recorder.path)
        assert not recorded[0] and sorted(run.scalar_keys()) == ['new/0', 'new/1', 'new/2']
        assert [run.scalar(f'new/{k}')[1].tolist() for k in range(3)] == [[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0]]

    def test_an_item_that_fails_leaves_none_of_its_rows(self, brain, tmp_path, monkeypatch) -> None:
        test = R.recorder._Writer._test_conditions
        def failing(writer, name, t, own):
            if t == 6:
                raise OSError('failed with the group half listed')
            return test(writer, name, t, own)
        monkeypatch.setattr(R.recorder._Writer, '_test_conditions', failing)
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)])
        runner = R.Runner(brain, recorder)
        for _ in range(3):
            runner.run(3, {'signal': SIGNAL})
        _wait_for(lambda: recorder._writer.error is not None)
        with pytest.raises(RuntimeError, match='writer'):
            recorder.close()
        run = R.load(recorder.path)
        # The groups before are listed with their scalars and written; the one that failed is in neither. The
        # steps of its call were listed before.
        assert [row[3] for row in run.rows('groups')] == [0, 3] and run.scalar(RATE)[0].tolist() == [0, 3]
        assert run.timeline('a')['group_t0'].tolist() == [0, 3] and [row[3] for row in run.rows('spans')] == [0, 3, 6]

    def test_a_reader_reads_in_pages(self, brain, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(R.SETTINGS, 'page_rows', 3)
        values = np.arange(20, dtype=np.float64)
        steps = [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 6, 7, 8]      # several rows of a step across pages
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Manual(), group=3)]) as recorder:
            for step, value in zip(steps, values):
                recorder.log(x=value, step=step)
                recorder.log(y=-value, step=step)
            for index in range(7):
                recorder.event('mark', step=index, index=index)
                recorder.tag(episode=index)
        run = R.load(recorder.path)
        t, got = run.scalar('x')
        assert t.tolist() == steps and got.tolist() == values.tolist()
        last, rows = run.scalar_rows(after=4, keys=['y'])
        assert last == 40 and rows['y'][:, 1].tolist() == (-values[2:]).tolist()
        assert [e['index'] for e in run.events('mark')] == list(range(7)) and len(run.tags('episode')) == 7
        assert [row[0] for row in run.rows('scalars', after=10)] == list(range(11, 41))

    def test_a_commit_a_process_left_unfinished_is_rolled_back_by_readers(self, brain, tmp_path) -> None:
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)]) as recorder:
            R.Runner(brain, recorder).run(3, {'signal': SIGNAL})
            recorder.log(x=1.0)
        # A process writing the index ends in the middle of a commit: its journal is left, and the database
        # holds part of the transaction.
        pid = int(_python(f'''
            import os, sqlite3
            connection = sqlite3.connect({str(recorder.path / 'index.sqlite')!r}, isolation_level=None)
            connection.execute('PRAGMA cache_size = 1')
            connection.execute('BEGIN')
            connection.executemany('INSERT INTO scalars VALUES (?, ?, ?, ?)', [(t, 1, 9.0, 0.0) for t in range(50_000)])
            print(os.getpid(), flush=True)
            os._exit(1)
        ''', returncode=1))
        assert (recorder.path / 'index.sqlite-journal').exists()
        # A run is rolled back by readers once its process is known to be gone.
        info = json.loads((recorder.path / 'run.json').read_text())
        R.store.write_json(recorder.path / 'run.json', {**info, 'status': 'running', 'process': {'host': R.store.process()['host'], 'pid': pid}})
        run = R.load(recorder.path)
        assert run.scalar('x')[1].tolist() == [1.0] and not (recorder.path / 'index.sqlite-journal').exists()

    def test_a_run_being_written_is_read_as_it_stands_rather_than_rolled_back(self, brain, tmp_path) -> None:
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)]) as recorder:
            recorder.log(x=1.0)
        info = json.loads((recorder.path / 'run.json').read_text())
        R.store.write_json(recorder.path / 'run.json', {**info, 'status': 'running', 'heartbeat': R.store.now(), 'process': R.store.process()})
        _python(f'''
            import os, sqlite3
            connection = sqlite3.connect({str(recorder.path / 'index.sqlite')!r}, isolation_level=None)
            connection.execute('PRAGMA cache_size = 1')
            connection.execute('BEGIN')
            connection.executemany('INSERT INTO events VALUES (?, ?, ?, ?)', [(t, 'x', '{{}}', 0.0) for t in range(50_000)])
            print('left', flush=True)
            os._exit(1)
        ''', returncode=1)
        # Where locks may not reach across hosts, a journal of a live run is not rolled back by a reader.
        with pytest.warns(UserWarning, match='as it stands'):
            R.load(recorder.path)
        assert (recorder.path / 'index.sqlite-journal').exists()

    def test_the_run_stays_where_it_was_created_whatever_the_working_directory(self, brain, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        recorder = R.Recorder('runs', brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)])
        runner = R.Runner(brain, recorder)
        runner.run(3, {'signal': SIGNAL})
        (tmp_path / 'elsewhere').mkdir()
        monkeypatch.chdir(tmp_path / 'elsewhere')
        runner.run(3, {'signal': SIGNAL})
        runner.close()
        assert recorder.path.is_absolute() and not (tmp_path / 'elsewhere' / 'runs').exists()
        assert _recorded_t0(R.load(recorder.path), 'a') == [0, 3]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestClosing:
    """
        Closing through interrupts, signals, and the end of the interpreter.
    """

    def test_an_interrupt_while_closing_waits_for_what_is_left(self, brain, tmp_path, monkeypatch) -> None:
        _slow_close(monkeypatch, 1.0)
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)])
        R.Runner(brain, recorder).run(3, {'signal': SIGNAL})
        interrupt = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT))
        interrupt.start()
        with pytest.warns(UserWarning, match='writing what is left'):
            with pytest.raises(KeyboardInterrupt):
                recorder.close()
        interrupt.join()
        recorder.close()                                                # closed already: nothing to do
        run = R.load(recorder.path)
        assert run.status == 'finished' and _recorded_t0(run, 'a') == [0] and not R.store.locked(recorder.path)

    def test_a_second_interrupt_stops_waiting_for_a_writer_that_hangs(self, brain, tmp_path, monkeypatch) -> None:
        _slow_close(monkeypatch, 3.0)
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)])
        R.Runner(brain, recorder).run(3, {'signal': SIGNAL})
        interrupts = [threading.Timer(delay, os.kill, (os.getpid(), signal.SIGINT)) for delay in (0.3, 0.6)]
        for interrupt in interrupts:
            interrupt.start()
        start = time.monotonic()
        with pytest.warns(UserWarning, match='Interrupt again'):
            with pytest.raises(KeyboardInterrupt):
                recorder.close()
        assert time.monotonic() - start < 2.0 and not recorder._finished
        for interrupt in interrupts:
            interrupt.join()
        recorder.close()                                                # waits for the writer this time
        assert R.load(recorder.path).status == 'finished' and not R.store.locked(recorder.path)

    def test_a_second_interrupt_stops_waiting_for_room_in_the_queue(self, brain, tmp_path, monkeypatch) -> None:
        handle = R.recorder._Writer._on_event
        monkeypatch.setattr(R.recorder._Writer, '_on_event', lambda writer, *args: (time.sleep(2.0), handle(writer, *args)))
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())], queue_size=1)
        recorder.event('slow')
        _wait_for(recorder._queue.empty)                                # taken by the writer
        recorder.event('slow')                                          # fills the queue
        interrupts = [threading.Timer(delay, os.kill, (os.getpid(), signal.SIGINT)) for delay in (0.3, 0.6)]
        for interrupt in interrupts:
            interrupt.start()
        start = time.monotonic()
        with pytest.warns(UserWarning, match='Interrupt again'):
            with pytest.raises(KeyboardInterrupt):
                recorder.close()
        assert time.monotonic() - start < 1.5 and recorder._stopped_waiting and not recorder._close_queued
        start = time.monotonic()
        R.recorder._close_at_exit()                                     # not waited for again at exit
        assert time.monotonic() - start < 0.5 and not recorder._finished
        for interrupt in interrupts:
            interrupt.join()
        recorder.close()                                                # waits for the writer this time
        assert R.load(recorder.path).status == 'finished' and not R.store.locked(recorder.path)

    def test_every_recorder_is_closed_before_a_signal_takes_effect_at_exit(self, brain, tmp_path, monkeypatch) -> None:
        for name in ('last_exc', 'last_value'):
            monkeypatch.delattr(sys, name, raising=False)
        handling = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())], signals=(signal.SIGINT,))
        other = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())])
        os.kill(os.getpid(), signal.SIGINT)                             # after the last chunk: raised once closed
        assert handling._signal == signal.SIGINT
        with pytest.raises(KeyboardInterrupt):
            R.recorder._close_at_exit()
        assert R.load(handling.path).status == 'preempted' and R.load(other.path).status == 'finished'
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    @pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='SIGUSR1 is POSIX')
    def test_a_handler_set_before_the_recorder_is_called_once(self, brain, tmp_path) -> None:
        received = []
        before = signal.signal(signal.SIGUSR1, lambda signum, frame: received.append(signum))
        try:
            with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())], signals=(signal.SIGUSR1,)) as recorder:
                R.Runner(brain, recorder).run(3, {'signal': SIGNAL})
                os.kill(os.getpid(), signal.SIGUSR1)                    # after the last chunk: not raised
            assert received == [signal.SIGUSR1] and R.load(recorder.path).status == 'preempted'
        finally:
            signal.signal(signal.SIGUSR1, before)

    @pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='SIGUSR1 is POSIX')
    def test_a_handler_set_over_the_recorder_may_pass_signals_on(self, brain, tmp_path) -> None:
        received = []
        original = signal.signal(signal.SIGUSR1, lambda signum, frame: received.append('original'))
        try:
            first = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())], signals=(signal.SIGUSR1,))
            ours = signal.getsignal(signal.SIGUSR1)
            def chained(signum, frame):                                 # as libraries pass signals to the handler they replaced
                received.append('chained')
                ours(signum, frame)
            signal.signal(signal.SIGUSR1, chained)
            first.close()
            os.kill(os.getpid(), signal.SIGUSR1)
            assert received == ['chained', 'original']
            second = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())], signals=(signal.SIGUSR1,))
            os.kill(os.getpid(), signal.SIGUSR1)
            assert received[2:] == ['chained', 'original'] and second._signal == signal.SIGUSR1
            second._signal = None
            second.close()
            assert signal.getsignal(signal.SIGUSR1) is chained
            os.kill(os.getpid(), signal.SIGUSR1)
            assert received[4:] == ['chained', 'original']
        finally:
            signal.signal(signal.SIGUSR1, original)
            R.recorder._SIGNALS.before.pop(signal.SIGUSR1, None)

    @pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='SIGUSR1 is POSIX')
    def test_a_second_signal_while_closing_is_held(self, brain, tmp_path, monkeypatch) -> None:
        _slow_close(monkeypatch, 1.0)
        with pytest.raises(R.Preempted):
            with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)], signals=(signal.SIGUSR1,)) as recorder:
                runner = R.Runner(brain, recorder)
                runner.run(3, {'signal': SIGNAL})
                os.kill(os.getpid(), signal.SIGUSR1)
                again = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGUSR1))
                again.start()
                runner.run(3, {'signal': SIGNAL})
        again.join()
        run = R.load(recorder.path)
        assert run.status == 'preempted' and _recorded_t0(run, 'a') == [0]
        assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL

    @pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='SIGUSR1 is POSIX')
    def test_recorders_share_the_handlers_of_their_signals(self, brain, tmp_path) -> None:
        received = []
        before = signal.signal(signal.SIGUSR1, lambda signum, frame: received.append(signum))
        try:
            first = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())], signals=(signal.SIGUSR1,))
            second = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())], signals=(signal.SIGUSR1,))
            os.kill(os.getpid(), signal.SIGUSR1)
            assert first._signal == second._signal == signal.SIGUSR1 and received == [signal.SIGUSR1]
            first._signal = second._signal = None
            first.close()                                               # closed in the order they were created
            second.close()
            os.kill(os.getpid(), signal.SIGUSR1)
            assert received == [signal.SIGUSR1] * 2 and first._signal is None
        finally:
            signal.signal(signal.SIGUSR1, before)

    @pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='SIGUSR1 is POSIX')
    def test_sigint_among_the_signals_stops_before_a_chunk(self, brain, tmp_path) -> None:
        with pytest.raises(R.Preempted):
            with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)], signals=(signal.SIGINT,)) as recorder:
                runner = R.Runner(brain, recorder)
                runner.run(3, {'signal': SIGNAL})
                os.kill(os.getpid(), signal.SIGINT)
                runner.run(3, {'signal': SIGNAL})
        assert R.load(recorder.path).status == 'preempted' and recorder.step == 3

    def test_signals_that_cannot_be_handled_are_refused(self, brain, tmp_path) -> None:
        if not hasattr(signal, 'SIGKILL'):
            pytest.skip('SIGKILL is POSIX')
        with pytest.raises(ValueError, match='cannot be handled'):
            R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())], signals=(signal.SIGKILL,))
        assert not list(tmp_path.iterdir())

    @pytest.mark.skipif(sys.platform == 'win32', reason='signals of POSIX')
    def test_a_signal_after_the_last_chunk_takes_effect_once_closed(self, tmp_path) -> None:
        path = _python(f'''
            import os, signal, sys
            sys.path.insert(0, {HERE!r})
            import numpy as np, spark
            from cases import CASES
            R = spark.recording
            brain, _, _ = CASES['brain']()
            with R.Recorder({str(tmp_path)!r}, brain, [R.Measurements('a', R.presets.summary(brain), trigger=R.Always(), group=5)], signals=(signal.SIGTERM,)) as recorder:
                R.Runner(brain, recorder).run(5, {{'signal': np.ones(8)}})
                print(recorder.path, flush=True)
                os.kill(os.getpid(), signal.SIGTERM)                    # after the last chunk: nothing raises it
            print('went on after SIGTERM', flush=True)
        ''', returncode=-signal.SIGTERM)
        run = R.load(path)
        assert run.status == 'preempted' and _recorded_t0(run, 'a') == [0]

    def test_a_checkpoint_at_the_end_of_a_script_is_written(self, tmp_path) -> None:
        path = _python(f'''
            import sys
            sys.path.insert(0, {HERE!r})
            import numpy as np, spark
            from cases import CASES
            R = spark.recording
            brain, _, _ = CASES['brain']()
            runner = R.Runner(brain, R.Recorder({str(tmp_path)!r}, brain, [R.Measurements('a', R.presets.summary(brain), group=5, trigger=R.Always())]))
            runner.run(5, {{'signal': np.ones(8)}})
            runner.checkpoint()                                         # written in the background, never closed
            print(runner.recorder.path, flush=True)
        ''')
        run = R.load(path)
        assert run.status == 'finished' and run.checkpoints() == [5]

    def test_interrupting_a_chunk_keeps_the_state_of_the_runner(self, brain, monkeypatch) -> None:
        runner = R.Runner(brain)
        runner.run(4, {'signal': SIGNAL})
        chunk = runner._fn
        def interrupted(*args, **kwargs):
            out = chunk(*args, **kwargs)
            os.kill(os.getpid(), signal.SIGINT)                         # after the state is donated
            time.sleep(0.05)
            return out
        runner._fn = interrupted
        with pytest.raises(KeyboardInterrupt):
            runner.run(4, {'signal': SIGNAL})
        runner._fn = chunk
        runner.run(4, {'signal': SIGNAL})
        assert np.isfinite(np.asarray(runner.model.first_pool.soma.potential.value)).all()

    def test_an_interrupt_during_a_chunk_is_raised_once_the_recorder_counted_it(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=4)]))
        runner.run(4, {'signal': SIGNAL})
        chunk = runner._fn
        def interrupted(*args, **kwargs):
            out = chunk(*args, **kwargs)
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)
            return out
        runner._fn = interrupted
        with pytest.raises(KeyboardInterrupt):
            runner.run(4, {'signal': SIGNAL})
        runner._fn = chunk
        assert (runner.recorder.step, runner.recorder._calls) == (8, 2)
        runner.close()
        assert _recorded_t0(R.load(runner.recorder.path), 'a') == [0, 4]

    def test_an_interrupt_while_the_records_are_handed_over_waits_until_they_are(self, brain, tmp_path, monkeypatch) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=4)]))
        runner.run(4, {'signal': SIGNAL})
        put = runner.recorder._put
        def interrupted(item):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)
            return put(item)
        monkeypatch.setattr(runner.recorder, '_put', interrupted)
        with pytest.raises(KeyboardInterrupt):
            runner.run(4, {'signal': SIGNAL})
        monkeypatch.undo()
        assert (runner.recorder.step, runner.recorder._calls) == (8, 2)
        runner.close()
        assert _recorded_t0(R.load(runner.recorder.path), 'a') == [0, 4]

    def test_a_second_interrupt_stops_waiting_for_room_for_a_chunk(self, brain, tmp_path, monkeypatch) -> None:
        handle = R.recorder._Writer._on_call
        monkeypatch.setattr(R.recorder._Writer, '_on_call', lambda writer, *args: (time.sleep(2.0), handle(writer, *args)))
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)], queue_size=1)
        runner = R.Runner(brain, recorder)
        runner.run(3, {'signal': SIGNAL})
        _wait_for(recorder._queue.empty)                                # taken by the writer
        runner.run(3, {'signal': SIGNAL})                               # fills the queue
        interrupts = [threading.Timer(delay, os.kill, (os.getpid(), signal.SIGINT)) for delay in (0.3, 0.6)]
        for interrupt in interrupts:
            interrupt.start()
        start = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            runner.run(3, {'signal': SIGNAL})
        assert time.monotonic() - start < 1.5 and (recorder.step, recorder._calls) == (9, 3)
        for interrupt in interrupts:
            interrupt.join()
        runner.close()

    def test_an_interrupt_while_a_chunk_compiles_stops_it_at_once(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=5)]))
        runner.run(4, {'signal': SIGNAL})
        chunk = runner._fn
        class Slow:
            def __call__(self, *args, **kwargs):
                return chunk(*args, **kwargs)
            def lower(self, *args, **kwargs):
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(1.0)                                         # tracing: interrupted here
                return chunk.lower(*args, **kwargs)
        runner._fn = Slow()
        start = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            runner.run(5, {'signal': SIGNAL})                           # a new number of steps compiles
        assert time.monotonic() - start < 0.5 and (runner.recorder.step, runner.recorder._calls) == (4, 1)
        runner._fn = chunk
        runner.run(5, {'signal': SIGNAL})
        runner.close()
        assert _recorded_t0(R.load(runner.recorder.path), 'a') == [0, 4]

    def test_chunks_compiled_by_warmup_are_not_compiled_again(self, brain, tmp_path) -> None:
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Every(2), group=3)]))
        runner.warmup(3, {'signal': SIGNAL})
        chunk, lowered = runner._fn, []
        class Counted:
            def __call__(self, *args, **kwargs):
                return chunk(*args, **kwargs)
            def lower(self, *args, **kwargs):
                lowered.append(kwargs['probes'])
                return chunk.lower(*args, **kwargs)
        runner._fn = Counted()
        for _ in range(4):
            runner.run(3, {'signal': SIGNAL})
        assert lowered == []
        runner.close()

    def test_an_interrupt_while_the_queue_is_full_keeps_the_counts(self, brain, tmp_path, monkeypatch) -> None:
        handle = R.recorder._Writer._on_call
        monkeypatch.setattr(R.recorder._Writer, '_on_call', lambda writer, *args: (time.sleep(0.5), handle(writer, *args)))
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)], queue_size=1)
        runner = R.Runner(brain, recorder)
        runner.run(3, {'signal': SIGNAL})
        runner.run(3, {'signal': SIGNAL})
        interrupt = threading.Timer(0.1, os.kill, (os.getpid(), signal.SIGINT))
        interrupt.start()
        with pytest.raises(KeyboardInterrupt):
            runner.run(3, {'signal': SIGNAL})                            # waits for room in the queue
        interrupt.join()
        assert recorder.step == 9 and recorder._calls == 3
        recorder.flush()
        assert recorder._queued_bytes == 0
        runner.run(3, {'signal': SIGNAL})
        runner.close()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRequests:

    def test_a_request_that_cannot_be_read_for_a_moment_is_read_later(self, brain, tmp_path, monkeypatch) -> None:
        read, failures = R.store.read_request, [2]
        def flaky(path):
            if failures[0]:
                failures[0] -= 1
                raise OSError(116, 'Stale file handle')
            return read(path)
        monkeypatch.setattr(R.store, 'read_request', flaky)
        runner = R.Runner(brain, R.Recorder(tmp_path, brain, [R.Measurements('m', PROBE, trigger=R.Manual(), group=3)]))
        runner.run(3, {'signal': SIGNAL})
        run = R.load(runner.recorder.path)
        run.record('m')
        _wait_for(lambda: run.requests()[0]['status'] == 'received')
        runner.run(3, {'signal': SIGNAL})
        runner.close()
        assert [r['status'] for r in R.load(runner.recorder.path).requests()] == ['applied'] and not failures[0]

    def test_requests_received_but_not_applied_expire_on_resume(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('m', PROBE, trigger=R.Manual(), group=3)])
        recorder.close()
        connection = sqlite3.connect(recorder.path / 'index.sqlite')
        connection.execute("INSERT INTO requests VALUES ('1-x', 'record', '{}', 0, 'received', 0)")
        connection.commit()
        connection.close()
        R.Recorder.resume(recorder.path, brain).close()
        assert [r['status'] for r in R.load(recorder.path).requests()] == ['expired']

    def test_a_request_written_while_the_run_is_resumed_is_kept(self, brain, tmp_path, monkeypatch) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('m', PROBE, trigger=R.Manual(), group=3)])
        recorder.close()
        early = R.store.write_request(recorder.path, 'record', {'measurements': 'm', 'steps': 1})
        listing = R.store.pending_requests
        def meanwhile(run_dir):
            found = listing(run_dir)
            if found and not (run_dir / 'requests' / 'late.json').exists():
                R.store.write_json(run_dir / 'requests' / 'late.json', {'kind': 'record', 'payload': {'measurements': 'm', 'steps': 1}})
            return found
        monkeypatch.setattr(R.store, 'pending_requests', meanwhile)
        resumed = R.Recorder.resume(recorder.path, brain)
        monkeypatch.setattr(R.store, 'pending_requests', listing)
        run = R.load(recorder.path)
        _wait_for(lambda: len(run.requests()) == 2 and run.requests()[1]['status'] == 'received')
        resumed.close()
        assert [(r['id'], r['status']) for r in R.load(recorder.path).requests()] == [(early, 'expired'), ('late', 'expired')]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestFailuresOfTheWriter:

    def test_the_progress_goes_on_after_the_writer_failed(self, brain, tmp_path, monkeypatch) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)], on_error='continue', heartbeat=0.2)
        runner = R.Runner(brain, recorder)
        runner.run(3, {'signal': SIGNAL})
        monkeypatch.setattr(R.recorder._Writer, '_on_event', lambda writer, *args: 1 / 0)
        recorder.event('boom')
        _wait_for(lambda: recorder._writer.error is not None)
        with pytest.warns(UserWarning, match='no longer recorded'):
            for _ in range(4):
                runner.run(3, {'signal': SIGNAL})
        runner.close()
        run = R.load(recorder.path)
        assert run.status == 'failed' and run.step == 15
        # Resuming goes on after the steps trained meanwhile.
        assert R.Recorder.resume(recorder.path, brain).step == 15

    def test_a_file_system_that_keeps_failing_does_not_stall_training(self, brain, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(R.SETTINGS, 'retry_after', (1.0, 1.0))
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3)], on_error='continue', heartbeat=0.1, queue_size=2)
        runner = R.Runner(brain, recorder)
        runner.run(3, {'signal': SIGNAL})
        def failing(path, data):
            raise OSError(5, 'Input/output error')
        # The failed status is written with retries once; the items after it are taken without waits.
        discarding, discard = threading.Event(), R.recorder._Writer._discard
        monkeypatch.setattr(R.recorder._Writer, '_discard', lambda writer: (discarding.set(), discard(writer)))
        monkeypatch.setattr(R.recorder._Writer, '_on_event', lambda writer, *args: 1 / 0)
        monkeypatch.setattr(R.store, 'write_json', failing)
        recorder.event('boom')
        assert discarding.wait(10.0)
        slowest = 0.0
        with pytest.warns(UserWarning, match='no longer recorded'):
            for _ in range(30):
                start = time.monotonic()
                runner.run(3, {'signal': SIGNAL})
                slowest = max(slowest, time.monotonic() - start)
                time.sleep(0.02)
        assert slowest < 0.5
        monkeypatch.undo()
        runner.close()

    def test_a_run_whose_creation_failed_leaves_nothing(self, brain, tmp_path, monkeypatch) -> None:
        def broken(run_dir):
            raise OSError('the file system failed while the run was created')
        monkeypatch.setattr(R.store, 'git_state', broken)
        with pytest.raises(OSError):
            R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())], run_id='job-7')
        assert not list(tmp_path.iterdir())
        monkeypatch.undo()
        with R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())], run_id='job-7') as recorder:
            pass
        assert recorder.path == tmp_path / 'job-7' and sorted(p.name for p in tmp_path.iterdir()) == ['job-7']

    @pytest.mark.skipif(sys.platform == 'win32' or os.geteuid() == 0, reason='permissions of POSIX, not enforced for root')
    def test_a_run_that_cannot_be_written_is_not_resumed(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())])
        recorder.close()
        (recorder.path / 'index.sqlite').chmod(0o444)
        try:
            with pytest.raises(PermissionError, match='cannot be written'):
                R.Recorder.resume(recorder.path, brain)
        finally:
            (recorder.path / 'index.sqlite').chmod(0o644)

    def test_a_lock_held_is_reported_with_the_heartbeat(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())])
        with pytest.raises(RuntimeError, match='last heartbeat .* s ago.*removing run.lock'):
            R.Recorder.resume(recorder.path, brain)
        recorder.close()

    def test_a_window_of_raw_alone_is_listed_on_resume(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Manual(), raw=('frame',), group=3)])
        recorder.close()
        # As when the process ended between writing a window of frames and listing it.
        R.store.write_arrays(recorder.path / 'windows' / 'a' / '000000.npz', {
            'span_t0': np.zeros(0, np.int64), 'span_steps': np.zeros(0, np.int64),
            'raw:frame': np.ones((2, 3)), 'raw:frame#t': np.array([4, 7]),
        })
        R.Recorder.resume(recorder.path, brain).close()
        (window,) = R.load(recorder.path).windows('a')
        assert (window.number, window.t0, window.t1, window.raw) == (0, 4, 7, ('frame',))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestResumeState:

    def test_step_and_tags_at_the_step_of_a_checkpoint(self, brain, tmp_path) -> None:
        measurements = [R.Measurements('a', PROBE, trigger=R.Every(2, tag='episode'), group=3), R.Measurements('b', PROBE, trigger=R.Every(6, length=3), group=3)]
        recorder = R.Recorder(tmp_path, brain, measurements)
        runner = R.Runner(brain, recorder)
        for episode in range(4):
            recorder.tag(episode=episode)
            runner.run(3, {'signal': SIGNAL})
            if episode < 2:
                runner.checkpoint()                                     # at steps 3 and 6
        runner.close()
        for step in (-1, True, 2.5):
            with pytest.raises(ValueError, match='non-negative integer'):
                R.Recorder.resume(recorder.path, brain, step=step)
        # The tags of the call that followed the step, and the last ones without a step.
        for step, (at, episode) in {None: (12, 3), 0: (0, 0), 3: (3, 1)}.items():
            resumed = R.Recorder.resume(recorder.path, brain, step=step)
            assert (resumed.step, resumed.tags) == (at, {'episode': episode})
            R.Runner(brain, resumed).run(3, {'signal': SIGNAL}) if step == 3 else None
            resumed.close()
        resumed = R.Recorder.resume(recorder.path, brain, step=6)
        runner = R.Runner(brain, resumed)
        runner.run(3, {'signal': SIGNAL})                                # step 6 and episode 2, recorded again
        runner.close()
        run = R.load(recorder.path)
        assert _recorded_t0(run, 'a') == _recorded_t0(run, 'b') == [0, 6, 6]
        assert run.timeline('a')['group_t0'].tolist() == [0, 6, 6]

    def test_runs_are_listed_by_the_time_they_were_created(self, brain, tmp_path) -> None:
        for name in ('zeta', 'alpha', 'job-10', 'job-9'):
            R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, group=3, trigger=R.Always())], run_id=name).close()
            time.sleep(1.1)
        assert [run.path.name for run in R.runs(tmp_path)] == ['zeta', 'alpha', 'job-10', 'job-9']
        # Created on hosts of other time zones: 08:00 and 09:00 UTC.
        for name, created in (('alpha', '2026-01-01T10:00:00+02:00'), ('zeta', '2026-01-01T09:00:00+00:00')):
            info = json.loads((tmp_path / name / 'run.json').read_text())
            R.store.write_json(tmp_path / name / 'run.json', {**info, 'created': created})
        assert [run.path.name for run in R.runs(tmp_path)][:2] == ['alpha', 'zeta']

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestChecks:
    """
        Mistakes of a day of use fail when they are made, with a message.
    """

    def test_push_takes_the_records_of_the_call_recorded(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), group=3), R.Measurements('b', R.presets.summary(brain), trigger=R.Manual(), group=3)])
        graph, state = spark.split((brain))
        chunk = jax.jit(lambda st, probes: RN._scan(graph, st, {'signal': spark.FloatArray(jnp.asarray(SIGNAL))}, steps=4, probes=probes)[2],
                        static_argnums=1)
        with pytest.raises(ValueError, match='without `probes`'):
            recorder.push(chunk(state, recorder._merged(frozenset({'a'}))), 4)
        for steps, wrong in ((4, recorder._merged(frozenset({'a', 'b'}))), (5, None)):
            probes = recorder.probes(4)
            records = chunk(state, wrong or probes)
            with pytest.raises(ValueError, match='not those of the probes' if wrong else '4 steps'):
                recorder.push(records, steps)
        for steps in (0, -3, 2.5, '4'):
            with pytest.raises(ValueError, match='positive integer'):
                recorder.push({}, steps)
        probes = recorder.probes(4)
        with pytest.raises(TypeError, match='as the call returned them'):
            recorder.push(chunk(state, probes).unpack(), 4)
        with pytest.raises(TypeError, match='takes payloads'):
            R.get_probe_targets(brain, {'signal': SIGNAL})
        recorder.probes(4)
        with pytest.warns(UserWarning, match='called again before `push`'):
            recorder.probes(4)
        recorder.close()

    def test_host_values_are_checked_when_given(self, brain, tmp_path) -> None:
        recorder = R.Recorder(tmp_path, brain, [R.Measurements('a', PROBE, trigger=R.Always(), raw=('frame',), group=3)])
        # Warned of once per cause, and dropped.
        for values in ({None: 1.0}, {1: 1.0}, {'': 1.0}):
            with pytest.warns(R.RecordingWarning, match='non-empty strings'):
                recorder.log(values)
        with pytest.warns(R.RecordingWarning, match='takes a number'):
            recorder.log({'x': None, 'y': 2.0})
        with pytest.warns(R.RecordingWarning, match='arrays of numbers'):
            recorder.raw('frame', None)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for frame in ({'x': 1}, [1, 'a']):
                recorder.raw('frame', frame)
        with pytest.warns(R.RecordingWarning, match='positive integer'):
            recorder.record('a', 0)
        with pytest.warns(R.RecordingWarning, match='No measurements "b"'):
            recorder.record('b')
        with pytest.warns(R.RecordingWarning, match='No measurements "c"'):
            assert not recorder.is_recording('c')
        with pytest.raises(ValueError, match='Unknown outputs'):
            R.Runner(brain, outputs='first')
        R.Runner(brain, recorder)
        with pytest.warns(UserWarning, match='has a runner already'):
            R.Runner(brain, recorder)
        recorder.close()
        with pytest.warns(R.RecordingWarning, match='recorded with warnings'):
            run = R.load(recorder.path)
        assert run.status == 'finished' and len(run.warnings()) == 8

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
