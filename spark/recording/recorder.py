#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
if tp.TYPE_CHECKING:
    from spark.nn.controllers.base import Controller

import os
import sys
import json
import math
import errno
import shutil
import time
import queue
import atexit
import signal
import sqlite3
import pathlib
import warnings
import importlib
import itertools
import threading
import collections
import concurrent.futures
import dataclasses as dc
import numpy as np
import jax
from spark.core.backend import split, merge
from spark.core.config import SparkConfig
from spark.core.recording_hooks import OPEN_RECORDERS
from spark.recording.probe import (
    Probe, SummaryProbe, TraceProbe, RasterProbe, SnapshotProbe, 
    DeltaProbe, SummaryReduction, DeltaReduction, BOUNDARY_PROBES, validate,
)
from spark.recording.reduce import (
    Packed, Start, merge_summaries, group_values, first_kept, 
    start_of, ends as group_ends, _summary_complete as complete_summary,
    _delta_complete as complete_delta,
)
from spark.recording.measurements import Measurements, merge_probes, scalar_key
from spark.recording.records import Record
from spark.recording.triggers import Every, At, Between, Always, When
from spark.recording.utils import whole, integer, deliver_signal, HeldInterrupt
from spark.recording.settings import SETTINGS
from spark.recording import store

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

PREEMPTION_NOTICE = 'RECEIVED_PREEMPTION_NOTICE'
"""
    Key the preemption service of ``jax.distributed`` sets when a process receives SIGTERM.
"""

RESERVED_EVENT_KEYS = ('t', 'kind', 'wall')
"""
    Keys of an event row, not accepted in its payload.
"""

ON_ERROR = ('raise', 'continue')
"""
    Accepted values of the ``on_error`` argument of `Recorder`.
"""

TRANSIENT_ERRORS = frozenset(getattr(errno, name) for name in ('EIO', 'ESTALE', 'EAGAIN', 'EBUSY', 'EINTR', 'ETIMEDOUT') if hasattr(errno, name))
"""
    Error numbers of file writes that are tried again.
"""

_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""
    Directory of the `spark` package. A warning points at the first frame of the stack outside it.
"""

_PROGRESS = 'INSERT INTO progress VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = MAX(value, excluded.value)'
"""
    Statement writing a row of the ``progress`` table of the index, which keeps the largest value.
"""

_SYNCS = itertools.count()
"""
    Numbers the recorders of the process, for the keys of their `_Sync`.
"""

_UNCATCHABLE = {getattr(signal, name) for name in ('SIGKILL', 'SIGSTOP') if hasattr(signal, name)}
"""
    Signals a process cannot handle.
"""

_OPEN: dict[Recorder, threading.Thread] = OPEN_RECORDERS
"""
    Recorders not closed yet, which `_close_at_exit` closes, in the order they were opened, with the
    thread that opened each. `spark.recording.current.open_recorder` picks among them the recorder the
    calls of a thread go to.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _on_host(leaf: tp.Any) -> tp.Any:
    """
        Returns a leaf of a state as a NumPy array, gathered from every process when it is spread over
        several. Collective when it is.
    """
    if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
        from jax.experimental import multihost_utils
        return multihost_utils.process_allgather(leaf, tiled=True)
    return jax.device_get(leaf)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _held(interrupt: KeyboardInterrupt, path: pathlib.Path | None) -> KeyboardInterrupt:
    """
        Warns that ``interrupt`` is raised once the writer is done, and returns it.
    """
    warnings.warn(
        f'The recorder of "{path}" is writing what is left of the run; the interrupt is raised once it is done. '
        f'Interrupt again to stop waiting: what is not written yet may be lost.'
    )
    return interrupt

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _model_name(config: SparkConfig | None) -> str | None:
    """
        Returns the class name of the model ``config`` configures, or None.
    """
    if config is None:
        return None
    try:
        return config.class_ref.__name__
    except Exception:
        return None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _heartbeat_age(path: pathlib.Path) -> str:
    """
        Describes the age of the last heartbeat of a run, as text.
    """
    try:
        age, period = store.heartbeat_age(store.read_info(path))
    except (OSError, ValueError):
        return 'no heartbeat read'
    return f'last heartbeat {age:.0f} s ago, every {period:g} s' if math.isfinite(age) else 'no heartbeat read'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _transient(error: sqlite3.Error) -> bool:
    """
        Returns whether an error of the index is transient, as network file systems give.
    """
    name = getattr(error, 'sqlite_errorname', '') or ''
    return name.startswith(('SQLITE_IOERR', 'SQLITE_CANTOPEN', 'SQLITE_PROTOCOL')) or any(
        text in str(error) for text in ('disk I/O error', 'unable to open database file')
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _retried(write: tp.Callable[..., None], *args: tp.Any) -> None:
    """
        Calls ``write``, and again after each delay of `SETTINGS.retry_after` while it fails.

        Only the errors of `TRANSIENT_ERRORS` are retried.
    """
    for delay in (*SETTINGS.retry_after, None):
        try:
            return write(*args)
        except OSError as error:
            if delay is None or error.errno not in TRANSIENT_ERRORS:
                raise
            time.sleep(delay)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RecordingWarning(UserWarning):
    """
        Warns of a mistake in what the loop hands over to a recorder. What it concerns is dropped,
        and the run goes on.

        Given by `Recorder.raw`, `Recorder.record`, `Recorder.log`, `Recorder.event` and
        `Recorder.is_recording`, and when measurements declaring a raw stream are recorded over a
        group, or a stretch of calls, without a frame of it. Each cause warns once per recorder, and
        is written to the run as a ``warning`` event, which `Run.warnings` lists.

        Notes
        -----
        A failure of the writer, such as a full disk, is not a warning: it follows the ``on_error``
        option of the recorder.

        Examples
        --------
        In a short run before a long one, every warning raises where it happens:

        >>> import warnings
        >>> warnings.simplefilter('error', spark.recording.RecordingWarning)

        From the command line: ``python -W error::spark.recording.RecordingWarning train.py``.
    """

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Preempted(SystemExit):
    """
        Exception raised by `Recorder.probes` after one of the signals the recorder handles.

        The call is not run. A recorder closed on it marks the run ``'preempted'``. Uncaught, it
        ends the process with exit code ``128 + signal``. With several processes, it is raised on
        every process at the same call.

        Parameters
        ----------
        signum : int
            The signal received.

        Attributes
        ----------
        signal : int
            The signal received.

        See Also
        --------
        Recorder : Records the measurements asked for and writes them to a run.
    """

    def __init__(self, signum: int) -> None:
        super().__init__(128 + int(signum))
        self.signal = int(signum)

    def __str__(self) -> str:
        return f'Stopped by {signal.Signals(self.signal).name}.'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True)
class _Call:
    """
        One call of the model, as handed to the writer.

        Holds the first step, the steps and the measurements recorded. ``slots`` gives the groups
        the call falls in, by measurements with a group, as ``(slot, group, first step, steps)``.
    """
    t0: int
    steps: int
    recorded: frozenset[str]
    slots: dict[str, tuple[tuple[int, tp.Any, int, int], ...]] = dc.field(default_factory=dict)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass
class _Open:
    """
        Group being recorded for a set of measurements, on process 0.

        ``key`` is the group number, or the first step of a group by tag. ``start`` and ``last``
        hold, by probe key, the value before the group and the last value after it, each as
        ``(array, row)`` of an array a call kept on the device.
    """
    key: tp.Any
    t0: int
    steps: int = 0
    start: dict[str, tp.Any] = dc.field(default_factory=dict)
    last: dict[str, tp.Any] = dc.field(default_factory=dict)
    tag: tp.Any = None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _same(first: tp.Any, second: tp.Any) -> bool:
    """
        Returns whether two values of a tag are the same once written to the run.
    """
    return store.to_json(first) == store.to_json(second)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _strip(records: Packed | dict | None) -> Packed | dict | None:
    """
        Returns the records of a call without the values kept on the device.
    """
    return Packed(records.buffer, records.layout, records.probes) if isinstance(records, Packed) else records

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _nbytes(records: tp.Any) -> int:
    """
        Returns the bytes of the arrays of ``records``, read from their shapes.

        Arrays on a device are not waited for.
    """
    return sum(int(getattr(leaf, 'nbytes', 0)) for leaf in jax.tree.leaves(records))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Window:
    """
        Records and raw frames of one set of measurements, buffered until written to one file.

        Holds the spans recorded, which need not be consecutive, the rows of the traces and rasters,
        one row per group, and the raw frames.
    """

    def __init__(self, measurements: Measurements, number: int) -> None:
        self.measurements = measurements
        self.number = number
        self.opened = time.monotonic()
        self.spans: list[tuple[int, int]] = []
        self.traces: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self.groups: list[tuple[int, int, dict[str, tp.Any]]] = []
        self.raw: dict[str, list[tuple[int, np.ndarray]]] = {}
        self.steps = 0
        self.nbytes = 0

    def empty(self) -> bool:
        return not self.spans and not self.groups and not self.raw

    def arrays(self) -> tuple[dict[str, np.ndarray], int, int]:
        """
            Returns the arrays of the file, its first step and the step after its last one.
        """
        arrays: dict[str, np.ndarray] = {}
        times = [(t, t + n) for t, n in self.spans] + [(t, t + n) for t, n, _ in self.groups]
        times += [(t, t) for frames in self.raw.values() for t, _ in frames]
        t0, t1 = min(t for t, _ in times), max(t for _, t in times)
        arrays['span_t0'] = np.array([t for t, _ in self.spans], dtype=np.int64)
        arrays['span_steps'] = np.array([n for _, n in self.spans], dtype=np.int64)
        for probe in self.measurements.probes:
            if isinstance(probe, (TraceProbe, RasterProbe)):
                parts = self.traces.get(probe.key)
                if not parts:
                    continue
                arrays[probe.key] = np.concatenate([rows for _, rows in parts], axis=0)
                arrays[f'{probe.key}#t'] = np.concatenate([times for times, _ in parts]).astype(np.int64)
                if isinstance(probe, RasterProbe):
                    arrays[f'{probe.key}#units'] = np.int64(arrays[probe.key].shape[-1])
                    arrays[probe.key] = np.packbits(arrays[probe.key], axis=-1)
            elif self.groups:
                values = [own.get(probe.key) for _, _, own in self.groups]
                if any(v is None for v in values):
                    # A snapshot or a delta of a group cut short, as by the end of the run: none for any group.
                    continue
                if isinstance(probe, SnapshotProbe):
                    arrays[probe.key] = np.stack(values)
                else:
                    for reduction in probe.reduce:
                        arrays[f'{probe.key}#{reduction}'] = np.stack([v[reduction] for v in values])
            # Releases the GIL to the main thread between probes.
            time.sleep(0)
        if self.groups:
            arrays['group_t0'] = np.array([t for t, _, _ in self.groups], dtype=np.int64)
            arrays['group_steps'] = np.array([n for _, n, _ in self.groups], dtype=np.int64)
        for name, frames in self.raw.items():
            arrays[f'raw:{name}'] = np.stack([frame for _, frame in frames])
            arrays[f'raw:{name}#t'] = np.array([t for t, _ in frames], dtype=np.int64)
        return arrays, t0, t1

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Writer(threading.Thread):
    """
        Thread that completes the records of a run and writes them.

        Once started, it is the only thread writing the files and the index of the run. Queue items
        are ``(kind, nbytes, *arguments)``, handled by ``_on_<kind>``, or by `_close` for
        ``'close'``. ``nbytes`` is released to the recorder once the item is handled.

        Notes
        -----
        The rows an item adds to the index are applied together, and an item that fails leaves none
        of them. The rows applied since the last commit are kept. After a transient error of the
        index (`_transient`), they are applied again on a new connection.
    """

    def __init__(self, recorder: Recorder, windows_from: dict[str, int]) -> None:
        super().__init__(name=f'spark-recorder-{recorder.path.name}', daemon=True)
        options = recorder._options
        self.path = recorder.path
        self.incoming: collections.deque = recorder._incoming
        self.notices: collections.deque = recorder._notices
        # Steps of the frames of each raw stream not yet checked, by measurements; the first step of the calls
        # recorded one after the other by measurements without a group; the streams warned of.
        self.frames: dict[str, dict[str, list[int]]] = {}
        self.stretches: dict[str, int] = {}
        self.unfed: set[tuple[str, str]] = set()
        # Tag of every scalar series, and the series warned of for being logged per another tag.
        self.key_tags: dict[str, str | None] = {}
        self.retagged: set[str] = set()
        self.watchers: dict[str, list[tuple[str, When]]] = recorder._watchers
        self.release = recorder._release
        # With several processes, signals received by the other processes are read with the requests.
        self.info = recorder._info
        self.measurements = {r.name: r for r in recorder.measurements}
        self.queue: queue.Queue = recorder._queue
        self.flush_steps: int | None = options.flush_steps
        self.flush_seconds: float = options.flush_seconds
        self.flush_bytes: int = options.flush_bytes
        self.heartbeat_every: float = options.heartbeat
        self.next_window = dict(windows_from)
        self.open: dict[str, _Window] = {}
        # Summary partials of the groups not ended yet, by measurements and group: those of every call by probe key,
        # merged when the group ends, the first step and the steps recorded.
        self.groups: dict[tuple[str, tp.Any], tuple[dict[str, list[dict[str, np.ndarray]]], int, int]] = {}
        self.error: BaseException | None = None
        self.done = threading.Event()
        self.progress = {'step': recorder.step, 'stalls': 0}
        self.connection: sqlite3.Connection | None = None
        self.keys: dict[str, int] = {}
        self.next_key = 1
        # Rows of the item being handled, and rows applied since the last commit, as (statement, rows).
        self.rows: list[tuple[str, list[tuple]]] = []
        self.uncommitted: list[tuple[str, list[tuple]]] = []
        # Commits so far, as `progress` counts them, and the number of the commit being written, if any.
        self.commits = 0
        self.committing: int | None = None
        self.opened = False
        self.last_poll = 0.0
        # Requests listed in the index, whose files are not read again, and failed reads of the others.
        self.handled: set[str] = set()
        self.unreadable: dict[str, int] = {}
        self.last_heartbeat = 0.0
        self.last_commit = time.monotonic()
        self.dirty = False

    # Loop.

    def run(self) -> None:
        closing = False
        try:
            self._database(self._reconnect)
            while True:
                try:
                    item = self.queue.get(timeout=min(self.heartbeat_every, SETTINGS.requests_every))
                except queue.Empty:
                    self._housekeeping(idle=True)
                    continue
                kind, nbytes, *arguments = item
                try:
                    if kind == 'close':
                        closing = True
                        self._close(*arguments)
                        return
                    getattr(self, f'_on_{kind}')(*arguments)
                    self._apply()
                    self._housekeeping()
                finally:
                    self.rows = []
                    self.release(nbytes)
                    self.queue.task_done()
        except BaseException as error:
            self.error = error
            self._salvage(error)
            if not closing:
                self._discard()
        finally:
            self.done.set()

    def _salvage(self, error: BaseException) -> None:
        """
            Writes what can still be written after a failure.

            Commits the index up to the last item handled, writes each open window on its own, and
            marks the run ``'failed'``.
        """
        self.rows = []
        for step in (self._commit, *(lambda name=name: self._flush(self.open.pop(name)) for name in list(self.open))):
            try:
                step()
            except Exception:
                pass
        try:
            if self.connection is not None:
                self.connection.close()
        except Exception:
            pass
        self._write_info(status='failed', error=f'writer: {type(error).__name__}: {error}')

    def _discard(self) -> None:
        """
            Takes the items left until ``'close'`` after a failure, without writing them.

            Keeps the progress of the run in ``run.json`` meanwhile. `Recorder._put` blocks while
            the queue is full.
        """
        while True:
            kind, nbytes, *arguments = self.queue.get()
            if kind in ('call', 'close'):
                self.progress.update(arguments[-1])
            self.release(nbytes)
            self.queue.task_done()
            # Once, without the waits of `SETTINGS.retry_after`: the queue is drained meanwhile.
            if kind == 'close' or time.monotonic() - self.last_heartbeat >= self.heartbeat_every:
                self._write_info(retry=False)
            if kind == 'close':
                return

    def _write_info(self, retry: bool = True, strict: bool = False, **fields: tp.Any) -> None:
        """
            Writes the progress and ``fields`` to ``run.json``, with a heartbeat.

            Without ``retry``, the write is tried once. Without ``strict``, an OSError is ignored.
        """
        self.info.update(self.progress, heartbeat=store.now(), heartbeat_every=self.heartbeat_every, **fields)
        try:
            (_retried if retry else lambda write, *args: write(*args))(store.write_json, self.path / 'run.json', self.info)
        except OSError:
            if strict:
                raise
        self.last_heartbeat = time.monotonic()

    def _housekeeping(self, idle: bool = False) -> None:
        if self.dirty and (idle or time.monotonic() - self.last_commit >= SETTINGS.commit_every):
            self._commit()
        self._flush_old()
        self._poll()
        self._heartbeat()

    # Index.

    def _add(self, statement: str, rows: list[tuple]) -> None:
        """
            Adds rows of the item being handled, applied together by `_apply`.
        """
        if rows:
            self.rows.append((statement, rows))

    def _apply(self) -> None:
        """
            Applies the rows of the item being handled, all of them or, after an error, none.
        """
        rows, self.rows = self.rows, []
        if not rows:
            return
        def apply() -> None:
            if not self.connection.in_transaction:
                self.connection.execute('BEGIN')
            self.connection.execute('SAVEPOINT item')
            try:
                for statement, values in rows:
                    self.connection.executemany(statement, values)
            except BaseException:
                try:
                    self.connection.execute('ROLLBACK TO item')
                except sqlite3.Error:
                    pass
                raise
            finally:
                try:
                    self.connection.execute('RELEASE item')
                except sqlite3.Error:
                    pass
        self._database(apply)
        self.uncommitted.extend(rows)
        self.dirty = True

    def _commit(self) -> None:
        """
            Writes the progress to the index and commits it.

            While another process holds the index, such as a reader in a long query, the commit
            waits, with a warning every `SETTINGS.busy_timeout` seconds. The queue of the recorder
            fills meanwhile.
        """
        self.commits += 1
        self._add(_PROGRESS, [('step', int(self.progress['step'])), ('commits', self.commits)])
        self._apply()
        self.committing = self.commits
        self._database(lambda: store.commit(self.connection, self.path))
        self.committing = None
        self.uncommitted = []
        self.dirty = False
        self.last_commit = time.monotonic()

    def _database(self, action: tp.Callable[[], tp.Any]) -> tp.Any:
        """
            Runs ``action`` on the index.

            After a transient error (`_transient`), waits the next delay of `SETTINGS.retry_after`,
            opens the index again and runs ``action`` again, with a warning.
        """
        delays = SETTINGS.retry_after
        for attempt, delay in enumerate((0.0, *delays)):
            try:
                if attempt:
                    time.sleep(delay)
                    self._reconnect()
                return action()
            except sqlite3.OperationalError as error:
                if attempt == len(delays) or not _transient(error):
                    raise
                warnings.warn(f'The index of "{self.path.name}" failed ({error}); it is opened again.')

    def _reconnect(self) -> None:
        """
            Opens the index and applies again the rows applied since the last commit.

            The rows are not applied again when the commit that failed was written after all. The
            first opening reads the keys, the requests listed and the number of commits instead.
        """
        if self.connection is not None:
            try:
                self.connection.close()
            except sqlite3.Error:
                pass
            self.connection = None
        self.connection = store.connect(self.path)
        if not self.opened:
            self.opened = True
            rows = self.connection.execute('SELECT name, id, tag FROM keys').fetchall()
            self.keys = {name: key for name, key, _ in rows}
            self.key_tags = {name: tag for name, _, tag in rows}
            self.next_key = max(self.keys.values(), default=0) + 1
            self.handled = {request for (request,) in self.connection.execute('SELECT id FROM requests')}
            self.commits = int((self.connection.execute("SELECT value FROM progress WHERE key = 'commits'").fetchone() or (0,))[0])
            return
        written = self.connection.execute("SELECT value FROM progress WHERE key = 'commits'").fetchone()
        if self.committing is not None and written is not None and int(written[0]) >= self.committing:
            # The commit that failed was written after all.
            self.uncommitted = []
            return
        self.connection.execute('BEGIN')
        for statement, rows in self.uncommitted:
            self.connection.executemany(statement, rows)

    def _key(self, name: str, tag: str | None = None) -> int:
        """
            Returns the id of the scalar series ``name``, added to the index when new with the tag
            it is written per, None for steps.

            A series keeps its first tag; another is warned of, once per series.
        """
        key = self.keys.get(name)
        if key is None:
            key = self.keys[name] = self.next_key
            self.key_tags[name] = tag
            self.next_key += 1
            self._add('INSERT INTO keys VALUES (?, ?, ?)', [(key, name, tag)])
        elif tag != self.key_tags.get(name) and name not in self.retagged:
            self.retagged.add(name)
            first = self.key_tags.get(name)
            message = (
                f'The scalar "{name}" was logged per {f"the tag {first!r}" if first else "step"}, and is now logged per '
                f'{f"the tag {tag!r}" if tag else "step"}; it stays per {f"the tag {first!r}" if first else "step"}.'
            )
            self._on_event(int(self.progress['step']), 'warning', {'message': message, 'scalar': name}, time.time())
            self.notices.append(message)
        return key

    # Items.

    def _on_call(self, records: Packed | dict | None, call: _Call, progress: dict[str, int]) -> None:
        """
            Lists the steps a call recorded and keeps the rows of its traces and rasters.

            Its summaries are merged into their groups. The traces and rasters of measurements
            without a group are tested by the conditions watching them.
        """
        self.progress.update(progress)
        # A stretch of calls of measurements without a group ends with the first call not recording them.
        for name in sorted(set(self.stretches) - call.recorded):
            self._check_frames(name, self.stretches.pop(name), call.t0)
        for name in call.recorded:
            if self.measurements[name].group is None and self.measurements[name].raw:
                self.stretches.setdefault(name, call.t0)
        if not call.recorded:
            return
        unpacked = records.unpack(completed=False) if isinstance(records, Packed) else {}
        kept = []
        for name in sorted(call.recorded):
            measurements = self.measurements[name]
            window = self.open.get(name)
            if window is not None and self.flush_steps is not None and window.steps >= self.flush_steps:
                # Written once the groups the calls before it ended are in it.
                self._flush(self.open.pop(name))
            window = self._window(name)
            # Listed at once; the data follows when the window is written.
            self._add('INSERT INTO spans VALUES (?, ?, ?, ?)', [(name, window.number, call.t0, call.steps)])
            traces = {}
            for probe in measurements.probes:
                if not isinstance(probe, (TraceProbe, RasterProbe)):
                    continue
                first = first_kept(probe.stride, call.t0 % probe.stride)
                steps = np.arange(first, call.steps, probe.stride)
                traces[probe.key] = (call.t0 + steps, np.array(unpacked[probe.key][:len(steps)], copy=True))
            self._merge(name, call, unpacked)
            if measurements.group is None and traces:
                arrays = {k: v for key, (t, rows) in traces.items() for k, v in ((key, rows), (f'{key}#t', t - call.t0))}
                self._test_conditions(name, call.t0, lambda: Record.from_arrays(call.t0, call.t0, np.arange(call.steps), arrays))
            kept.append((name, window, traces))
            time.sleep(0)
        # Added to the windows once the whole call is handled; its rows are applied before a window is written.
        for name, window, traces in kept:
            window.spans.append((call.t0, call.steps))
            window.steps += call.steps
            for key, part in traces.items():
                window.traces.setdefault(key, []).append(part)
                window.nbytes += part[1].nbytes
            self._flush_full(name, window)

    def _merge(self, name: str, call: _Call, unpacked: dict[str, tp.Any]) -> None:
        """
            Merges the summaries of a call into the groups of the measurements ``name``.
        """
        summaries = [p for p in self.measurements[name].probes if isinstance(p, SummaryProbe)]
        for slot, key, t0, steps in call.slots.get(name, ()):
            partials, first, recorded = self.groups.get((name, key), ({}, t0, 0))
            for probe in summaries:
                # A summary of a call crossing the end of a group has a leading axis of groups. Views of the
                # buffer of the call, merged when the group ends.
                part = {k: (v[slot] if np.ndim(unpacked[probe.key]['n']) else v) for k, v in unpacked[probe.key].items()}
                partials.setdefault(probe.key, []).append(part)
            self.groups[(name, key)] = (partials, min(first, t0), recorded + steps)

    def _on_group(self, name: str, key: tp.Any, t0: int | None, steps: int | None, boundary: dict[str, dict[str, tp.Any]]) -> None:
        """
            Writes one group of the measurements ``name``, of ``steps`` steps from step ``t0``.

            The group holds the reductions of its summaries, and its snapshots and deltas from
            ``boundary``, by probe key, as ``{'snapshot': value}`` or ``{'delta': partials}``.
            Without ``t0``, the steps are those the summaries were merged over.
        """
        measurements = self.measurements[name]
        partials, first, recorded = self.groups.pop((name, key), ({}, None, 0))
        if t0 is None:
            t0, steps = first, recorded
        self._check_frames(name, t0, None if t0 is None else t0 + steps)
        own = {}
        for probe in measurements.probes:
            if isinstance(probe, SummaryProbe) and probe.key in partials:
                own[probe.key] = complete_summary(probe, merge_summaries(partials[probe.key]))
            elif probe.key in boundary and isinstance(probe, SnapshotProbe):
                own[probe.key] = np.array(boundary[probe.key]['snapshot'], copy=True)
            elif probe.key in boundary and isinstance(probe, DeltaProbe):
                own[probe.key] = complete_delta(probe, {k: np.asarray(v) for k, v in boundary[probe.key]['delta'].items()})
        if t0 is None or not own:
            return
        window = self._window(name)
        self._add('INSERT INTO groups VALUES (?, ?, ?, ?)', [(name, window.number, t0, steps)])
        wall = time.time()
        # Summaries are per the tag their measurements are grouped by, or per step.
        tag = measurements.group if isinstance(measurements.group, str) else None
        scalars = []
        for probe in measurements.probes:
            value = own.get(probe.key)
            if isinstance(value, dict):
                reductions = SummaryReduction if isinstance(probe, SummaryProbe) else DeltaReduction
                for reduction in probe.reduce:
                    if reductions(reduction).scalar:
                        scalars.append((t0, self._key(scalar_key(name, probe.key, reduction), tag), float(value[reduction]), wall))
        self._add('INSERT INTO scalars VALUES (?, ?, ?, ?)', scalars)
        def record() -> Record:
            arrays = {}
            for probe_key, value in own.items():
                if isinstance(value, dict):
                    arrays.update({f'{probe_key}#{reduction}': part for reduction, part in value.items()})
                else:
                    arrays[probe_key] = value
            return Record.from_arrays(key, t0, np.arange(steps), arrays)
        self._test_conditions(name, t0, record)
        window.groups.append((t0, steps, own))
        window.nbytes += _nbytes(own)
        self._flush_full(name, window)

    def _flush_full(self, name: str, window: _Window) -> None:
        """
            Writes ``window`` once it holds ``flush_bytes`` bytes.

            A window holding ``flush_steps`` steps is written by `_on_call`, when the next call of
            its measurements arrives.
        """
        if window.nbytes >= self.flush_bytes:
            self._flush(self.open.pop(name))

    def _test_conditions(self, name: str, t: int, record: tp.Callable[[], Record]) -> None:
        """
            Tests the `When` conditions watching ``name`` on a record starting at step ``t``.

            ``record`` builds the record, once, when a condition watches ``name``.

            The measurements of the conditions that hold are handed to the recorder. A condition
            that raises is removed, with an ``error`` event and a warning.
        """
        watchers = list(self.watchers.get(name, ()))
        built = record() if watchers else None
        for target, trigger in watchers:
            try:
                met = bool(trigger.condition(built))
            except Exception as error:
                # A condition that raises is removed from the watchers.
                self.watchers[name].remove((target, trigger))
                message = f'{type(error).__name__}: {error}'
                self._on_event(t, 'error', {'in': 'condition', 'measurements': target, 'watch': name, 'error': message}, time.time())
                warnings.warn(f'The condition recording "{target}" failed and is no longer tested: {message}')
                continue
            if met:
                self.incoming.append((target, trigger.length, {'by': 'condition', 'watch': name, 'at': t}))

    def _on_scalars(self, t: int, values: dict[str, float], wall: float, tag: str | None = None) -> None:
        self._add('INSERT INTO scalars VALUES (?, ?, ?, ?)', [(t, self._key(k, tag), v, wall) for k, v in values.items()])

    def _on_event(self, t: int, kind: str, payload: dict, wall: float) -> None:
        self._add('INSERT INTO events VALUES (?, ?, ?, ?)', [(t, kind, store.to_json(payload), wall)])

    def _on_tags(self, t: int, tags: dict, wall: float) -> None:
        self._add('INSERT INTO tags VALUES (?, ?, ?, ?)', [(t, k, store.to_json(v), wall) for k, v in tags.items()])

    def _check_frames(self, name: str, t0: int | None, t1: int | None) -> None:
        """
            Warns of the raw streams of the measurements ``name`` without a frame in the steps
            ``[t0, t1)`` they recorded, a group or a stretch of calls, once per stream.

            The frames before ``t1`` are then forgotten. Without ``t0`` or ``t1``, the steps are
            unbounded on that side.
        """
        low, high = -math.inf if t0 is None else t0, math.inf if t1 is None else t1
        frames = self.frames.get(name, {})
        for stream in self.measurements[name].raw:
            steps = frames.get(stream, [])
            fed = any(low <= t < high for t in steps)
            frames[stream] = [t for t in steps if t >= high]
            if fed or (name, stream) in self.unfed:
                continue
            self.unfed.add((name, stream))
            at = int(t0) if t0 is not None else int(self.progress['step'])
            message = (
                f'The measurements "{name}" were recorded from step {at} without a frame of their raw stream "{stream}". '
                f'Check the name given to `raw`.'
            )
            self._on_event(at, 'warning', {'message': message, 'measurements': name, 'raw': stream}, time.time())
            self.notices.append(message)

    def _on_raw(self, t: int, name: str, frame: np.ndarray, measurements: tuple[str, ...]) -> None:
        for measurements in measurements:
            self.frames.setdefault(measurements, {}).setdefault(name, []).append(t)
            window = self._window(measurements)
            window.raw.setdefault(name, []).append((t, frame))
            window.nbytes += frame.nbytes
            self._flush_full(measurements, window)

    def _on_applied(self, request: str) -> None:
        self._add("UPDATE requests SET status = 'applied', handled = ? WHERE id = ?", [(time.time(), request)])

    def _on_flush(self) -> None:
        self._flush_all()
        self._commit()
        self._heartbeat(force=True)

    # Requests.

    def _poll(self, final: bool = False) -> None:
        """
            Hands the requests written by `Run.record` to the recorder and lists them in the index.

            The recorder applies them before its next call. A request that cannot be read is
            rejected after `SETTINGS.request_reads` attempts, and an invalid one at once. With
            ``final``, every request left expires.
        """
        if not final and time.monotonic() - self.last_poll < SETTINGS.requests_every:
            return
        self.last_poll = time.monotonic()
        listed = []
        for file in store.pending_requests(self.path):
            if file.stem in self.handled:
                continue
            try:
                request = store.read_request(file)
            except OSError:
                self.unreadable[file.stem] = self.unreadable.get(file.stem, 0) + 1
                if self.unreadable[file.stem] < SETTINGS.request_reads and not final:
                    continue
                request = store.blank_request(file)
            payload, status = request['payload'], 'rejected'
            if final:
                status = 'expired'
            else:
                name, steps = payload.get('measurements'), whole(payload.get('steps', 1))
                valid = steps is not None and 1 <= steps <= store.MAX_STEPS
                if request['kind'] == 'record' and isinstance(name, str) and name in self.measurements and valid:
                    self.incoming.append((name, steps, {'by': 'request', 'request': request['id']}))
                    status = 'received'
            self._add(
                'INSERT OR IGNORE INTO requests VALUES (?, ?, ?, ?, ?, ?)',
                [(request['id'], request['kind'], store.to_json(payload), request['wall'], status, time.time())],
            )
            listed.append(file)
        if listed:
            # Listed before the files go; a request read twice after a crash is listed once.
            self._commit()
            for file in listed:
                self.handled.add(file.stem)
                try:
                    file.unlink()
                except OSError:
                    pass

    # Windows.

    def _window(self, name: str) -> _Window:
        window = self.open.get(name)
        if window is None:
            window = _Window(self.measurements[name], self.next_window.get(name, 0))
            self.next_window[name] = window.number + 1
            self._add(_PROGRESS, [(f'window:{name}', window.number + 1)])
            self.open[name] = window
        return window

    def _flush(self, window: _Window) -> None:
        if window.empty():
            return
        arrays, t0, t1 = window.arrays()
        relative = store.window_file(window.measurements.name, window.number)
        # The spans and groups of the window are listed before its file exists; a file written without its
        # row is listed when the run is resumed.
        self._commit()
        _retried(store.write_arrays, self.path / relative, arrays)
        self._add(
            'INSERT INTO windows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [(window.measurements.name, window.number, t0, t1, len(window.spans), len(window.groups), str(relative),
              json.dumps(sorted(window.raw)), time.time())],
        )
        self._commit()

    def _flush_all(self) -> None:
        for name in list(self.open):
            self._flush(self.open.pop(name))

    def _flush_old(self) -> None:
        for name, window in list(self.open.items()):
            if time.monotonic() - window.opened >= self.flush_seconds:
                self._flush(self.open.pop(name))

    # Bookkeeping.

    def _heartbeat(self, force: bool = False) -> None:
        if not force and time.monotonic() - self.last_heartbeat < self.heartbeat_every:
            return
        self._write_info(strict=True)

    def _close(self, status: str, error: str | None, progress: dict[str, int]) -> None:
        self.progress.update(progress)
        for name, t0 in sorted(self.stretches.items()):
            self._check_frames(name, t0, None)
        self.stretches = {}
        self._poll(final=True)
        # Requests received after the last call; the recorder no longer takes them.
        unapplied = [(detail['request'],) for _, _, detail in self.incoming if 'request' in detail]
        self._add("UPDATE requests SET status = 'expired' WHERE id = ?", unapplied)
        self._apply()
        # Groups the recorder did not end, as after it failed to: their summaries as they stand.
        for name, key in list(self.groups):
            self._on_group(name, key, None, None, {})
            self._apply()
        self._flush_all()
        self._commit()
        self.connection.close()
        self.connection = None
        self._write_info(strict=True, status=status, error=error, finished=store.now())

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Sync:
    """
        Channel for the decisions of process 0, through the key-value store of ``jax.distributed``.

        The n-th recorder of each process shares the keys of the n-th recorder of process 0. Every
        process creates its recorders in the same order.
    """

    def __init__(self) -> None:
        try:
            from jax._src.distributed import global_state
            client = global_state.client
        except (ImportError, AttributeError):
            client = None
        if client is None or not hasattr(client, 'blocking_key_value_get'):
            raise RuntimeError(
                'Measurements with several processes needs the key-value store of jax.distributed; call '
                'jax.distributed.initialize first.'
            )
        self.client = client
        self.prefix = f'spark/recording/{next(_SYNCS)}'

    def publish(self, name: str, data: dict[str, tp.Any]) -> None:
        self.client.key_value_set(f'{self.prefix}/{name}', json.dumps(data))

    def receive(self, name: str, seconds: float | None = None) -> dict[str, tp.Any]:
        timeout = int(SETTINGS.sync_timeout * 1000) if seconds is None else max(int(seconds * 1000), 1)
        return json.loads(self.client.blocking_key_value_get(f'{self.prefix}/{name}', timeout))

    def listed(self, directory: str) -> list[dict[str, tp.Any]]:
        """
            Returns the values under ``directory``.
        """
        return [json.loads(value) for _, value in self.client.key_value_dir_get(f'{self.prefix}/{directory}/')]

    def forget(self, name: str) -> None:
        try:
            self.client.key_value_delete(f'{self.prefix}/{name}')
        except Exception:
            pass

    def noticed(self) -> bool:
        """
            Returns whether the preemption service of ``jax.distributed`` noted a SIGTERM.

            A SIGTERM of any process counts.
        """
        try:
            self.client.key_value_try_get(PREEMPTION_NOTICE)
        except Exception:
            return False
        return True

    def meet(self, name: str, seconds: float | None = None) -> None:
        """
            Waits until every process reaches the barrier ``name``.
        """
        limit = int(SETTINGS.sync_timeout * 1000)
        timeout = limit if seconds is None else min(int(seconds * 1000), limit)
        self.client.wait_at_barrier(f'{self.prefix}/{name}', timeout)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _decision(call: int) -> str:
    """
        Returns the key of the decision of process 0 about ``call``, in the directory of its block.
    """
    return f'call/{call // SETTINGS.decision_block}/{call}'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _preemption_service() -> bool:
    """
        Returns whether the preemption service of ``jax.distributed`` runs.

        The service notes the SIGTERM of any process.
    """
    try:
        from jax._src.distributed import global_state
        return global_state.preemption_sync_manager is not None
    except (ImportError, AttributeError):
        return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Signals:
    """
        Signal handlers of the open recorders, one per signal for the process.

        A signal is noted by every open recorder handling it, then passed to the handler ours
        replaced when that is a Python function other than `signal.default_int_handler`. It takes
        its usual effect once no recorder handles it.
    """

    def __init__(self) -> None:
        # The handlers ours replaced, one per time it was set, the last on top. A handler set over ours may
        # pass signals on to ours, which hands them on to the handler under it.
        self.before: dict[int, list[tp.Any]] = {}
        self.recorders: dict[int, list[Recorder]] = {}
        # Calls of `handle` in progress, when a handler it called passed the signal back to it.
        self.depth: dict[int, int] = {}
        # Signals delivered once every recorder is closed, while `_close_at_exit` closes them.
        self.postponed: list[tuple[int, tp.Any]] | None = None

    def add(self, recorder: Recorder, signums: tp.Iterable[int]) -> None:
        for signum in signums:
            if signum not in self.recorders:
                self.before.setdefault(signum, []).append(signal.signal(signum, self.handle))
                self.recorders[signum] = []
            self.recorders[signum].append(recorder)

    def remove(self, recorder: Recorder, deliver: int | None = None) -> None:
        """
            Stops handling the signals of ``recorder``.

            On the main thread, a signal no other recorder handles gets back the handler ours
            replaced. ``deliver``, a signal the recorder received and did not raise, is then passed
            to that handler, unless `handle` called it already.
        """
        for signum in [s for s, recorders in self.recorders.items() if recorder in recorders]:
            self.recorders[signum].remove(recorder)
            if self.recorders[signum] or threading.current_thread() is not threading.main_thread():
                continue
            restored, before = self._restore(signum)
            if restored and signum == deliver and not _called(before):
                if self.postponed is not None:
                    self.postponed.append((signum, before))
                else:
                    deliver_signal(signum, before)

    def _restore(self, signum: int) -> tuple[bool, tp.Any]:
        """
            Gives ``signum`` back the handler ours replaced.

            Does nothing when another handler was set over ours since. Returns whether it did, and
            the handler ours replaced.
        """
        del self.recorders[signum]
        befores = self.before[signum]
        if signal.getsignal(signum) != self.handle:
            return False, befores[-1]
        before = befores.pop()
        if not befores:
            del self.before[signum]
        signal.signal(signum, before if before is not None else signal.SIG_DFL)
        return True, before

    def handle(self, signum: int, frame: tp.Any) -> None:
        depth = self.depth.get(signum, 0)
        befores = self.before.get(signum, [])
        recorders = self.recorders.get(signum)
        if depth == 0 and recorders == []:
            # Every recorder handling it closed away from the main thread: as without them.
            restored, before = self._restore(signum)
            if restored:
                deliver_signal(signum, before, frame)
                return
        if depth == 0:
            for recorder in recorders or ():
                recorder._flag(signum)
        if depth >= len(befores):
            return
        # The handler under ours; when a handler ours called passes the signal back, the one under that.
        before = befores[-1 - depth]
        # The default handler of SIGINT raises KeyboardInterrupt anywhere; the recorder stops before a call instead.
        if not (_called(before) or (callable(before) and not recorders)):
            return
        self.depth[signum] = depth + 1
        try:
            deliver_signal(signum, before, frame)
        finally:
            self.depth[signum] = depth

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _called(handler: tp.Any) -> bool:
    """
        Returns whether `_Signals.handle` calls ``handler`` when a recorder notes the signal.
    """
    return callable(handler) and handler is not signal.default_int_handler

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True)
class _Options:
    """
        The options of a recorder, given to `Recorder` and `Recorder.resume` as keyword arguments.

        Each is described with `Recorder`, and checked when given. ``signals`` is held as a tuple of
        distinct signals.
    """
    queue_size: int = 64
    queue_bytes: int = 2 ** 30
    flush_steps: int | None = None
    flush_seconds: float = 60.0
    flush_bytes: int = 256 * 2 ** 20
    heartbeat: float = 5.0
    on_error: str = 'raise'
    signals: tp.Sequence[int] = ()

    def __post_init__(self) -> None:
        keep = lambda name, value: object.__setattr__(self, name, value)
        keep('queue_size', integer(self.queue_size, 'queue_size', lowest=1))
        if self.flush_steps is not None:
            keep('flush_steps', integer(self.flush_steps, 'flush_steps', lowest=1))
        for name in ('queue_bytes', 'flush_seconds', 'flush_bytes', 'heartbeat'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not value > 0 or not math.isfinite(value):
                raise ValueError(f'"{name}" must be a positive number, got {value!r}.')
        if self.on_error not in ON_ERROR:
            raise ValueError(f'Unknown on_error "{self.on_error}". Expected one of: {", ".join(ON_ERROR)}.')
        signals = tuple(dict.fromkeys(signal.Signals(s) for s in self.signals))
        uncatchable = [s.name for s in signals if s in _UNCATCHABLE]
        if uncatchable:
            raise ValueError(f'{", ".join(uncatchable)} cannot be handled.')
        if signals and threading.current_thread() is not threading.main_thread():
            raise ValueError('Signal handlers are installed from the main thread; create the recorder there or pass no signals.')
        keep('signals', signals)

    @classmethod
    def of(cls, options: dict[str, tp.Any]) -> _Options:
        """
            Returns the options given by keyword. Raises a TypeError for a name that is not an option.
        """
        names = [field.name for field in dc.fields(cls)]
        unknown = sorted(set(options) - set(names))
        if unknown:
            raise TypeError(f'Unknown options of the recorder: {", ".join(unknown)}. Expected any of: {", ".join(names)}.')
        return cls(**options)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Recorder:
    """
        Records the measurements asked for and writes them to a run.

        `record` asks for measurements over the next steps. A trigger, when the measurements have
        one, also asks for steps on its own. With the default trigger, `Manual`, measurements are
        recorded only when asked. Host values are written with `log`, `event`, `tag` and `raw`.

        The recorder works one call of the model at a time. While it is open, every call of a
        `spark.jit` function running a `spark.scan` is a call of the recorder: before the call, the
        recorder decides what it records, `spark.scan` records it, and the records are handed over
        after the call. `spark.jit` and `Runner` follow the same steps: before a call, `probes`
        returns the probes of the measurements recorded for it, and `start` where the call starts.
        After the call, `push` takes its records, with as many steps as `probes` was given. Their
        transfers start at once, and a writer thread completes the records and writes them.

        Records are placed on the steps of the run. Groups and strides do not depend on how the
        steps are split into calls. The recorder decides at the start of each call what it records,
        and records the call whole.

        The calls of a thread go to the last recorder it opened among those still open. A recorder
        opened within a run, such as one for an evaluation, takes the calls until it closes. Runs
        trained in threads of one process are recorded apart, each by the recorder its thread
        opened. A thread that opened no recorder uses the recorder open in the process, when only
        one is open, and raises at its first call when several are.

        Parameters
        ----------
        root : str or path-like, default 'runs'
            Directory holding the runs. The run gets its own directory inside, see `path`.
        config : SparkConfig or Controller, optional
            The configuration of the model recorded, written to the run. A built model can be given
            in its place: its configuration is written, and the probes are checked against it.
        measurements : sequence of Measurements, optional
            What can be recorded. Defaults to `presets.default` for a model given as ``config``,
            recorded by triggers.
        name : str, optional
            Name of the run, part of its directory name. Defaults to the class name of the model,
            or ``'run'`` without ``config``.
        experiment : str, optional
            Name of the experiment the run belongs to, such as one configuration trained with
            several seeds. Written to ``run.json``, and read as `Run.experiment`. The run viewer
            draws the runs of one experiment as a whole.
        run_id : str, optional
            Name of the directory of the run, in place of ``<date>-<time>_<name>_<id>``. A job
            restarted by its scheduler finds its run again with it, see `resume`.
        hparams : dict, optional
            Parameters of the experiment, written to ``hparams.json``.
        source : str or path-like, optional
            File the model configuration was read from. Its metadata, such as the node positions of
            the editor, is written with the configuration of the run.
        queue_size : int, default 64
            Items waiting to be written, such as the records of a call or a host value, past which
            calls block.
        queue_bytes : int, default 1 GiB
            Bytes of records and raw frames waiting to be written, past which calls block.
        flush_steps : int, optional
            Recorded steps after which a file of a set of measurements is written. By default, files
            are written by time and size alone.
        flush_seconds : float, default 60.0
            Seconds after which an open file is written.
        flush_bytes : int, default 256 MiB
            Host memory, in bytes, of the records and raw frames of an open file, past which it is
            written.
        heartbeat : float, default 5.0
            Seconds between updates of ``run.json`` while the run is open.
        on_error : str, default 'raise'
            What follows a failure of the writer, as on a full disk. With ``'raise'``, the next call
            to the recorder raises RuntimeError, and the run stops. With ``'continue'``, it warns
            once and training goes on without recording. Mistakes in what the loop hands over are
            not failures of the writer: they give a `RecordingWarning` either way.
        signals : sequence of int, optional
            Signals after which `probes` raises `Preempted` before the next call, such as the
            SIGTERM a cluster scheduler sends before it stops a job. The run is then closed as
            ``'preempted'``.

        Attributes
        ----------
        path : pathlib.Path
            Directory of the run.
        measurements : tuple of Measurements
            The measurements that can be recorded.
        step : int
            Current step of the run, advanced by `push`.
        tags : dict
            The last value of every tag.

        Raises
        ------
        ValueError
            When ``measurements`` is not given and ``config`` is not a model, when an option is
            invalid, or when two measurements share a name or disagree on a probe they share.
        FileExistsError
            When the directory of ``run_id`` exists.
        RuntimeError
            With several processes, when ``jax.distributed`` is not initialized, or when a process
            fails to open the run or to create its recorder within `SETTINGS.open_timeout`.

        Notes
        -----
        A run directory holds:

        * ``run.json``, the identity, environment, status and progress of the run.
        * ``model.scfg`` and ``hparams.json``, the configuration of the model and the parameters.
        * ``recorder.json``, the measurements.
        * ``index.sqlite``, the scalars, events, tags, spans and groups recorded, windows, requests
          and progress.
        * ``run.lock``, held by the recorder writing the run.
        * ``requests/``, the requests not read yet.
        * ``windows/<measurements>/<number>.npz``, the window files.
        * ``checkpoints/<step>.spark``, the checkpoints written by `checkpoint`.

        A window file holds what one set of measurements recorded over some time: the rows of its
        traces and rasters, one per step kept; its summaries, snapshots and deltas, one row per
        group of steps; and the raw frames kept meanwhile. An open
        file is written ``flush_seconds`` after it is opened, or once it holds ``flush_bytes``
        bytes. With ``flush_steps``, a file holding that many steps is written when the next call of
        its measurements arrives.

        Files are written to a temporary name, flushed to disk and renamed. After a crash, each file is
        complete or absent. The index is committed about once a second (`SETTINGS.commit_every`) and
        when a window is written.

        A group of steps (`Measurements.group`) is written once it ends. A group of a number of
        steps ends with its last step, and a group by tag when the tag takes a different value. The
        group in progress when the recorder closes is written cut short. A group whose steps were
        not all recorded, as when the measurements were recorded from the middle of it, is written
        with the steps it covers.

        While the queue of the writer is full, in items or in bytes, calls to the recorder block
        until the writer catches up. The waits are counted in ``run.json`` as ``stalls``. Records
        waiting to be written hold device memory. A single item larger than ``queue_bytes`` is let
        through when nothing else waits. After a failure of the writer, the open windows are written
        if they can be, and the run is marked ``'failed'``.

        What stops a run, and what does not. A failure of the writer, such as a full disk or a
        quota, stops the run with ``on_error='raise'``, the default: the next call to the recorder
        raises RuntimeError. A mistake in what the loop hands over never stops it: a frame of a raw
        stream no measurements declare, or of another shape than the first, measurements named
        without existing, a scalar that is not a number, or a declared raw stream recorded without
        a frame. Each gives a `RecordingWarning` once, is written to the run as a ``warning`` event
        (`Run.warnings`), and what it concerns is dropped. A short run with
        ``warnings.simplefilter('error', RecordingWarning)`` raises each where it happens, before a
        long run is started.

        A recorder left open is closed when the interpreter exits, before the thread pools of the
        interpreter stop, and waits for the checkpoints written in the background. The run is marked
        ``'preempted'`` after one of ``signals``, ``'failed'`` after an uncaught exception, and
        ``'finished'`` otherwise, `sys.exit` with an error code included. Closed by a ``with``
        block, the run is marked ``'failed'`` after `sys.exit` with an error code too. In a
        notebook, an uncaught exception does not mark the run ``'failed'``.

        An interrupt (Ctrl-C) received during a call of a `spark.jit` function is held until the call
        returns its result, and raised by the next call of the recorder from the loop (`probes`,
        `record`, `log`, `event`, `tag` or `raw`), before it does anything. The steps of the run then
        match the state the loop holds. A second interrupt during the call raises at once. A long
        compilation is not held. An interrupt (Ctrl-C) received while the recorder closes is raised
        once what is left is written. A second interrupt stops the wait, and what is not written yet may be lost. A
        process killed by a signal it does not handle, such as the SIGKILL that follows SIGTERM,
        loses the windows being filled and the groups in progress. `resume` marks the spans and
        groups of those windows as lost.

        Measurements can also be recorded from outside the training process, by the run viewer of
        the editor or by `Run.record` from any process or host with access to the run directory. The
        writer thread reads those requests, and the recorder applies them before its next call.

        The handlers of ``signals`` are installed from the main thread and shared by the open
        recorders. A handler set before is still called, except the one of SIGINT raising
        KeyboardInterrupt. A signal received after the last call, which `probes` never raised, takes
        its usual effect once the run is closed. Without a handler set before, the process then
        ends. In a notebook, the kernel sets the handler of SIGINT again for every cell, and SIGINT
        is handled only in the cell creating the recorder.

        With several JAX processes (``jax.distributed``), the recorder records one model sharded
        across them, and every process runs the same calls. Every process creates the recorder and
        calls it in the same order. Process 0 creates and writes the run, and decides what every
        call records. The other processes receive its decisions through the key-value store of
        ``jax.distributed`` and write nothing. A recorder created on process 0 alone raises after
        `SETTINGS.open_timeout`.

        Tags, `record`, `When` conditions and requests take effect from process 0. The processes
        stop together. A failure of the writer with ``on_error='raise'``, or a signal of ``signals``
        received by any process, raises from `probes` on every process at the same call. When the
        preemption service of ``jax.distributed`` runs, it notes SIGTERM and keeps its handler.
        Other signals are published by the process receiving them. Process 0 reads both from the
        key-value store. Records are replicated on every device, where process 0 reads them. A
        snapshot of a value too large for one device cannot be recorded.

        See Also
        --------
        Measurements : A named set of probes recorded together.
        spark.jit : Compiles a function whose calls the open recorder records.
        Runner : Steps a model and records what its recorder asks for.
        Run : A run read back from its directory.

        Examples
        --------
        >>> @partial(spark.jit, static_argnames=['steps'])
        ... def run(graph, state, steps, **inputs):
        ...     def step(state, _):
        ...         model = spark.merge(graph, state)
        ...         outputs = model(**inputs)
        ...         return spark.split(model)[1], outputs
        ...     return spark.scan(step, state, length=steps)
        >>> episode = spark.recording.Measurements('episode', probes, group='episode')
        >>> graph, state = spark.split(brain)
        >>> hparams = {'lr': 0.1}
        >>> with spark.recording.Recorder('runs', brain.config, [episode], hparams=hparams) as recorder:
        ...     for number in range(100):
        ...         recorder.tag(episode=number)
        ...         if number % 20 == 0:
        ...             recorder.record('episode')
        ...         state, outputs = run(graph, state, steps=50, **inputs)
        ...         recorder.log(reward=reward)
    """

    def __init__(
            self,
            root: str | pathlib.Path = 'runs',
            config: SparkConfig | Controller | None = None,
            measurements: tp.Sequence[Measurements] | None = None,
            *,
            name: str | None = None,
            experiment: str | None = None,
            run_id: str | None = None,
            hparams: dict[str, tp.Any] | None = None,
            source: str | pathlib.Path | None = None,
            **options: tp.Any,
        ) -> None:
        if experiment is not None and (not isinstance(experiment, str) or not experiment.strip()):
            raise ValueError(f'"experiment" names the experiment of the run, got {experiment!r}.')
        self._prepare(_Options.of(options))
        # A model gives its configuration, and the probes are checked against it.
        model = None if config is None or isinstance(config, SparkConfig) else config
        config = getattr(model, 'config', None) if model is not None else config
        if measurements is None:
            if model is None:
                raise ValueError('Expected measurements, or a model given as `config` to derive them from.')
            from spark.recording.presets import default
            measurements = default(model)
        self._setup(model, measurements)
        model_name = type(model).__name__ if model is not None else _model_name(config)

        def create() -> dict[str, tp.Any]:
            run_name = name or model_name or 'run'
            # Built under a temporary name and moved into place complete: a job ended while it starts
            # leaves no directory under the name it will look for when it is restarted.
            self.path, final = store.new_run_dir(root, run_name, run_id)
            self._lock = store.lock(self.path)
            self._info = {
                'id': final.name,
                'name': run_name,
                'experiment': experiment,
                'created': store.now(),
                'status': 'running',
                'heartbeat': store.now(),
                'heartbeat_every': self._options.heartbeat,
                'finished': None,
                'error': None,
                'step': 0,
                'stalls': 0,
                'resumed': [],
                'model': model_name,
                'environment': store.environment(),
                'process': store.process(),
                'git': store.git_state(self.path),
            }
            self._write_setup(config, hparams, source)
            store.write_json(self.path / 'run.json', self._info)
            self.path = store.move_run_dir(self.path, final)
            self._start({})
            return self._setup_data()
        self._open(create)

    @classmethod
    def resume(
            cls,
            path: str | pathlib.Path,
            model: Controller | None = None,
            measurements: tp.Sequence[Measurements] | None = None,
            *,
            step: int | None = None,
            **options: tp.Any,
        ) -> Recorder:
        """
            Reopens a run and appends to it.

            The step count continues from the last step written to the run, or from the step of a
            later checkpoint. Window numbers continue after the last window of each set of
            measurements.

            Parameters
            ----------
            path : str or path-like
                Directory of the run. With several processes, only process 0 reads it.
            model : Controller, optional
                The model recorded. When it is built, the probes are checked against it.
            measurements : sequence of Measurements, optional
                What can be recorded. Defaults to the measurements the run was last opened with.
            step : int, optional
                Step to continue from, such as the step of the checkpoint the model was restored
                from. The last step written by default.
            **options
                The options of `Recorder`: ``queue_size``, ``queue_bytes``, ``flush_steps``,
                ``flush_seconds``, ``flush_bytes``, ``heartbeat``, ``on_error`` and ``signals``.

            Returns
            -------
            Recorder
                The recorder writing the run.

            Raises
            ------
            ValueError
                When ``step`` is not a non-negative integer, or an option is invalid.
            PermissionError
                When the run directory cannot be written.
            RuntimeError
                When another recorder is writing the run.

            Notes
            -----
            The spans and groups listed for a window that was never written are marked lost, with
            window -1. A window file written but not listed is listed. Requests left pending expire,
            and files left half written are removed.

            The condition of a `When` trigger is code and is not written to the run. Without
            ``measurements``, such measurements come back with a `Manual` trigger, with a warning.

            With ``step``, the records written after it are kept, and the steps after it are
            recorded twice. The tags take the values they had at ``step``. A group holding ``step``
            starts again there, and is written twice too.

            A lock left by the file system of a node that failed is released by removing
            ``run.lock``, once no process writes the run.

            Examples
            --------
            A job that its scheduler may restart:

            >>> run = pathlib.Path('runs') / os.environ['SLURM_JOB_ID']
            >>> step = None                     # the step of the last checkpoint, 0 without one
            >>> if (run / 'run.json').exists():
            ...     step = max(spark.recording.load(run).checkpoints(), default=0)
            >>> if jax.process_count() > 1:     # every process goes on as process 0 does
            ...     found = np.asarray(-1 if step is None else step)
            ...     step = int(multihost_utils.broadcast_one_to_all(found))
            ...     step = None if step < 0 else step
            >>> if step is None:
            ...     brain = build(config)
            ...     recorder = Recorder('runs', brain, run_id=os.environ['SLURM_JOB_ID'])
            ... else:
            ...     saved = spark.recording.load(run)
            ...     if step in saved.checkpoints():
            ...         brain = saved.restore(step)
            ...     else:
            ...         brain = build(saved.config())
            ...     recorder = Recorder.resume(run, brain, step=step)

            ``build`` builds a model from a configuration and calls it once with example inputs. The
            model is restored from the last checkpoint, if any, and built from the configuration of
            the run otherwise. The recording goes on from the step of that checkpoint, or from the
            start without one.

            With several processes, one process may see the files of the run a moment before
            another. Every process takes the decision of process 0. ``multihost_utils`` is
            ``jax.experimental.multihost_utils``.
        """
        if step is not None:
            step = integer(step, 'step', lowest=0)
        recorder = cls.__new__(cls)
        recorder._prepare(_Options.of(options))
        path = pathlib.Path(path).resolve()
        given = measurements is not None
        if given:
            recorder._setup(model, measurements)

        def create() -> dict[str, tp.Any]:
            if not given:
                data = json.loads((path / 'recorder.json').read_text())
                recorder._setup(model, [Measurements.from_dict(r) for r in data['measurements']])
                lost = [r['name'] for r in data['measurements'] if r['trigger'].get('kind') == 'When']
                if lost:
                    warnings.warn(
                        f'The measurements {lost} of "{path.name}" were recorded by conditions, which are code and are not '
                        f'written to the run; they are recorded by hand only until the measurements are given to resume.'
                    )
            if not all(os.access(f, os.W_OK) for f in (path, path / 'index.sqlite', path / 'run.json')):
                raise PermissionError(f'"{path}" cannot be written; a run is resumed where it can be written.')
            lock = store.lock(path, wait=5.0)
            if lock is None:
                raise RuntimeError(
                    f'"{path}" is being written by another recorder ({_heartbeat_age(path)}). A lock kept by the file '
                    f'system of a node that failed is released by removing run.lock, once no process writes the run.'
                )
            recorder.path, recorder._lock = path, lock
            recorder._info = store.read_info(path)
            recorder.step, windows_from = recorder._reopen()
            if step is not None:
                recorder.step = int(step)
            recorder.tags = recorder._last_tags(step)
            recorder._info.update(
                status='running', heartbeat=store.now(), heartbeat_every=recorder._options.heartbeat, finished=None, error=None,
            )
            recorder._info.setdefault('resumed', []).append(store.now())
            recorder._info['process'] = store.process()
            recorder._write_measurements()
            recorder._start(windows_from)
            return recorder._setup_data()

        def received(setup: dict[str, tp.Any]) -> None:
            if not given:
                recorder._setup(model, [Measurements.from_dict(r) for r in setup['measurements']])
            recorder.tags = dict(setup.get('tags', {}))
        recorder._open(create, received)
        return recorder

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    def _prepare(self, options: _Options) -> None:
        """
            Sets the state that does not depend on the measurements, with the options given.
        """
        self._writes = jax.process_index() == 0
        self._sync: _Sync | None = _Sync() if jax.process_count() > 1 else None
        # With several processes, the preemption service of JAX keeps the handler it installed for SIGTERM, and
        # notes the SIGTERM of any process; process 0 reads the note. Its sync point is left to other code.
        self._preemption = self._sync is not None and signal.SIGTERM in options.signals and _preemption_service()
        if self._preemption:
            options = dc.replace(options, signals=tuple(s for s in options.signals if s != signal.SIGTERM))
        self._options = options
        self.path: pathlib.Path | None = None
        self._info: dict[str, tp.Any] = {}
        self._writer: _Writer | None = None
        self._lock = None
        self._closed = False
        self._close_queued = False
        self._finished = False
        self._warned = False
        self._signal: int | None = None
        self._raised = False
        self._published = False
        self._polled = 0.0
        # Set when a second interrupt stopped the wait for the writer while closing.
        self._stopped_waiting = False
        # With several processes, the call last decided by `probes`, the measurements recorded for it, and why the
        # run stops there. Calls are counted from the opening of the recorder, the same on every process.
        self._decided: tuple[int, frozenset[str], dict | None] | None = None
        self._calls = 0
        self.step = 0
        self.tags: dict[str, tp.Any] = {}
        # Step at which every tag took its last value, where scalars logged per it are placed.
        self._tag_steps: dict[str, int] = {}

    def _setup(self, model: Controller | None, measurements: tp.Sequence[Measurements]) -> None:
        """
            Checks the measurements and sets the state that depends on them.

            Shared by new and resumed recorders, and by every process.
        """
        self.measurements: tuple[Measurements, ...] = tuple(measurements)
        names = [r.name for r in self.measurements]
        if len(set(names)) != len(names):
            raise ValueError(f'Measurements share names: {sorted({n for n in names if names.count(n) > 1})}.')
        if model is not None and getattr(model, '__built__', False):
            for measurements in self.measurements:
                validate(model, measurements.probes)
        # Raises when two measurements disagree on a probe they share.
        merge_probes(p for r in self.measurements for p in r.probes)
        self._by_name = {r.name: r for r in self.measurements}
        self._raw = {m: [r.name for r in self.measurements if m in r.raw] for r in self.measurements for m in r.raw}
        self._raw_formats: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        # Measurements recorded when a condition holds for other measurements, by the measurements they watch.
        self._watchers: dict[str, list[tuple[str, When]]] = {}
        for measurements in self.measurements:
            if isinstance(measurements.trigger, When):
                watch = measurements.trigger.watch
                if watch not in names or watch == measurements.name:
                    raise ValueError(f'The trigger of "{measurements.name}" watches "{watch}", which names no other measurements.')
                self._watchers.setdefault(watch, []).append((measurements.name, measurements.trigger))
        # Measurements keeping the calls before their trigger, and those calls.
        self._rings = {r.name: collections.deque() for r in self.measurements if r.lookback > 0}
        self._variants: dict[frozenset[str], tuple[Probe, ...]] = {frozenset(): ()}
        self._handed: set[frozenset[str]] = {frozenset()}
        # Measurements recorded by hand, by the step up to which they are recorded.
        self._manual: dict[str, int] = {}
        # The group being recorded for each set of measurements, on process 0.
        self._current: dict[str, _Open] = {}
        # Recorded and captured measurements, their probes and the tags, from `probes` until `push`, and the steps
        # `probes` was given.
        self._pending: tuple[frozenset[str], frozenset[str], tuple[Probe, ...], dict[str, tp.Any]] | None = None
        self._pending_steps: int | None = None
        self._last_steps: int | None = None
        # A SIGINT `spark.jit` held during a call, with the handler it was meant for.
        self._interrupt: HeldInterrupt | None = None
        self._incoming: collections.deque = collections.deque()
        # Causes warned of, and the warnings of the writer, given from the loop.
        self._causes: set[tuple] = set()
        self._notices: collections.deque = collections.deque()
        # Writes the checkpoints in the background, one at a time.
        self._checkpointer: concurrent.futures.ThreadPoolExecutor | None = None
        self._checkpointing: concurrent.futures.Future | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=int(self._options.queue_size))
        self._room = threading.Condition()
        self._queued_bytes = 0
        self._stalls = 0

    def _open(self, create: tp.Callable[[], dict], received: tp.Callable[[dict], None] | None = None) -> None:
        """
            Creates or reopens the run on process 0, and hands its path and progress to the others.

            Installs the signal handlers first. A failure on process 0 raises on every process.
        """
        _SIGNALS.add(self, self._options.signals)
        try:
            if self._writes:
                self._open_run(create)
            else:
                self._follow_run(received)
        except BaseException:
            _SIGNALS.remove(self)
            raise
        _OPEN[self] = threading.current_thread()

    def _open_run(self, create: tp.Callable[[], dict]) -> None:
        """
            Creates or reopens the run, on process 0.

            With several processes, hands it to the others and waits until each has taken it, for
            `SETTINGS.open_timeout` seconds at most.
        """
        try:
            setup = create()
        except BaseException as error:
            if self._sync is not None:
                self._sync.publish('setup', {'error': f'{type(error).__name__}: {error}'})
            self._release_run()
            raise
        if self._sync is None:
            return
        self._sync.publish('setup', setup)
        timeout = SETTINGS.open_timeout
        deadline = time.monotonic() + timeout
        for process in range(1, jax.process_count()):
            try:
                answer = self._sync.receive(f'open/{process}', max(deadline - time.monotonic(), 0.001))
            except Exception as error:
                self._release_run()
                raise RuntimeError(
                    f'Process 0 created a recorder and process {process} did not within {timeout:.0f} s. Every '
                    f'process creates the recorder, and calls it the same way; only process 0 writes.'
                ) from error
            if answer.get('error'):
                self._release_run()
                raise RuntimeError(f'Process {process} could not open the run: {answer["error"]}')

    def _follow_run(self, received: tp.Callable[[dict], None] | None) -> None:
        """
            Takes the run process 0 opened, on the other processes, and reports back to process 0.
        """
        setup = self._sync.receive('setup')
        if setup.get('error'):
            raise RuntimeError(f'Process 0 could not open the run: {setup["error"]}')
        try:
            if received is not None:
                received(setup)
        except BaseException as error:
            self._sync.publish(f'open/{jax.process_index()}', {'error': f'{type(error).__name__}: {error}'})
            raise
        self._sync.publish(f'open/{jax.process_index()}', {'error': None})
        self.path = pathlib.Path(setup['path'])
        self.step = int(setup['step'])

    def _release_run(self) -> None:
        """
            Stops the writer and releases the lock after a failure while opening the run.

            A new run not moved into place is removed.
        """
        if self._writer is not None and self._writer.is_alive():
            self._queue.put(('close', 0, 'failed', 'The recorder failed while opening the run.', self._progress()))
            self._writer.join()
        if self._lock is not None:
            store.unlock(self._lock)
            self._lock = None
        if self.path is not None and store.is_temporary(self.path):
            shutil.rmtree(self.path, ignore_errors=True)

    def _setup_data(self) -> dict[str, tp.Any]:
        """
            Returns what process 0 hands to the other processes when the run opens: its path, step,
            measurements and tags.
        """
        return {'path': str(self.path), 'step': self.step, 'measurements': [r.to_dict() for r in self.measurements], 'tags': self.tags}

    def _write_measurements(self) -> None:
        """
            Writes the measurements to ``recorder.json``.
        """
        store.write_json(self.path / 'recorder.json', {'measurements': [r.to_dict() for r in self.measurements]})

    def _write_setup(self, config: SparkConfig | None, hparams: dict | None, source: str | pathlib.Path | None) -> None:
        store.write_json(self.path / 'hparams.json', hparams or {})
        self._write_measurements()
        if config is not None:
            metadata = None
            if source is not None:
                try:
                    metadata = SparkConfig.metadata_from_file(str(source))
                except Exception as error:
                    warnings.warn(f'The metadata of "{source}" could not be read: {error}')
            try:
                config.to_file(str(self.path / 'model.scfg'), verbose=False, metadata=metadata)
            except Exception as error:
                warnings.warn(f'The configuration of the model could not be written to the run: {error}')
        # The tables exist before run.json names the run.
        store.connect(self.path).close()

    def _reopen(self) -> tuple[int, dict[str, int]]:
        """
            Repairs the index of a reopened run and returns its step and next window numbers.

            The step is the last one written, or that of a later checkpoint. Windows are listed or
            marked lost by `_list_windows`, and window numbers are not used again. Requests left
            pending expire, and files left half written are removed.
        """
        store.remove_partial(self.path)
        step = int(self._info.get('step', 0))
        windows_from: dict[str, int] = {}
        connection = store.connect(self.path)
        try:
            for key, value in connection.execute('SELECT key, value FROM progress'):
                if key == 'step':
                    step = max(step, int(value))
                elif key.startswith('window:'):
                    windows_from[key[7:]] = max(windows_from.get(key[7:], 0), int(value))
            for table in ('windows', 'spans', 'groups'):
                for measurements, last in connection.execute(f'SELECT measurements, MAX(window) FROM {table} GROUP BY measurements'):
                    windows_from[measurements] = max(windows_from.get(measurements, 0), int(last) + 1)
            for measurements, number in store.window_files(self.path):
                windows_from[measurements] = max(windows_from.get(measurements, 0), number + 1)
            # The progress is written with every commit: a run resumed from an earlier step goes on from there.
            self._list_windows(connection)
            now = time.time()
            files = store.pending_requests(self.path)
            for file in files:
                try:
                    request = store.read_request(file)
                except OSError:
                    request = store.blank_request(file)
                connection.execute(
                    'INSERT OR IGNORE INTO requests VALUES (?, ?, ?, ?, ?, ?)',
                    (request['id'], request['kind'], store.to_json(request['payload']), request['wall'], 'expired', now),
                )
            # Requests received but not applied when the process ended are not applied now.
            connection.execute("UPDATE requests SET status = 'expired', handled = ? WHERE status IN ('pending', 'received')", (now,))
            store.commit(connection, self.path)
            for file in files:
                try:
                    file.unlink()
                except OSError:
                    pass                                                # listed; the writer does not read it again
            # A checkpoint of the model is at a step it reached, whatever the index lost.
            step = max([step, *store.checkpoint_steps(self.path)])
        finally:
            connection.close()
        return step, windows_from

    def _list_windows(self, connection: sqlite3.Connection) -> None:
        """
            Lists the window files written but not listed, and marks lost the windows never written.

            The spans and groups of a lost window get window -1. A file whose spans or groups do not
            match those listed for it is treated as never written.
        """
        listed = set(connection.execute('SELECT measurements, window FROM windows'))
        counts = {
            table: {
                (measurements, number): count for measurements, number, count in
                connection.execute(f'SELECT measurements, window, COUNT(*) FROM {table} WHERE window >= 0 GROUP BY measurements, window')
            }
            for table in ('spans', 'groups')
        }
        found = store.window_files(self.path)
        for measurements, number in sorted((set(counts['spans']) | set(counts['groups']) | found) - listed):
            relative = store.window_file(measurements, number)
            written = None
            if (measurements, number) in found:
                try:
                    with np.load(self.path / relative) as data:
                        starts, steps = data['span_t0'], data['span_steps']
                        groups = (data['group_t0'], data['group_steps']) if 'group_t0' in data.files else (np.zeros(0, np.int64),) * 2
                        raw = sorted(k.removeprefix('raw:') for k in data.files if k.startswith('raw:') and not k.endswith('#t'))
                        times = [
                            *starts.tolist(), *(starts + steps).tolist(), *groups[0].tolist(), *(groups[0] + groups[1]).tolist(),
                            *(int(t) for m in raw for t in data[f'raw:{m}#t']),
                        ]
                        matches = len(starts) == counts['spans'].get((measurements, number), 0) and len(groups[0]) == counts['groups'].get((measurements, number), 0)
                        if matches and times:
                            written = (min(times), max(times), len(starts), len(groups[0]), raw)
                except Exception:
                    written = None
            if written is not None:
                t0, t1, spans, groups, raw = written
                connection.execute(
                    'INSERT INTO windows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (measurements, number, t0, t1, spans, groups, str(relative), json.dumps(raw), time.time()),
                )
            else:
                # Lost spans and groups keep their rows; the row ids a reader has seen are not reused.
                for table in ('spans', 'groups'):
                    connection.execute(f'UPDATE {table} SET window = -1 WHERE measurements = ? AND window = ?', (measurements, number))

    def _last_tags(self, step: int | None = None) -> dict[str, tp.Any]:
        """
            Returns the last value of every tag of the run, or the value it had at ``step``.
        """
        before, parameters = ('', ()) if step is None else ('WHERE t <= ?', (int(step),))
        connection = store.connect(self.path)
        try:
            rows = connection.execute(
                f'SELECT key, value FROM tags WHERE rowid IN (SELECT MAX(rowid) FROM tags {before} GROUP BY key)', parameters,
            ).fetchall()
        finally:
            connection.close()
        return {key: json.loads(value) for key, value in rows}

    def _start(self, windows_from: dict[str, int]) -> None:
        store.write_json(self.path / 'run.json', self._info)
        self._writer = _Writer(self, windows_from)
        self._writer.start()

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    def _flag(self, signum: int) -> None:
        """
            Notes a signal of the recorder, raised by the next `probes`.

            Only the first signal is kept.
        """
        if self._signal is None:
            self._signal = signum

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    def _check(self) -> None:
        """
            Raises when the recorder is closed or its writer failed.

            A failed writer raises with ``on_error='raise'`` and warns once with ``'continue'``.
            With several processes, `probes` raises it instead, on every process.
        """
        if self._closed:
            raise RuntimeError(f'The recorder of "{self.path}" is closed.')
        error = self._writer.error if self._writer is not None else None
        if error is None:
            return
        if self._options.on_error == 'raise':
            if self._sync is None:
                raise RuntimeError(f'The writer of "{self.path}" failed.') from error
            return
        if not self._warned:
            self._warned = True
            warnings.warn(f'The writer of "{self.path}" failed; the run is no longer recorded. {type(error).__name__}: {error}')

    def _hold_interrupt(self, held: HeldInterrupt) -> None:
        """
            Keeps a SIGINT received during a call of `spark.jit`, for `_deliver_interrupt`.

            The call returned its result to the loop, and its records were handed over.
        """
        self._interrupt = held

    def _warn(self, cause: tuple, message: str, **detail: tp.Any) -> None:
        """
            Warns with a `RecordingWarning`, once per ``cause``, and writes a ``warning`` event.
        """
        if cause in self._causes:
            return
        self._causes.add(cause)
        if not self._closed and self._writer is not None and self._writer.error is None:
            self._put(('event', 0, self.step, 'warning', {'message': message, **detail}, time.time()))
        warnings.warn(message, RecordingWarning, skip_file_prefixes=(_PACKAGE,))

    def _give_notices(self) -> None:
        """
            Gives the warnings of the writer, from the thread of the loop.
        """
        while self._notices:
            warnings.warn(self._notices.popleft(), RecordingWarning, skip_file_prefixes=(_PACKAGE,))

    def _deliver_interrupt(self) -> None:
        """
            Delivers the SIGINT held during the last call to its handler, if any.

            Called first by the methods of the loop (`probes`, `record`, `log`, `event`, `tag`, `raw`).
            With the default handler, raises KeyboardInterrupt.
        """
        held, self._interrupt = self._interrupt, None
        if held is not None:
            held.deliver()

    def _put(self, item: tuple) -> None:
        """
            Hands ``item``, ``(kind, nbytes, *arguments)``, to the writer.

            Blocks while the queue is full, in items or in bytes, and counts the wait as a stall.
        """
        self._check()
        if self._writer is None:
            return
        nbytes, stalled = int(item[1]), False
        with self._room:
            while self._queued_bytes and self._queued_bytes + nbytes > self._options.queue_bytes and self._writer.is_alive():
                stalled = True
                self._room.wait(timeout=1.0)
            self._queued_bytes += nbytes
        if self._queue.full():
            stalled = True
        self._stalls += stalled
        try:
            self._queue.put(item)
        except BaseException:
            self._release(nbytes)
            raise

    def _release(self, nbytes: int) -> None:
        """
            Frees ``nbytes`` of the queue once the writer has handled an item.
        """
        if nbytes:
            with self._room:
                self._queued_bytes -= nbytes
                self._room.notify_all()

    def _progress(self) -> dict[str, int]:
        return {'step': self.step, 'stalls': self._stalls}

    def _counters(self) -> dict[str, int]:
        counters = {'step': self.step}
        # A copy: tags can be set from another thread.
        for key, value in list(self.tags.items()):
            if isinstance(value, (bool, int, np.integer)):
                counters.setdefault(key, int(value))
        return counters

    def _recorded_for(self, counters: dict[str, int], spans: dict[str, int] | None) -> frozenset[str]:
        return frozenset(
            r.name for r in self.measurements
            if r.trigger.recorded(counters, spans) or self._manual.get(r.name, 0) > counters['step'] or self._continues(r)
        )

    def _continues(self, measurements: Measurements) -> bool:
        """
            Returns whether the group recorded for ``measurements`` goes on at the current step.

            A group of steps goes on until its last step, and a group by tag while the tag keeps its
            value.
        """
        group = self._current.get(measurements.name)
        if group is None:
            return False
        return isinstance(measurements.group, int) or _same(self.tags.get(measurements.group), group.tag)

    def _recorded_here(self) -> frozenset[str]:
        """
            Returns the measurements recorded for the call starting at the current step.

            Between `probes` and `push`, it is the call being run. Otherwise it is the next call,
            taken with the counters as they are and the length of the last call.
        """
        if self._pending is not None:
            return self._pending[0]
        return self._recorded_for(self._counters(), {'step': self._last_steps} if self._last_steps else None)

    def _merged(self, recorded: frozenset[str], handed: bool = True) -> tuple[Probe, ...]:
        """
            Returns the probes of the measurements ``recorded``, the same tuple for the same set.

            With ``handed``, a new set counts towards `SETTINGS.variant_warning`. The sets `warmup_sets`
            compiles do not count.
        """
        probes = self._variants.get(recorded)
        if probes is None:
            probes = merge_probes(p for r in self.measurements if r.name in recorded for p in r.probes)
            self._variants[recorded] = probes
        if handed and recorded not in self._handed:
            self._handed.add(recorded)
            if len(self._handed) == SETTINGS.variant_warning + 1:
                warnings.warn(
                    f'{SETTINGS.variant_warning} different sets of recorded measurements so far; each is compiled once. '
                    f'Triggers whose periods divide each other keep the number of sets small.'
                )
        return probes

    def _transfer(self, records: Packed | dict) -> None:
        """
            Starts moving ``records`` to the host.
        """
        for leaf in jax.tree.leaves(records):
            start = getattr(leaf, 'copy_to_host_async', None)
            if start is not None:
                start()

    def _check_tags(self) -> None:
        """
            Warns about tags that triggers count or measurements are grouped by, when never set.

            A tag a trigger counts is also reported when it is not an integer.
        """
        for measurements in self.measurements:
            tag = measurements.trigger.tag
            if tag is not None and not isinstance(measurements.trigger, When):
                if tag not in self.tags:
                    warnings.warn(f'The trigger of "{measurements.name}" counts "{tag}", which no call to `tag` has set in {self._calls} calls.')
                elif not isinstance(self.tags[tag], (bool, int, np.integer)):
                    warnings.warn(f'The trigger of "{measurements.name}" counts "{tag}", a tag set to {self.tags[tag]!r}, which is not an integer.')
            if isinstance(measurements.group, str) and measurements.group not in self.tags:
                warnings.warn(
                    f'Measurements "{measurements.name}" are grouped by "{measurements.group}", which no call to `tag` has set in '
                    f'{self._calls} calls: its steps so far are one group.'
                )

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    def probes(self, steps: int | None = None) -> tuple[Probe, ...]:
        """
            Returns the probes of the measurements recorded for the next call of the model.

            The same tuple object is returned for the same set of recorded measurements. Pass it as
            a static keyword argument, ``()`` included, with `start`. Measurements with a
            ``lookback`` are captured on every call.

            Parameters
            ----------
            steps : int, optional
                Steps of the next call. A trigger counting steps records the call when any of its
                steps is recorded. Without ``steps``, it records the call when its first step is.
                When given, `push` must receive the same steps.

            Returns
            -------
            tuple of Probe
                Probes to run in the call. Empty when nothing is recorded or captured.

            Raises
            ------
            Preempted
                When one of the signals of the recorder was received. The call is not run.
            RuntimeError
                When the recorder is closed, or its writer failed with ``on_error='raise'``.

            Notes
            -----
            With one process, a second call before `push` warns, and the call the first one recorded
            is not recorded. With several processes, every process calls it for every call. Process
            0 decides what the call records, and the other processes receive the decision.
        """
        self._deliver_interrupt()
        self._check()
        self._give_notices()
        if self._pending is not None and self._sync is None:
            warnings.warn('`probes` was called again before `push`: the call it recorded, if it ran, is not recorded.')
        self._pending_steps = int(steps) if steps else None
        if self._sync is not None and self._writes and time.monotonic() - self._polled >= SETTINGS.signals_every:
            # The signals of the other processes, and a SIGTERM noted by the preemption service of JAX.
            self._polled = time.monotonic()
            for value in self._sync.listed('signal'):
                self._flag(int(value['signal']))
            if self._preemption and self._sync.noticed():
                self._flag(signal.SIGTERM)
        if self._sync is not None and not self._writes and self._signal is not None and not self._published:
            # Handed to process 0, which stops every process at a call it decides.
            self._published = True
            self._sync.publish(f'signal/{jax.process_index()}', {'signal': int(self._signal)})
        fresh = True
        if self._sync is not None and self._decided is not None and self._decided[0] == self._calls:
            # Asked again for the same call, as after a call that raised: the decision every process read.
            recorded, stop, fresh = self._decided[1], self._decided[2], False
        elif self._writes:
            self._take_incoming()
            if self._calls == SETTINGS.tag_warning_calls:
                self._check_tags()
            recorded = self._recorded_for(self._counters(), {'step': int(steps)} if steps else None)
            error = self._writer.error if self._writer is not None else None
            stop: dict[str, tp.Any] | None = None
            if self._signal is not None:
                stop = {'signal': int(self._signal)}
            elif error is not None and self._options.on_error == 'raise':
                stop = {'error': f'{type(error).__name__}: {error}'}
            if self._sync is not None:
                self._sync.publish(_decision(self._calls), {'record': sorted(recorded), 'stop': stop})
        else:
            decision = self._sync.receive(_decision(self._calls))
            recorded, stop = frozenset(decision['record']), decision['stop']
        block = SETTINGS.decision_block
        if self._sync is not None and fresh and self._calls % block == block - 1:
            # No process runs more than a block ahead of another; the block before the last is not read again.
            self._sync.meet(f'block/{self._calls // block}')
            if self._writes and self._calls >= block:
                self._sync.forget(f'call/{self._calls // block - 1}')
        self._decided = (self._calls, recorded, stop)
        if stop is not None and 'error' in stop:
            if self._writes:
                raise RuntimeError(f'The writer of "{self.path}" failed.') from self._writer.error
            raise RuntimeError(f'The writer of "{self.path}" on process 0 failed: {stop["error"]}')
        if stop is not None:
            self._raised = True
            raise Preempted(stop['signal'])
        # Measurements keeping the calls before their trigger are captured on every call.
        captured = recorded | frozenset(self._rings)
        probes = self._merged(captured)
        self._pending = (recorded, captured, probes, dict(self.tags))
        return probes

    def start(self, probes: tuple[Probe, ...]) -> Start:
        """
            Returns where the next call starts on the steps of the run, for the recorded scan.

            Called after `probes`. The result holds the current step modulo the group sizes and strides
            of ``probes``. The recorded scan aligns its groups and strides on the steps of the run with
            it, however the steps are split into calls. The result also tells whether the call crosses
            the end of a group, from the steps given to `probes`. Without them, the call may cross one.

            Parameters
            ----------
            probes : tuple of Probe
                The probes `probes` returned for the call.

            Returns
            -------
            Start
                Phases of the current step, and whether the call may cross the end of a group.

            Examples
            --------
            >>> probes = recorder.probes(50)
            >>> start = recorder.start(probes)
            >>> outputs, state, records = call(graph, state, inputs, start, steps=50, probes=probes)
            >>> recorder.push(records, 50)
        """
        return start_of(probes, self.step, self._pending_steps)

    def _take_incoming(self) -> None:
        """
            Records the measurements the writer asked for, from requests and `When` conditions.

            A ``record`` event is written unless the measurements were recorded already. Requests
            are then marked applied.
        """
        while self._incoming:
            name, steps, detail = self._incoming.popleft()
            if self._manual.get(name, 0) <= self.step:
                self._put(('event', 0, self.step, 'record', {'measurements': name, 'steps': steps, **detail}, time.time()))
            self.record(name, steps)
            if 'request' in detail:
                self._put(('applied', 0, detail['request']))

    def record(self, name: str, steps: int = 1) -> None:
        """
            Records the measurements ``name`` for the next ``steps`` steps.

            The steps count from the current step. Measurements with any trigger can be recorded
            this way. Every call of the model holding one of those steps is recorded whole.
            Measurements with a group are recorded to the end of every group holding one of those
            steps, with one record per group. Called again before those steps end, the recording
            lasts until the later end.

            Parameters
            ----------
            name : str
                Name of the measurements.
            steps : int, default 1
                Steps to record them for.

            Warns
            -----
            RecordingWarning
                When there are no measurements ``name``, or ``steps`` is not a positive integer.
                Nothing is recorded.

            Notes
            -----
            With several processes, only the calls on process 0 take effect.

            Examples
            --------
            >>> recorder.record('activity', 500)  # the next 500 steps
            >>> recorder.record('episode')        # grouped by a tag: to the end of the episode
        """
        self._deliver_interrupt()
        if name not in self._by_name:
            self._warn(
                ('record', name), f'No measurements "{name}"; recording them is skipped. Expected one of: '
                f'{", ".join(r.name for r in self.measurements)}.', measurements=name,
            )
            return
        try:
            steps = integer(steps, 'steps', lowest=1)
        except ValueError as error:
            self._warn(('record steps', name), f'Recording "{name}" is skipped: {error}', measurements=name)
            return
        self._manual[name] = max(self._manual.get(name, 0), self.step + steps)

    def push(self, records: Packed | dict, steps: int) -> None:
        """
            Hands over the records of the call just dispatched and advances the step count.

            Call it after every call of the model, recorded or not. The transfers of ``records``
            start at once, and it returns without waiting for them.

            Parameters
            ----------
            records : Packed or dict
                The records the call returned. `Packed`, or an empty dictionary when nothing was
                recorded.
            steps : int
                Steps of the call. The steps given to `probes`, if any.

            Raises
            ------
            TypeError
                When ``records`` is neither `Packed` nor empty.
            ValueError
                When ``steps`` is not a positive integer or differs from the steps given to
                `probes`, or when ``records`` does not match the probes `probes` returned for the
                call.
            RuntimeError
                When the recorder is closed, or its writer failed with ``on_error='raise'``.

            Notes
            -----
            Blocks while the queue of the writer is full. The step count advances even when handing
            the records over is interrupted. With several processes, only process 0 hands records to
            the writer.
        """
        self._check()
        steps = integer(steps, 'steps', lowest=1)
        pending, self._pending = self._pending, None
        recorded, captured, probes, tags = pending if pending is not None else (frozenset(), frozenset(), (), dict(self.tags))
        if records and not isinstance(records, Packed):
            raise TypeError(
                f'`push` takes the records of the call as the call returned them, not {type(records).__name__}; '
                f'`Packed.unpack` is for reading them.'
            )
        if records and pending is None:
            raise ValueError('Records were pushed for a call without `probes`: nothing was recorded for it.')
        if probes and not records:
            raise ValueError('Probes were recorded for this call but no records were given.')
        if isinstance(records, Packed) and records.probes != probes:
            raise ValueError('The records are not those of the probes recorded for this call.')
        if self._pending_steps is not None and pending is not None and steps != self._pending_steps:
            raise ValueError(f'`probes` was given {self._pending_steps} steps for this call, and `push` {steps}.')
        try:
            if self._writes:
                self._hand_over(records, recorded, captured, steps, tags)
        finally:
            # The call ran: the counts advance even when handing its records over is interrupted.
            self.step += steps
            self._calls += 1
            self._last_steps = steps

    def _hand_over(self, records: Packed | dict, recorded: frozenset[str], captured: frozenset[str], steps: int, tags: dict[str, tp.Any]) -> None:
        """
            Queues the records of a call for the writer, with the groups they end.

            The groups of measurements no longer recorded end first. The calls kept by ``lookback``
            for the measurements recorded on this call are queued before it.
        """
        failed = self._writer is not None and self._writer.error is not None
        call = _Call(self.step, steps, recorded)
        if failed:
            # Nothing is recorded any more: no group goes on.
            self._current.clear()
        # The group of measurements no longer recorded has ended, a group by tag with the tag.
        for name in [n for n in self._current if n not in recorded]:
            self._put(self._end(name))
        for name, ring in self._rings.items():
            if name not in recorded:
                # Kept on the device until the measurements are recorded. A call run without `probes` captured nothing.
                if name in captured and not failed:
                    ring.append((records, dc.replace(call, recorded=frozenset({name})), tags))
                    while sum(kept.steps for _, kept, _ in ring) - ring[0][1].steps >= self._by_name[name].lookback:
                        ring.popleft()
                continue
            # Empty when the measurements were recorded for the previous call too. Kept, their steps count as recorded.
            for kept, kept_call, kept_tags in ring:
                self._queue_call(kept, kept_call, kept_tags, self._progress(), kept=True)
            ring.clear()
        recorded = records if recorded and not failed else None
        self._queue_call(recorded, call if recorded is not None else dc.replace(call, recorded=frozenset()), tags,
                     {'step': self.step + steps, 'stalls': self._stalls})

    def _queue_call(self, records: Packed | dict | None, call: _Call, tags: dict[str, tp.Any], progress: dict[str, int], kept: bool = False) -> None:
        """
            Queues one call for the writer, then the items ending the groups it ends.

            The call carries its groups. The steps of a call ``kept`` by `Measurements.lookback`
            count as recorded.
        """
        held = records.held if isinstance(records, Packed) else {}
        slots, ended = {}, []
        for name in sorted(call.recorded):
            measurements = self._by_name[name]
            if measurements.group is not None:
                slots[name], done = self._advance(measurements, call, held, tags, kept)
                ended += done
        recorded = _strip(records)
        if recorded is not None:
            self._transfer(recorded)
        self._put(('call', _nbytes(recorded), recorded, dc.replace(call, slots=slots), progress))
        for item in ended:
            self._put(item)

    def _requested_within(self, measurements: Measurements, first: int, stop: int, tags: dict[str, tp.Any]) -> bool:
        """
            Returns whether a step of ``[first, stop)`` is asked for ``measurements``.

            A step counts when the trigger, `record` or a request asks for it, and not when it is
            recorded only to end a group.
        """
        counters = {'step': first, **{k: int(v) for k, v in tags.items() if isinstance(v, (bool, int, np.integer))}}
        return measurements.trigger.recorded(counters, {'step': stop - first}) or self._manual.get(measurements.name, 0) > first

    def _advance(
            self, measurements: Measurements, call: _Call, held: dict[str, dict[str, tp.Any]], tags: dict[str, tp.Any], kept: bool = False,
        ) -> tuple[tuple, list[tuple]]:
        """
            Returns the groups a call falls in, and the items ending the groups it ends.

            Groups are ``(slot, group, first step, steps)``, and the items read the values the call
            kept on the device. A group is recorded from the first call holding a step of it asked
            for to its end. The other steps of a call recorded to end a group are in no group.
        """
        name, group = measurements.name, measurements.group
        boundary = [p for p in measurements.probes if isinstance(p, (SnapshotProbe, DeltaProbe))]
        deltas = [p for p in boundary if isinstance(p, DeltaProbe)]
        t, steps = call.t0, call.steps
        slots, ended = [], []
        current = self._current.get(name)
        if isinstance(group, str):
            tag = tags.get(group)
            if current is not None and not _same(current.tag, tag):
                ended.append(self._end(name))
                current = None
            if current is None and not (kept or self._requested_within(measurements, t, t + steps, tags)):
                return (), ended
            if current is None:
                current = self._current[name] = _Open(key=t, t0=t, tag=tag, start={p.key: (held[p.key]['first'], 0) for p in deltas})
            current.steps += steps
            current.last = {p.key: (held[p.key]['last'], 0) for p in boundary}
            return ((0, current.key, t, steps),), ended
        first_group = t // group
        for number in range(first_group, (t + steps - 1) // group + 1):
            first, stop = max(t, number * group), min(t + steps, (number + 1) * group)
            if current is not None and current.key != number:
                ended.append(self._end(name))
                current = None
            if current is None and not (kept or self._requested_within(measurements, first, stop, tags)):
                continue
            if current is None:
                # The value before the group: after the end of the one before, within the call, or before the call.
                start = {
                    p.key: (held[p.key]['ends'], number - first_group - 1) if number > first_group else (held[p.key]['first'], 0)
                    for p in deltas
                }
                current = self._current[name] = _Open(key=number, t0=first, start=start)
            current.steps += stop - first
            slots.append((number - first_group, number, first, stop - first))
            if stop == (number + 1) * group:
                # The value after the group: at its end within the call, or after the call.
                ends = {
                    p.key: (held[p.key]['last'], 0) if stop == t + steps else (held[p.key]['ends'], number - first_group)
                    for p in boundary
                }
                ended.append(self._end(name, ends))
                current = None
            else:
                current.last = {p.key: (held[p.key]['last'], 0) for p in boundary}
        return tuple(slots), ended

    def _end(self, name: str, values: dict[str, tp.Any] | None = None) -> tuple:
        """
            Ends the group of the measurements ``name`` and returns the item writing it.

            Its snapshots and deltas are computed from ``values``, the values after its last step by
            probe key. Without ``values``, the last values kept are used, as for a group by tag or a
            group cut short.
        """
        group = self._current.pop(name)
        values = group.last if values is None else values
        probes = tuple(
            p for p in self._by_name[name].probes if p.key in values and (isinstance(p, SnapshotProbe) or p.key in group.start)
        )
        boundary: dict[str, dict[str, tp.Any]] = {}
        if probes:
            starts = [group.start.get(p.key, (None, 0)) if isinstance(p, DeltaProbe) else (None, 0) for p in probes]
            results = group_values(
                probes, tuple(values[p.key][0] for p in probes), tuple(np.int32(values[p.key][1]) for p in probes),
                tuple(start for start, _ in starts), tuple(np.int32(row) for _, row in starts),
            )
            boundary = {p.key: {p.mode.value: value} for p, value in zip(probes, results)}
        self._transfer(boundary)
        return ('group', _nbytes(boundary), name, group.key, group.t0, group.steps, boundary)

    def _warm_groups(self, shapes: tp.Callable[[tuple[Probe, ...]], dict[str, tp.Any]], steps: int) -> None:
        """
            Compiles `group_values` for the snapshots and deltas with a group.

            The calls have ``steps`` steps. ``shapes`` gives the shapes of the values of a set of
            probes, by probe key, as `read_boundary` reads them. Nothing then compiles when a group
            ends.
        """
        grouped = merge_probes(p for r in self.measurements for p in r.probes if isinstance(p, BOUNDARY_PROBES) and p.group is not None)
        if not grouped:
            return
        values = shapes(grouped)
        row = jax.ShapeDtypeStruct((), np.int32)
        for measurements in self.measurements:
            probes = tuple(p for p in measurements.probes if isinstance(p, BOUNDARY_PROBES) and p.group is not None)
            if not probes:
                continue
            # A group ends after a call, or within one; a delta starts before a call, or within one.
            for within_end in (False, True):
                for within_start in (False, True):
                    ends, starts = [], []
                    for probe in probes:
                        value = values[probe.key]
                        kept = (len(probe.units),) if isinstance(probe, SnapshotProbe) and probe.units is not None else tuple(value.shape)
                        rows = lambda within: group_ends(probe, steps) + 1 if within and group_ends(probe, steps) else 1
                        ends.append(jax.ShapeDtypeStruct((rows(within_end), *kept), value.dtype))
                        starts.append(None if isinstance(probe, SnapshotProbe) else jax.ShapeDtypeStruct((rows(within_start), *value.shape), value.dtype))
                    group_values.lower(probes, tuple(ends), (row,) * len(probes), tuple(starts), (row,) * len(probes)).compile()

    def _end_all(self) -> None:
        """
            Writes the groups in progress, cut short, before the recorder closes.

            Warns when they cannot be written.
        """
        if not self._writes or self._writer is None or self._writer.error is not None:
            self._current.clear()
            return
        try:
            for name in list(self._current):
                self._put(self._end(name))
        except Exception as error:
            self._current.clear()
            warnings.warn(f'The groups in progress of "{self.path}" could not be written: {type(error).__name__}: {error}')

    def warmup_sets(self, steps: int | None = None) -> list[tuple[Probe, ...]]:
        """
            Returns the probe sets `spark.Jit.warmup` and `Runner.warmup` compile.

            The sets are those the triggers counting steps record over the next `SETTINGS.warmup_calls`
            calls of ``steps`` steps. Measurements grouped by steps stay recorded until their group
            ends. The measurements recorded otherwise are added to each set. These are the measurements
            recorded by `record`, requests, conditions or triggers counting a tag, and those grouped by
            a tag. The set of all measurements, which needs the most memory, is included. Measurements
            with a ``lookback`` are captured in every set.

            Parameters
            ----------
            steps : int, optional
                Steps of every call. Without it, all measurements count as recorded otherwise.

            Returns
            -------
            list of tuple of Probe
                The distinct probe sets, each the tuple `probes` returns for it.

            Notes
            -----
            The triggers counting steps are simulated over one period of their combined pattern. The
            simulation starts from the current step, from the offsets of `Every`, and from the steps
            where `At` and `Between` start and stop. When the measurements recorded otherwise make more
            than `SETTINGS.warmup_subsets` sets, each is added alone and all together, instead of in
            every combination.

            Measurements with a schedule of their own that `record` records as well, as from the
            viewer, can record a set not compiled ahead. That set compiles when it first occurs.
        """
        # Measurements grouped by a tag stay recorded while the tag keeps its value, unless recorded on every step.
        scheduled = [
            r for r in self.measurements
            if steps and type(r.trigger) in (Every, At, Between, Always) and r.trigger.tag is None
            and (not isinstance(r.group, str) or isinstance(r.trigger, Always))
        ]
        free = sorted({r.name for r in self.measurements} - {r.name for r in scheduled})
        seen: dict[frozenset[str], None] = {}
        if scheduled:
            steps = int(steps)

            def call_at(step: int, end: bool = False) -> int:
                """
                    Returns the number of calls from now to the call holding ``step``.

                    With ``end``, counts to the first call starting at or after ``step``.
                """
                offset = step - self.step
                return max(-(-offset // steps), 0) if end else max(offset, 0) // steps

            period, starts, lead = 1, {0}, 1
            for measurements in scheduled:
                trigger = measurements.trigger
                if isinstance(trigger, Every):
                    period = min(math.lcm(period, trigger.n // math.gcd(trigger.n, steps)), SETTINGS.warmup_calls)
                    starts.add(call_at(trigger.offset))
                elif isinstance(trigger, At):
                    starts.update(call_at(p) for p in trigger.points)
                    starts.update(call_at(p + trigger.length, end=True) for p in trigger.points)
                elif isinstance(trigger, Between):
                    starts.add(call_at(trigger.start))
                    if trigger.stop is not None:
                        starts.add(call_at(trigger.stop, end=True))
                if isinstance(measurements.group, int):
                    period = min(math.lcm(period, measurements.group // math.gcd(measurements.group, steps)), SETTINGS.warmup_calls)
                    lead = max(lead, -(-measurements.group // steps) + 1)
            simulated = 0
            spans = {'step': steps}
            for first in sorted(starts):
                if simulated >= SETTINGS.warmup_calls:
                    break
                # The step the group open ends at, by measurements; a stretch starts with no group open.
                until: dict[str, int] = {}
                for call in range(first, first + period + lead):
                    step = self.step + call * steps
                    recorded = frozenset(
                        r.name for r in scheduled if r.trigger.recorded({'step': step}, spans) or until.get(r.name, -1) > step
                    )
                    for measurements in scheduled:
                        if measurements.name not in recorded or not isinstance(measurements.group, int):
                            continue
                        k = measurements.group
                        if until.get(measurements.name, -1) <= step + steps:
                            until.pop(measurements.name, None)
                        # A group holding a step asked for opens, and keeps the measurements recorded until it ends.
                        for number in range((step + steps - 1) // k, step // k - 1, -1):
                            first_step, stop = max(step, number * k), min(step + steps, (number + 1) * k)
                            if measurements.trigger.recorded({'step': first_step}, {'step': stop - first_step}):
                                if (number + 1) * k > step + steps:
                                    until[measurements.name] = (number + 1) * k
                                break
                    seen.setdefault(recorded, None)
                    simulated += 1
        else:
            seen[frozenset()] = None
        if len(seen) * 2 ** len(free) <= SETTINGS.warmup_subsets:
            extra = [frozenset(c) for n in range(len(free) + 1) for c in itertools.combinations(free, n)]
        else:
            extra = [frozenset(), *(frozenset({f}) for f in free), frozenset(free)]
        combined = [recorded | more for recorded in seen for more in extra]
        combined.append(frozenset(r.name for r in self.measurements))
        always = frozenset(self._rings)
        return list(dict.fromkeys(self._merged(recorded | always, handed=False) for recorded in combined))

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    def log(self, values: dict[str, tp.Any] | None = None, *, step: int | None = None, tag: str | None = None, **scalars: tp.Any) -> None:
        """
            Writes scalars held on the host, at the current step or at ``step``.

            A series is logged per step, or per the integer tag ``tag``, such as once per episode.
            Its tag is written with its name once, read by `Run.tag_of`; the run viewer draws the
            series against the values of the tag. Scalars logged per a tag are written at the step
            the tag took its value, where the summaries grouped by the tag are, unless ``step`` is
            given.

            Parameters
            ----------
            values : dict of str to float, optional
                Scalars by name.
            step : int, optional
                Step of the scalars. The current step by default, or the step the tag ``tag`` took
                its value.
            tag : str, optional
                Name of the integer tag, set with `tag`, the scalars are logged per. Per step
                without one.
            **scalars : float
                Scalars by name, added to ``values``. Scalars named ``step`` or ``tag`` are given in
                ``values``.

            Raises
            ------
            RuntimeError
                When the recorder is closed, or its writer failed with ``on_error='raise'``.

            Warns
            -----
            RecordingWarning
                When a name is not a non-empty string, or a value is not a number. That value is
                dropped. When ``tag`` is not a name, or is ``'step'``: the scalars are logged per
                step. When ``tag`` is not set to an integer yet: the rows logged before it is have
                no value of it. When a series is logged per another tag than before: it keeps the
                first.

            Examples
            --------
            >>> recorder.log(reward=1.0, episode_steps=212)
            >>> recorder.log({'episode/steps': 212}, tag='episode')
        """
        self._deliver_interrupt()
        if tag is not None and (not isinstance(tag, str) or not tag or tag == 'step'):
            self._warn(('log tag', repr(tag)), f'Scalars are logged per step, or per an integer tag named by a string other than "step", '
                                               f'got {tag!r}; they are logged per step.')
            tag = None
        if tag is not None and not isinstance(self.tags.get(tag), (bool, int, np.integer)):
            self._warn(('log tag unset', tag), f'Scalars are logged per the tag "{tag}", which is not set to an integer yet; the rows '
                                               f'logged before it is have no value of it.', tag=tag)
        values = {**(values or {}), **scalars}
        kept = {}
        for key, value in values.items():
            if not isinstance(key, str) or not key:
                self._warn(('log name', repr(key)), f'Scalars are named by non-empty strings, got {key!r}; its value is dropped.')
                continue
            try:
                kept[key] = float(value)
            except (TypeError, ValueError):
                self._warn(('log value', key), f'The scalar "{key}" takes a number, got {type(value).__name__}; the value is dropped.', scalar=key)
        at = self.step if step is None else int(step)
        if tag is not None and step is None and isinstance(self.tags.get(tag), (bool, int, np.integer)):
            at = self._tag_steps.get(tag, self.step)
        if kept:
            self._put(('scalars', 0, at, kept, time.time(), tag))

    def event(self, kind: str, *, step: int | None = None, **payload: tp.Any) -> None:
        """
            Writes an event, such as the end of an episode, with a JSON payload.

            Parameters
            ----------
            kind : str
                Kind of the event.
            step : int, optional
                Step of the event. The current step by default.
            **payload
                Payload of the event, written as JSON. It takes no key of `RESERVED_EVENT_KEYS`.

            Raises
            ------
            RuntimeError
                When the recorder is closed, or its writer failed with ``on_error='raise'``.

            Warns
            -----
            RecordingWarning
                When the payload holds a key of `RESERVED_EVENT_KEYS`. The key is dropped.
        """
        self._deliver_interrupt()
        reserved = [key for key in payload if key in RESERVED_EVENT_KEYS]
        if reserved:
            self._warn(
                ('event', kind), f'The payload of an event takes no keys {reserved}; they name the fields of the event. They are '
                f'dropped from the events "{kind}".', event=kind,
            )
            payload = {key: value for key, value in payload.items() if key not in RESERVED_EVENT_KEYS}
        self._put(('event', 0, self.step if step is None else int(step), kind, payload, time.time()))

    def tag(self, **tags: tp.Any) -> None:
        """
            Sets tags on the timeline, such as ``episode=3``.

            The tags are written at the current step. Integer tags can be counted by triggers.
            Measurements grouped by a tag, as ``group='episode'``, start a new group of steps each
            time the tag takes a different value. An integer held in a NumPy or JAX array without
            dimensions is read as an int.

            Parameters
            ----------
            **tags
                Values by tag name.

            Raises
            ------
            RuntimeError
                When the recorder is closed, or its writer failed with ``on_error='raise'``.
        """
        self._deliver_interrupt()
        for key, value in tags.items():
            if not isinstance(value, (bool, int)) and getattr(value, 'shape', None) == () and np.issubdtype(getattr(value, 'dtype', object), np.integer):
                tags[key] = int(value)
        for key, value in tags.items():
            if key not in self.tags or self.tags[key] != value:
                self._tag_steps[key] = self.step
        self.tags.update(tags)
        self._put(('tags', 0, self.step, dict(tags), time.time()))

    def raw(self, name: str, frame: tp.Any, *, step: int | None = None) -> None:
        """
            Keeps a frame of a raw stream when measurements declaring the stream are recorded.

            The frame is copied and kept when measurements declaring ``name`` are recorded for the
            call starting at the current step. That call is the one being run between `probes` and
            `push`, or else the next one. When none are recorded, nothing is kept. It can be called
            on every step of an environment.

            Parameters
            ----------
            name : str
                Name of the raw stream, as declared by `Measurements`.
            frame : array-like
                Array of numbers or bools, such as an observation.
            step : int, optional
                Step of the frame. The current step by default.

            Raises
            ------
            RuntimeError
                When a frame is kept after the recorder closed, or after its writer failed with
                ``on_error='raise'``.

            Warns
            -----
            RecordingWarning
                When no measurements declare ``name``, when ``frame`` holds no numbers or bools, or
                when it differs in shape or dtype from the first frame of ``name``. The frame is
                dropped.

            Notes
            -----
            With several processes, only process 0 keeps frames.

            Measurements declaring ``name`` that are recorded over a group, or over calls one after
            the other, without a frame of it also give a `RecordingWarning`, from the next call of
            the model.
        """
        self._deliver_interrupt()
        if name not in self._raw:
            self._warn(
                ('raw', name), f'No measurements declare the raw stream "{name}"; its frames are dropped. Declared: '
                f'{", ".join(self._raw) or "none"}.', raw=name,
            )
            return
        if not (hasattr(frame, 'shape') and hasattr(frame, 'dtype')):
            try:
                frame = np.asarray(frame)
            except (TypeError, ValueError):
                frame = np.asarray(frame, dtype=object)
        found = (tuple(frame.shape), np.dtype(frame.dtype))
        if found[1].kind not in 'biufc':
            self._warn(('raw dtype', name), f'A frame of "{name}" holds {found[1]}; frames are arrays of numbers or bools. It is dropped.', raw=name)
            return
        expected = self._raw_formats.setdefault(name, found)
        if found != expected:
            self._warn(
                ('raw format', name), f'Frames of "{name}" have shape {expected[0]} and dtype {expected[1]}, got {found[0]} and '
                f'{found[1]}. Such frames are dropped.', raw=name,
            )
            return
        if not self._writes:
            return
        recorded = self._recorded_here()
        measurements = tuple(r for r in self._raw[name] if r in recorded)
        if measurements:
            frame = np.array(frame, copy=True)
            self._put(('raw', frame.nbytes, self.step if step is None else int(step), name, frame, measurements))

    def is_recording(self, name: str) -> bool:
        """
            Returns whether the measurements ``name`` are recorded for the call at the current step.

            That call is the one being run between `probes` and `push`, or else the next one, as for
            `raw`.

            Parameters
            ----------
            name : str
                Name of the measurements.

            Returns
            -------
            bool

            Notes
            -----
            With several processes, processes other than process 0 know it between `probes` and
            `push` only.

            Warns
            -----
            RecordingWarning
                When there are no measurements ``name``. Returns False.
        """
        if name not in self._by_name:
            self._warn(
                ('record', name), f'No measurements "{name}"; they are never recorded. Expected one of: '
                f'{", ".join(r.name for r in self.measurements)}.', measurements=name,
            )
            return False
        return name in self._recorded_here()

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    def checkpoint(self, model: Controller, *, step: int | None = None) -> pathlib.Path:
        """
            Saves a model to ``checkpoints/<step>.spark`` in the run, with `Controller.checkpoint`.

            The state of the model is copied to the host before it returns, and the file is written
            in the background. A ``checkpoint`` event is written when the save starts.
            `Run.checkpoints` lists the checkpoints written completely, and `Run.restore` reads them
            back.

            Parameters
            ----------
            model : Controller
                The model, such as ``spark.merge(graph, state)`` in a loop over its state.
            step : int, optional
                Step the checkpoint is filed under. The current step by default.

            Returns
            -------
            pathlib.Path
                File of the checkpoint.

            Raises
            ------
            RuntimeError
                When the recorder is closed, its writer failed with ``on_error='raise'``, or the
                previous checkpoint could not be written.

            Notes
            -----
            A save first waits for the previous one to be written, and `close` waits for the last
            one. A checkpoint at the same step is replaced. With several processes, every process
            calls it, the state is gathered on each of them, and process 0 writes it.

            Examples
            --------
            >>> recorder.checkpoint(spark.merge(graph, state))
        """
        self._check()
        step = self.step if step is None else int(step)
        path = store.checkpoint_file(self.path, step)
        self._wait_checkpoint()
        # On the host before the next call, which may donate the state. A state spread over several
        # processes is gathered by all of them.
        graph, state = split((model))
        state = jax.tree.map(_on_host, state)
        if self._writes:
            model = merge(graph, state)
            if self._checkpointer is None:
                self._checkpointer = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix='spark-checkpoint')
            self._checkpointing = self._checkpointer.submit(model.checkpoint, path, overwrite=True, verbose=False)
        self.event('checkpoint', step=step, path=path.relative_to(self.path).as_posix())
        return path

    def _wait_checkpoint(self) -> None:
        """
            Waits for the checkpoint being written, if any, and raises the error that stopped it.
        """
        pending, self._checkpointing = self._checkpointing, None
        if pending is not None:
            pending.result()

    def flush(self) -> None:
        """
            Writes everything handed over so far and waits until it is written.

            The open windows are written as they stand. Each call ends the current file of every set
            of measurements.

            Raises
            ------
            RuntimeError
                When the recorder is closed, or its writer failed with ``on_error='raise'``.

            Notes
            -----
            With several processes, it returns at once on processes other than process 0.
        """
        self._check()
        if self._writer is None:
            return
        self._put(('flush', 0))
        self._queue.join()
        self._check()

    def close(self, status: str = 'finished', error: str | None = None) -> None:
        """
            Writes what is left, marks the run with ``status`` and releases it.

            The groups in progress are written cut short, and the last checkpoint is waited for. A
            recorder that received one of its signals marks the run ``'preempted'`` in place of
            ``'finished'``. Does nothing when closed already.

            Parameters
            ----------
            status : str, default 'finished'
                Status written to ``run.json``.
            error : str, optional
                Error written to ``run.json``.

            Raises
            ------
            RuntimeError
                When the writer failed, with ``on_error='raise'``.

            Notes
            -----
            An interrupt (Ctrl-C) received while the writer writes what is left is raised once it is
            done. A second interrupt stops the wait. An interrupt `spark.jit` held during the last call
            is dropped: the loop it would have stopped is over.
        """
        self._interrupt = None
        if self._finished:
            return
        if self._signal is not None and status == 'finished':
            status, error = 'preempted', str(Preempted(self._signal))
        first = not self._closed
        try:
            if first:
                self._end_all()
        finally:
            self._closed = True
            try:
                if first and self._checkpointer is not None:
                    try:
                        self._wait_checkpoint()
                    finally:
                        self._checkpointer.shutdown()
            finally:
                self._finish(status, error)
        if self._writer is not None and self._writer.error is not None and self._options.on_error == 'raise':
            raise RuntimeError(f'The writer of "{self.path}" failed.') from self._writer.error
        self._give_notices()

    def _finish(self, status: str, error: str | None) -> None:
        """
            Closes the writer and releases the run.

            An interrupt received while the writer writes what is left is raised once it is done.
            The signals of the recorder stay handled until then.
        """
        interrupted: KeyboardInterrupt | None = None
        if self._writer is not None:
            while not self._close_queued and not self._writer.done.is_set():
                try:
                    self._queue.put(('close', 0, status, error, self._progress()), timeout=0.5)
                    self._close_queued = True
                except queue.Full:
                    pass
                except KeyboardInterrupt as interrupt:
                    interrupted = self._interrupted(interrupt, interrupted)
            while not self._writer.done.is_set():
                try:
                    self._writer.done.wait(0.5)
                except KeyboardInterrupt as interrupt:
                    interrupted = self._interrupted(interrupt, interrupted)
        if self._lock is not None:
            store.unlock(self._lock)
            self._lock = None
        _OPEN.pop(self, None)
        self._finished = True
        # A signal received after the last call, never raised, takes its usual effect now the run is closed.
        _SIGNALS.remove(self, deliver=self._signal if not self._raised else None)
        if interrupted is not None:
            raise interrupted

    def _interrupted(self, interrupt: KeyboardInterrupt, before: KeyboardInterrupt | None) -> KeyboardInterrupt:
        """
            Holds the first interrupt received while closing, with a warning, and raises the second.

            After the second interrupt, the writer goes on alone, as when the file system hangs.
        """
        if before is not None:
            self._stopped_waiting = True
            raise interrupt
        return _held(interrupt, self.path)

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, kind, error, traceback) -> None:
        if isinstance(error, Preempted):
            self.close(status='preempted', error=str(error))
        elif self._signal is not None:
            self.close(status='preempted', error=f'{kind.__name__}: {error}' if error is not None else str(Preempted(self._signal)))
        elif error is None or (isinstance(error, SystemExit) and error.code in (0, None)):
            self.close()
        else:
            self.close(status='failed', error=f'{kind.__name__}: {error}')

    def __repr__(self) -> str:
        return f'Recorder("{self.path}", {len(self.measurements)} measurements, step {self.step})'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _close_at_exit() -> None:
    # An uncaught exception of a script is left in sys.last_exc before exit handlers run; a notebook keeps
    # the last error of any cell there. A SystemExit is not left there.
    error = None if 'ipykernel' in sys.modules else getattr(sys, 'last_exc', None) or getattr(sys, 'last_value', None)
    interrupted = None
    # A signal the recorders received after their last call takes its usual effect once they are all closed.
    _SIGNALS.postponed = []
    try:
        for recorder in list(_OPEN):
            if recorder._stopped_waiting and recorder._writer is not None and not recorder._writer.done.is_set():
                continue                                                    # interrupted twice while closing
            try:
                if recorder._closed:
                    recorder._finish('failed', 'Interrupted while closing.')             # the status given to close stands
                elif recorder._signal is not None:
                    recorder.close('preempted', str(Preempted(recorder._signal)))
                elif error is not None:
                    recorder.close('failed', f'{type(error).__name__}: {error}')
                else:
                    recorder.close()
            except KeyboardInterrupt as interrupt:
                interrupted = interrupt
                if recorder._stopped_waiting:
                    break
            except Exception:
                pass
    finally:
        postponed, _SIGNALS.postponed = _SIGNALS.postponed, None
        for signum, before in postponed:
            deliver_signal(signum, before)
    if interrupted is not None:
        raise interrupted

#-----------------------------------------------------------------------------------------------------------------------------------------------#

_SIGNALS = _Signals()
"""
    The signal handlers of the open recorders.
"""

# Closed before the thread pools of the interpreter stop, which the checkpoints written in the background need.
# Registered after `concurrent.futures`: of these handlers, those registered later run first.
importlib.import_module('concurrent.futures.thread')
try:
    threading._register_atexit(_close_at_exit)
except (AttributeError, RuntimeError):
    pass
atexit.register(_close_at_exit)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
