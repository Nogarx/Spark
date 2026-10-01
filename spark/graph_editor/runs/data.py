#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import json
import time
import sqlite3
import pathlib
import collections

import numpy as np

from spark.recording.run import Run, Window
from spark.recording.probe import Probe, SummaryProbe, DeltaProbe, CALL
from spark.recording.measurements import Measurements, scalar_key

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

POINTS = 100000
"""
    Approximate number of rows of a scalar series kept in memory. A longer series is kept as its
    envelope, reduced again once it holds more than twice this number of rows.
"""

RAW_FAILURES = 8
"""
    Number of windows that fail to be read after which a search for a raw frame stops.
"""

RETRY_READS = 5.0
"""
    Seconds for which a search for a raw frame skips a window that could not be read.
"""

READ_ERRORS = (sqlite3.Error, OSError, ValueError, KeyError)
"""
    Errors from reading a run that the viewer reports and continues after. They cover a run moved,
    deleted or being replaced, and file system errors.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Table:
    """
        Numeric columns of rows ordered by the first column, grown in place.

        Buffers double when full. Rows that arrive out of order are sorted into new buffers, and
        arrays returned before keep their content.
    """

    def __init__(self, *dtypes: tp.Any) -> None:
        self._columns = [np.zeros(16, dtype) for dtype in dtypes]
        self.size = 0
        # Counts the times the rows were sorted again or reduced: rows before ``size`` then changed.
        self.version = 0

    def extend(self, rows: tp.Any) -> None:
        """
            Appends rows, given as a ``(rows, columns)`` array or a list of tuples.
        """
        data = np.asarray(rows, dtype=np.float64).reshape(-1, len(self._columns))
        n, size = len(data), self.size
        if not n:
            return
        if size + n > len(self._columns[0]):
            capacity = max(2 * len(self._columns[0]), size + n)
            self._columns = [np.concatenate([c[:size], np.zeros(capacity - size, c.dtype)]) for c in self._columns]
        first = data[:, 0]
        ordered = bool(np.all(first[1:] >= first[:-1])) and (size == 0 or first[0] >= self._columns[0][size - 1])
        for column, values in zip(self._columns, data.T):
            column[size:size + n] = values
        self.size += n
        if not ordered:
            order = np.argsort(self._columns[0][:self.size], kind='stable')
            self._columns = [c[:self.size][order] for c in self._columns]
            self.version += 1

    def columns(self) -> tuple[np.ndarray, ...]:
        return tuple(c[:self.size] for c in self._columns)

    def reduce(self, points: int) -> None:
        """
            Reduces more than ``points`` rows to their envelope, as `Run.scalar` does.

            Keeps the rows with the lowest and the highest second column in each of ``points // 2``
            equal bins of the first column, and the first NaN of each bin holding one.
        """
        if self.size <= points:
            return
        first, second = self.columns()[:2]
        width = (first[-1] - first[0]) // max(points // 2, 1) + 1
        span = (first - first[0]) // width
        keep = []
        nan = np.isnan(second)
        if nan.any():
            where = np.flatnonzero(nan)
            keep.append(where[np.r_[True, span[where][1:] != span[where][:-1]]])
        finite = np.flatnonzero(~nan)
        if len(finite):
            order = finite[np.lexsort((second[finite], span[finite]))]
            ordered = span[order]
            change = ordered[1:] != ordered[:-1]
            keep += [order[np.r_[True, change]], order[np.r_[change, True]]]
        keep = np.unique(np.concatenate(keep))
        self._columns = [c[:self.size][keep] for c in self._columns]
        self.size = len(keep)
        self.version += 1

_NONE = _Table(np.int64, np.int64, np.int64)
"""
    Empty table of spans or groups, for measurements with none listed.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RunData:
    """
        Object used to serve data to a RunViewer. 
        
        RunData wraps a `spark.recording.Run` to stay up to date while the run is being written.

        Events, spans, groups, and windows are read once and each `refresh` reads the new rows added since. 
        Values are sorted up by node of the graph and by step.

        Parameters
        ----------
        path : str or path-like
            Directory of the run.
        cached_bytes : int, default 256 MiB
            Memory kept for arrays read from window files.

        Attributes
        ----------
        run : Run
            The run read.
        path : pathlib.Path
            Directory of the run.
        config : SparkConfig or None
            Configuration of the model recorded, or None when it cannot be loaded.
        config_error : str or None
            Error of loading the configuration, or None.
        dt : float
            Duration of a step, in milliseconds, from the configuration. 1.0 without one.
        events : list of dict
            Events read so far, each as ``{**payload, 't', 'kind', 'wall'}``.
        windows : list of Window
            Windows read so far.
        error : str or None
            Error of the last read that failed, as ``'Type: message'``, until a `refresh` succeeds.

        Notes
        -----
        The arrays read from window files are kept up to ``cached_bytes``, and the least recently
        used are dropped first. When the run is resumed, or a commit read before is rolled back,
        everything read is dropped and read again. When more than ``4 * POINTS`` scalar rows were
        written since the last read, the series read so far are dropped and read again as envelopes.

        See Also
        --------
        RunViewerWindow : The window showing a run over the graph of its model.
        spark.recording.Run : A run written by a `Recorder`, read back.
    """

    def __init__(self, path: str | pathlib.Path, cached_bytes: int = 256 * 2 ** 20) -> None:
        # Opened without `load`, which warns of the warnings of the run: the viewer lists them.
        self.run: Run = Run(path)
        self.path = self.run.path
        self.config = None
        self.config_error: str | None = None
        try:
            self.config = self.run.config()
        except Exception as error:
            self.config_error = f'{type(error).__name__}: {error}'
        self.dt = float(getattr(self.config, 'dt', 1.0) or 1.0)
        self.error: str | None = None
        # Series read so far, by key; those kept as their envelope; and the names of every series.
        self.scalars: dict[str, _Table] = {}
        self._reduced: set[str] = set()
        self._keys: list[str] = []
        self._cache: collections.OrderedDict[tuple[str, int, str], np.ndarray | None] = collections.OrderedDict()
        self._cached_bytes = 0
        self._cache_limit = int(cached_bytes)
        # Windows that could not be read, by the time they failed; tried again after `RETRY_READS` seconds.
        self._unreadable: dict[tuple[str, int], float] = {}
        self._status: str | None = None
        self._resumed = len(self.run.info.get('resumed') or [])
        self._reset()
        self.refresh()

    def _reset(self) -> None:
        """
            Drops the rows read so far. The next read starts from the first row of every table.
        """
        self.scalars.clear()
        self._reduced.clear()
        self.events: list[dict[str, tp.Any]] = []
        self.windows: list[Window] = []
        self._after = {'events': 0, 'windows': 0, 'spans': 0, 'groups': 0, 'scalars': 0}
        # Spans and groups listed by measurements: first step, steps and window number.
        self._recorded: dict[str, _Table] = {}
        self._groups: dict[str, _Table] = {}
        self._windows: dict[str, dict[int, Window]] = {}
        # Joined spans of every set of measurements, as (spans, 2) arrays, with the size and the version of the table of
        # spans they were made from.
        self._segments: dict[str, tuple[np.ndarray, int, int]] = {}
        self._last = 0

    @property
    def measurements(self) -> dict[str, Measurements]:
        """
            The measurements of the run, by name.
        """
        return self.run.measurements

    @property
    def status(self) -> str:
        """
            Status of the run, as given by `Run.status`.
        """
        return self.run.status

    @property
    def step(self) -> int:
        """
            Number of steps written so far.

            The larger of `Run.step` and the end of the last window read.
        """
        return max(self.run.step, self._last)

    def refresh(self) -> set[str]:
        """
            Reads what was written since the last call.

            A read that fails with one of `READ_ERRORS` is kept in `error`. What was read before the
            failure is kept.

            Returns
            -------
            set of str
                What changed, among ``'info'``, ``'scalars'``, ``'events'`` and ``'windows'``. Holds
                ``'error'`` when the read failed, or succeeded after a failure.
        """
        changed: set[str] = set()
        try:
            with self.run.reading():
                self._read(changed)
        except READ_ERRORS as error:
            self.error = f'{type(error).__name__}: {error}'
            changed.add('error')
        else:
            if self.error is not None:
                self.error = None
                changed.add('error')
        return changed

    def _read(self, changed: set[str]) -> None:
        before = (self.run.info.get('status'), self.run.info.get('step'), self.run.info.get('heartbeat'))
        try:
            self.run.refresh()
        except (OSError, ValueError):
            if not (self.path / 'run.json').exists():
                raise
            # run.json being replaced; read on the next call.
        if (self.run.info.get('status'), self.run.info.get('step'), self.run.info.get('heartbeat')) != before:
            changed.add('info')
        resumed = len(self.run.info.get('resumed') or [])
        rolled_back = any(self.run._last(table) < after for table, after in self._after.items())
        if resumed != self._resumed or rolled_back:
            # Resuming marks lost spans in place, in rows read before, and rolls back a commit a process that
            # ended left unfinished, whose rows may have been read: everything is read again.
            self._resumed = resumed
            self._reset()
            changed.update(('windows', 'scalars', 'events'))
        # A run that stops without closing turns crashed as its heartbeat ages, with nothing written.
        status = self.run.status
        if status != self._status:
            self._status = status
            self._segments.clear()
            changed.add('info')
        keys = self.run.scalar_keys()
        if keys != self._keys:
            self._keys = keys
            changed.add('scalars')
        if self.scalars and self.run._last('scalars') - self._after['scalars'] > 4 * POINTS:
            # Many rows written since the last read, as after a long outage: the series shown are read again,
            # as envelopes, rather than every row since.
            self.scalars.clear()
            self._reduced.clear()
            changed.add('scalars')
        last, rows = self.run.scalar_rows(self._after['scalars'], keys=list(self.scalars))
        for key, data in rows.items():
            table = self.scalars[key]
            table.extend(data)
            if table.size > 2 * POINTS:
                table.reduce(POINTS)
                self._reduced.add(key)
        if rows:
            changed.add('scalars')
        self._after['scalars'] = last
        for rowid, t, kind, payload, wall in self.run.rows('events', self._after['events']):
            self.events.append({**json.loads(payload), 't': t, 'kind': kind, 'wall': wall})
            self._after['events'] = rowid
            changed.add('events')
        for table, tables in (('spans', self._recorded), ('groups', self._groups)):
            for page in self.run._row_pages(table, self._after[table]):
                added: dict[str, list[tuple[int, int, int]]] = {}
                for rowid, measurements, number, t0, steps in page:
                    if number >= 0:                                     # -1: lost with the process filling its window
                        added.setdefault(measurements, []).append((t0, steps, number))
                for measurements, entries in added.items():
                    tables.setdefault(measurements, _Table(np.int64, np.int64, np.int64)).extend(entries)
                    changed.add('windows')
                self._after[table] = page[-1][0]
        for rowid, measurements, number, t0, t1, spans, groups, file, raw, _ in self.run.rows('windows', self._after['windows']):
            window = Window(measurements, number, t0, t1, spans, groups, self.path / file, _raw(raw))
            self.windows.append(window)
            self._windows.setdefault(measurements, {})[number] = window
            if self._status != 'running':
                self._segments.pop(measurements, None)                     # spans of written windows only
            self._last = max(self._last, t1)
            self._after['windows'] = rowid
            changed.add('windows')

    def release(self) -> None:
        """
            Drops the arrays and rows read so far.

            Called when the viewer window closes.
        """
        self._cache.clear()
        self._cached_bytes = 0
        self._reset()

    def probes_of(self, node: str) -> list[tuple[str, Probe]]:
        """
            Returns the probes addressing a node of the graph.

            Parameters
            ----------
            node : str
                Name of a module, or of an input of the model.

            Returns
            -------
            list of (str, Probe)
                Name of the measurements and probe, for every probe whose path starts at the module
                ``node`` or that reads the input ``node`` of the model.
        """
        found = []
        for name, measurements in self.measurements.items():
            for probe in measurements.probes:
                if probe.path and probe.path[0] == node:
                    found.append((name, probe))
                elif probe.path == (CALL,) and probe.name == node:
                    found.append((name, probe))
        return found

    def scalar_keys(self) -> list[str]:
        """
            Returns the names of every scalar series, as of the last `refresh`.

            Returns
            -------
            list of str
        """
        return list(self._keys)

    def scalar(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """
            Returns the steps and values of a scalar series, ordered by step.

            The series is read from the run on the first call and extended by each `refresh`. A
            series of more than `POINTS` rows is kept as its envelope.

            Parameters
            ----------
            key : str
                Name of the series.

            Returns
            -------
            steps : ndarray
                Step of every row.
            values : ndarray
                Value of every row.

            Notes
            -----
            An unknown key gives empty arrays. So does a read that fails, which is kept in `error`.
            A series held in memory is reduced to its envelope again once it has more than twice
            `POINTS` rows.
        """
        series = self.scalars.get(key)
        if series is None:
            if key not in self._keys:
                return np.zeros(0, np.int64), np.zeros(0, np.float64)
            try:
                steps, values, reduced = self.run._scalar(key, POINTS, self._after['scalars'])
            except READ_ERRORS as error:
                self.error = f'{type(error).__name__}: {error}'
                return np.zeros(0, np.int64), np.zeros(0, np.float64)
            series = _Table(np.int64, np.float64)
            series.extend(np.column_stack([steps, values]))
            self.scalars[key] = series
            if reduced:
                self._reduced.add(key)
        return series.columns()

    def value_at(self, key: str, t: int) -> float | None:
        """
            Returns the last value of a scalar series at or before a step.

            A series held whole in memory gives the value from memory. Otherwise the value is read
            from the run, without reading the series.

            Parameters
            ----------
            key : str
                Name of the series.
            t : int
                Step.

            Returns
            -------
            float or None
                The value, or None when the series has none at or before ``t`` or the read fails.
        """
        if key not in self.scalars or key in self._reduced:
            try:
                return self.run.scalar_at(key, t)
            except READ_ERRORS:
                return None
        times, values = self.scalar(key)
        index = int(np.searchsorted(times, t, side='right')) - 1
        return float(values[index]) if index >= 0 else None

    def logged_keys(self) -> list[str]:
        """
            Returns the names of the scalar series logged with `Recorder.log`.

            These are the series of `scalar_keys` that no probe of the measurements wrote.

            Returns
            -------
            list of str
        """
        recorded = {
            scalar_key(name, probe.key, reduction)
            for name, measurements in self.measurements.items() for probe in measurements.probes
            if isinstance(probe, (SummaryProbe, DeltaProbe)) for reduction in probe.reduce
        }
        return [key for key in self._keys if key not in recorded]

    def activity(self, node: str, t: int) -> float | None:
        """
            Returns the firing rate of a node at a step, in Hz.

            The rate is the mean of the ``active_fraction`` reductions of the summaries of the output
            ports of the node whose name holds ``spike``. Each is taken at its last value at or before ``t``.

            Parameters
            ----------
            node : str
                Name of the node.
            t : int
                Step.

            Returns
            -------
            float or None
                The rate, or None when no such summary has a value at or before ``t``.

            Notes
            -----
            The ``active_fraction`` reduction gives spikes per unit and step. It is converted to Hz with `dt`.
        """
        values = []
        for measurements, probe in self.probes_of(node):
            # One node has few such probes; each is one lookup in the index.
            if not isinstance(probe, SummaryProbe) or 'active_fraction' not in probe.reduce or probe.kind != 'port':
                continue
            # Inputs of the node are ports too; they are not its activity.
            if CALL in probe.path or 'spike' not in probe.name:
                continue
            value = self.value_at(scalar_key(measurements, probe.key, 'active_fraction'), t)
            if value is not None:
                values.append(value)
        return float(np.mean(values)) / self.dt * 1000.0 if values else None

    def recorded(self, measurements: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Returns the spans listed for a set of measurements, ordered by first step.

            One span is listed per call of the model. Spans lost with the process writing their
            window are left out.

            Parameters
            ----------
            measurements : str
                Name of the measurements.

            Returns
            -------
            t0 : ndarray
                First step of every span.
            steps : ndarray
                Steps of every span.
            number : ndarray
                Window number of every span.
        """
        return self._recorded.get(measurements, _NONE).columns()

    def groups(self, measurements: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Returns the groups listed for a set of measurements, ordered by first step.

            Groups lost with the process writing their window are left out.

            Parameters
            ----------
            measurements : str
                Name of the measurements.

            Returns
            -------
            t0 : ndarray
                First step recorded in every group.
            steps : ndarray
                Steps of every group.
            number : ndarray
                Window number of every group.
        """
        return self._groups.get(measurements, _NONE).columns()

    def written(self, measurements: str, number: int) -> bool:
        """
            Returns whether the file of a window is written.

            A live run lists its spans and groups before it writes their window.

            Parameters
            ----------
            measurements : str
                Name of the measurements.
            number : int
                Window number.

            Returns
            -------
            bool
        """
        return number in self._windows.get(measurements, {})

    def spans(self, measurements: str) -> np.ndarray:
        """
            Returns the steps recorded by a set of measurements, as joined spans.

            Overlapping and consecutive spans are joined. While the run is written, spans listed
            ahead of their window count as recorded, and the joined spans are extended with the
            spans added. Once it is not, only spans whose window is written count.

            Parameters
            ----------
            measurements : str
                Name of the measurements.

            Returns
            -------
            ndarray
                ``(spans, 2)`` array of the ``[start, end)`` steps of every joined span.

            Notes
            -----
            The result is cached until the spans listed, the windows read or the status of the run
            change.
        """
        table = self._recorded.get(measurements, _NONE)
        cached = self._segments.get(measurements)
        if cached is not None and cached[1:] == (table.size, table.version):
            return cached[0]
        t0, steps, number = table.columns()
        if self._status == 'running' and cached is not None and cached[2] == table.version and cached[1] < table.size:
            spans, added = cached[0], _spans(t0[cached[1]:], steps[cached[1]:])
            if len(spans) and len(added):
                # The new spans starting before the end of the last one join it.
                joined = int(np.searchsorted(added[:, 0], spans[-1, 1], side='right'))
                if joined:
                    spans = spans.copy()
                    spans[-1, 1] = max(spans[-1, 1], added[:joined, 1].max())
                    added = added[joined:]
            spans = np.concatenate([spans, added]) if len(spans) else added
        else:
            if self._status != 'running':
                keep = np.isin(number, np.fromiter(self._windows.get(measurements, {}), np.int64))
                t0, steps = t0[keep], steps[keep]
            spans = _spans(t0, steps)
        self._segments[measurements] = (spans, table.size, table.version)
        return spans

    def segments(self, measurements: str) -> list[tuple[int, int]]:
        """
            Returns `spans` as a list.

            Parameters
            ----------
            measurements : str
                Name of the measurements.

            Returns
            -------
            list of (int, int)
                ``(start, end)`` of every joined span.
        """
        return [tuple(span) for span in self.spans(measurements).tolist()]

    def segment_at(self, measurements: str, t: int) -> tuple[int, int] | None:
        """
            Returns the joined span holding a step, or else the last one before it.

            Parameters
            ----------
            measurements : str
                Name of the measurements.
            t : int
                Step.

            Returns
            -------
            tuple of (int, int) or None
                ``(start, end)`` of the span, or None when no joined span starts at or before ``t``.
        """
        spans = self.spans(measurements)
        index = int(np.searchsorted(spans[:, 0], t, side='right')) - 1
        return (int(spans[index, 0]), int(spans[index, 1])) if index >= 0 else None

    def span_at(self, measurements: str, t: int, written: bool = False) -> tuple[int, int, int] | None:
        """
            Returns the last span listed that starts at or before a step.

            Parameters
            ----------
            measurements : str
                Name of the measurements.
            t : int
                Step.
            written : bool, default False
                Skip the spans whose window is not written yet.

            Returns
            -------
            tuple of (int, int, int) or None
                ``(t0, steps, window number)`` of the span, or None without one.
        """
        return _at(self.recorded(measurements), t, self._windows.get(measurements, {}) if written else None)

    def group_at(self, measurements: str, t: int) -> tuple[int, int, int] | None:
        """
            Returns the last written group that starts at or before a step.

            Parameters
            ----------
            measurements : str
                Name of the measurements.
            t : int
                Step.

            Returns
            -------
            tuple of (int, int, int) or None
                ``(t0, steps, window number)`` of the group, or None without one.
        """
        return _at(self.groups(measurements), t, self._windows.get(measurements, {}))

    def arrays(self, measurements: str, number: int, keys: tp.Sequence[str]) -> dict[str, np.ndarray] | None:
        """
            Returns arrays of a window.

            Only the arrays not kept from an earlier call are read from the file. A file that cannot
            be read gives the arrays kept before, and is read again on the next call.

            Parameters
            ----------
            measurements : str
                Name of the measurements.
            number : int
                Window number.
            keys : sequence of str
                Names of the arrays.

            Returns
            -------
            dict of str to ndarray or None
                Arrays of ``keys`` that the window holds, or None while the window is not written.

            Notes
            -----
            The arrays read are kept up to ``cached_bytes``, and the least recently used are dropped
            first. The arrays of the current call are never dropped.
        """
        window = self._windows.get(measurements, {}).get(number)
        if window is None:
            return None
        missing = [key for key in keys if (measurements, number, key) not in self._cache]
        if missing:
            try:
                loaded = window.read(missing)
            except Exception:
                loaded = None                                           # not readable yet, or damaged
            if loaded is not None:
                for key in missing:
                    value = loaded.get(key)
                    self._cache[(measurements, number, key)] = value
                    self._cached_bytes += 0 if value is None else value.nbytes
        found = {}
        for key in keys:
            if (measurements, number, key) in self._cache:
                self._cache.move_to_end((measurements, number, key))
                if self._cache[(measurements, number, key)] is not None:
                    found[key] = self._cache[(measurements, number, key)]
        while self._cached_bytes > self._cache_limit and len(self._cache) > len(keys):
            _, value = self._cache.popitem(last=False)
            self._cached_bytes -= 0 if value is None else value.nbytes
        return found

    def rows_at(self, measurements: str, t: int) -> tuple[int, tuple[int, int]] | None:
        """
            Returns the window and the joined span that `rows` reads at a step.

            These are the window of the last written span starting at or before ``t``, and the
            joined span holding that span. Equal results give equal rows.

            Parameters
            ----------
            measurements : str
                Name of the measurements.
            t : int
                Step.

            Returns
            -------
            tuple of (int, (int, int)) or None
                Window number and ``(start, end)`` of the joined span, or None without them.
        """
        span = self.span_at(measurements, t, written=True)
        if span is None:
            return None
        return span[2], self.segment_at(measurements, span[0])

    def rows(self, measurements: str, key: str, t: int) -> tuple[np.ndarray, np.ndarray] | None:
        """
            Returns the steps and rows of a trace or raster around a step, ordered by step.

            The rows are those of the joined span of `rows_at`, read from the window of `rows_at`.
            The joined span holds consecutive recorded steps, such as an episode for measurements
            recorded by episode.

            Parameters
            ----------
            measurements : str
                Name of the measurements.
            key : str
                Key of the trace or raster.
            t : int
                Step.

            Returns
            -------
            tuple of (ndarray, ndarray) or None
                Steps and rows, or None without a written span or when the window lacks ``key``.

            Notes
            -----
            The rows of a window written in order are views of the arrays kept, without a copy.
        """
        found = self.rows_at(measurements, t)
        if found is None:
            return None
        number, (start, end) = found
        arrays = self.arrays(measurements, number, (key, f'{key}#t'))
        if arrays is None or key not in arrays:
            return None
        times, values = arrays[f'{key}#t'], arrays[key]
        if len(times) < 2 or np.all(times[1:] >= times[:-1]):
            # In order, as a window written in order is: a slice, without a copy.
            first, last = np.searchsorted(times, start, 'left'), np.searchsorted(times, end, 'left')
            return times[first:last], values[first:last]
        keep = (times >= start) & (times < end)
        times, values = times[keep], values[keep]
        order = np.argsort(times, kind='stable')
        return times[order], values[order]

    def group_value(self, measurements: str, key: str, t: int) -> tuple[int, np.ndarray] | None:
        """
            Returns a value recorded once per group, for the group at a step.

            The group is the last written group starting at or before ``t``.

            Parameters
            ----------
            measurements : str
                Name of the measurements.
            key : str
                Name of the value, ``<probe key>#<reduction>`` or the key of a snapshot.
            t : int
                Step.

            Returns
            -------
            tuple of (int, ndarray) or None
                First step of the group and the value, or None without such a group or when its
                window lacks the value.
        """
        group = self.group_at(measurements, t)
        if group is None:
            return None
        arrays = self.arrays(measurements, group[2], ('group_t0', key))
        if arrays is None or key not in arrays or 'group_t0' not in arrays:
            return None
        index = np.flatnonzero(arrays['group_t0'] == group[0])
        if not len(index):
            return None
        return group[0], arrays[key][int(index[-1])]

    def raw_at(self, measurements: str, name: str, t: int) -> tuple[int, np.ndarray] | None:
        """
            Returns the last frame of a raw stream at or before a step.

            Searches the windows holding frames of the stream, the latest first. A window that
            cannot be read is skipped for `RETRY_READS` seconds. The search stops after
            `RAW_FAILURES` windows that cannot be read.

            Parameters
            ----------
            measurements : str
                Name of the measurements.
            name : str
                Name of the raw stream.
            t : int
                Step.

            Returns
            -------
            tuple of (int, ndarray) or None
                Step and frame, or None when no frame is found.
        """
        key = f'raw:{name}'
        windows = sorted((w for w in self._windows.get(measurements, {}).values() if name in w.raw and w.t0 <= t), key=lambda w: w.number)
        failures = 0
        for window in reversed(windows):
            failed = self._unreadable.get((measurements, window.number))
            if failed is not None and time.monotonic() - failed < RETRY_READS:
                continue
            arrays = self.arrays(measurements, window.number, (key, f'{key}#t'))
            if not arrays:
                self._unreadable[(measurements, window.number)] = time.monotonic()
                failures += 1
                if failures >= RAW_FAILURES:
                    return None                                         # as when the file system fails: not every window
                continue
            self._unreadable.pop((measurements, window.number), None)
            if key not in arrays:
                continue
            times = arrays[f'{key}#t']
            before = np.flatnonzero(times <= t)
            if len(before):
                index = int(before[np.argmax(times[before])])
                return int(times[index]), arrays[key][index]
        return None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _at(columns: tuple[np.ndarray, np.ndarray, np.ndarray], t: int, files: dict[int, Window] | None) -> tuple[int, int, int] | None:
    """
        Returns ``(t0, steps, window number)`` of the last row starting at or before step ``t``.

        With ``files``, rows whose window is not in ``files`` are skipped. None when no row is left.
    """
    t0, steps, number = columns
    index = int(np.searchsorted(t0, t, side='right')) - 1
    if files is not None:
        while index >= 0 and int(number[index]) not in files:
            index -= 1
    return (int(t0[index]), int(steps[index]), int(number[index])) if index >= 0 else None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _spans(t0: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """
        Joins spans ordered by first step into a ``(spans, 2)`` array of ``[start, end)`` steps.

        Overlapping and consecutive spans are joined.
    """
    if not len(t0):
        return np.zeros((0, 2), np.int64)
    ends = t0 + steps
    reach = np.maximum.accumulate(ends)
    first = np.flatnonzero(np.r_[True, t0[1:] > reach[:-1]])
    return np.stack([t0[first], np.maximum.reduceat(ends, first)], axis=1).astype(np.int64)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _raw(text: str | None) -> tuple[str, ...]:
    """
        Returns the raw streams listed in the ``raw`` column of a window.

        Entries that are not strings are dropped. Text that is not a JSON list gives ``()``.
    """
    try:
        names = json.loads(text) if text else []
    except ValueError:
        return ()
    return tuple(n for n in names if isinstance(n, str)) if isinstance(names, list) else ()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
