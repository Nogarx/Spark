#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
if tp.TYPE_CHECKING:
    from spark.core.config import SparkConfig
    from spark.nn.controllers.base import Controller

import math
import json
import sqlite3
import pathlib
import warnings
import contextlib
import dataclasses as dc
import numpy as np
from spark.recording.measurements import Measurements, canonical_scalar_key, scalar_key, scalar_key_forms
from spark.recording.records import Record, split_key
from spark.recording.settings import SETTINGS
from spark.recording.utils import integer
from spark.recording import store

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

TABLES = ('scalars', 'keys', 'events', 'tags', 'windows', 'spans', 'groups', 'requests')
"""
    Tables read by `Run.rows`.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True)
class Window:
    """
        One file of a set of measurements.

        A window holds what the measurements recorded over some time: the rows of their traces and
        rasters, one per step kept; their summaries, snapshots and deltas, one row per group of
        steps; and the raw frames kept meanwhile.

        Attributes
        ----------
        measurements : str
            Name of the measurements.
        number : int
            Number of the file within the measurements, from 0.
        t0, t1 : int
            First step covered, and the step after the last one.
        spans : int
            Number of spans in the file, one per call that recorded the measurements.
        groups : int
            Number of groups in the file, one row each.
        file : pathlib.Path
            Path of the ``.npz`` file.
        raw : tuple of str
            Raw streams with frames in the file.

        See Also
        --------
        Run.windows : The windows of a run.
        Run.window : The arrays of one window, by name and number.
    """
    measurements: str
    number: int
    t0: int
    t1: int
    spans: int
    groups: int
    file: pathlib.Path
    raw: tuple[str, ...] = ()

    def read(self, keys: tp.Iterable[str] | None = None) -> dict[str, np.ndarray]:
        """
            Reads the arrays of the file.

            Parameters
            ----------
            keys : iterable of str, optional
                Keys to read. All of them by default. Missing keys are left out.

            Returns
            -------
            dict of str to ndarray
                Arrays by key, as listed by `Run.window`. Rasters are bool arrays of
                ``(steps, units)``.

            Notes
            -----
            Rasters are stored as bits along the unit axis, with their number of units as
            ``<key>#units``. Those entries are not returned.
        """
        with np.load(self.file) as data:
            names = [k for k in (data.files if keys is None else keys) if k in data.files and not k.endswith('#units')]
            arrays = {k: data[k] for k in names}
            for key in names:
                if key.endswith('@raster'):
                    arrays[key] = np.unpackbits(arrays[key], axis=-1, count=int(data[f'{key}#units'])).view(bool)
        return arrays

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Run:
    """
        A run written by a `Recorder`, opened for reading.

        A run can be opened while it is being written. Reads return what was written so far.

        Parameters
        ----------
        path : str or path-like
            Directory of the run.

        Attributes
        ----------
        path : pathlib.Path
            Directory of the run.
        info : dict
            Contents of ``run.json``: identity, environment, status and progress.
        hparams : dict
            Contents of ``hparams.json``.
        measurements : dict of str to Measurements
            The measurements of the run, by name.

        Raises
        ------
        FileNotFoundError
            When ``path`` holds no ``run.json``.
        ValueError
            When the index was written by another version of the tables.

        Notes
        -----
        `info` and the progress of the run are read when it is opened and by `refresh`.

        Each query opens a read-only connection to ``index.sqlite`` and closes it, unless it runs inside
        `reading`. Tables are read in pages of `SETTINGS.page_rows` rows or row ids, one statement per
        page.

        See Also
        --------
        load : Opens a run.
        runs : Opens every run of a directory.
        Recorder : Writes a run.
        Window : One file of a set of measurements.

        Examples
        --------
        >>> run = spark.recording.load('runs/20260923-101500_Brain_1a2b3c')
        >>> t, rate = run.scalar('summary/first_pool.soma:spikes/active_fraction')
        >>> t, potential = run.read('episode')[20].traces['first_pool.soma.potential']
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        if not (self.path / 'run.json').exists():
            raise FileNotFoundError(f'No run at "{self.path}".')
        self.info = store.read_info(self.path)
        self.hparams = json.loads((self.path / 'hparams.json').read_text())
        data = json.loads((self.path / 'recorder.json').read_text())
        self.measurements = {r['name']: Measurements.from_dict(r) for r in data['measurements']}
        self._connection: sqlite3.Connection | None = None
        # Whether the index keeps the tags of the series, read when first needed.
        self._keeps_tags: bool | None = None
        version = self._query('PRAGMA user_version')[0][0]
        if version != store.SCHEMA_VERSION:
            raise ValueError(
                f'"{self.path.name}" has version {version} of the index; this version of Spark reads version {store.SCHEMA_VERSION}.'
            )
        self._progress = self._read_progress()

    @contextlib.contextmanager
    def reading(self) -> tp.Generator[Run, None, None]:
        """
            Reads the run through one connection until the block ends.

            Queries inside the block share one read-only connection. A block nested in another
            reuses its connection. The index is held only while a statement runs.

            Returns
            -------
            context manager
                Yields the run itself.

            Examples
            --------
            >>> with run.reading():
            ...     series = {key: run.scalar(key) for key in run.scalar_keys()}
        """
        if self._connection is not None:
            yield self
            return
        self._connection = store.connect_read_only(self.path)
        try:
            yield self
        finally:
            connection, self._connection = self._connection, None
            connection.close()

    def _read_progress(self) -> dict[str, int]:
        """
            Reads the ``progress`` table of the index.
        """
        return {key: int(value) for key, value in self._query('SELECT key, value FROM progress')}

    def _query(self, sql: str, args: tuple = ()) -> list[tuple]:
        """
            Runs one statement and returns its rows.

            Uses the connection of `reading` when open, and a connection of its own otherwise.
        """
        if self._connection is not None:
            return self._connection.execute(sql, args).fetchall()
        connection = store.connect_read_only(self.path)
        try:
            return connection.execute(sql, args).fetchall()
        finally:
            connection.close()

    def _last(self, table: str) -> int:
        """
            Returns the largest row id of ``table``, or 0 when it is empty.
        """
        return int(self._query(f'SELECT MAX(rowid) FROM {table}')[0][0] or 0)

    def _paged(self, sql: str, args: tuple, after: int, last: int) -> list[tuple]:
        """
            Runs ``sql`` over the row ids from ``after`` to ``last``, `SETTINGS.page_rows` row ids per
            statement.

            The last two parameters of ``sql`` bound the row ids (``rowid > ? AND rowid <= ?``).
        """
        with self.reading():
            return [row for page in self._pages(sql, args, after, last) for row in page]

    def _pages(self, sql: str, args: tuple, after: int, last: int) -> tp.Iterator[list[tuple]]:
        """
            Yields the rows of `_paged` one page at a time.
        """
        size = SETTINGS.page_rows
        for start in range(int(after), int(last), size):
            yield self._query(sql, (*args, start, min(start + size, int(last))))

    def _sorted_rows(self, table: str, columns: str, column: str | None = None, value: tp.Any = None) -> list[tuple]:
        """
            Returns ``columns`` of every row of ``table``, sorted by the first two of them.

            With ``column``, only the rows whose ``column`` holds ``value``.
        """
        sql, args = f'SELECT {columns} FROM {table} WHERE ', ()
        if column is not None:
            sql, args = sql + f'{column} = ? AND ', (value,)
        with self.reading():
            return sorted(self._paged(sql + 'rowid > ? AND rowid <= ?', args, 0, self._last(table)), key=lambda row: row[:2])

    def _row_pages(self, table: str, after: int) -> tp.Iterator[list[tuple]]:
        """
            Yields the rows of `rows` one page at a time, skipping empty pages.
        """
        with self.reading():
            sql = f'SELECT rowid, * FROM {table} WHERE rowid > ? AND rowid <= ? ORDER BY rowid'
            for page in self._pages(sql, (), after, self._last(table)):
                if page:
                    yield page

    def refresh(self) -> Run:
        """
            Reads ``run.json`` and the progress of the run again.

            Returns
            -------
            Run
                The run itself.
        """
        self.info = store.read_info(self.path)
        self._progress = self._read_progress()
        return self

    def rows(self, table: str, after: int = 0) -> list[tuple]:
        """
            Returns the rows of a table of the index with a row id greater than ``after``.

            Parameters
            ----------
            table : str
                Name of the table, one of `TABLES`.
            after : int, default 0
                Row id after which rows are returned.

            Returns
            -------
            list of tuple
                Rows ordered by row id, each led by its row id.

            Raises
            ------
            ValueError
                When ``table`` is not one of `TABLES`.
        """
        if table not in TABLES:
            raise ValueError(f'Unknown table "{table}". Expected one of: {", ".join(TABLES)}.')
        return [row for page in self._row_pages(table, after) for row in page]

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    @property
    def status(self) -> str:
        """
            Status of the run, from ``run.json``.

            One of ``'running'``, ``'finished'``, ``'failed'`` and ``'preempted'``, or ``'unknown'``
            when it is missing. A run marked as running reads as ``'crashed'`` when its last
            heartbeat is older than `SETTINGS.crashed_after` heartbeat periods plus one second, and no
            recorder is known to hold its lock.
        """
        status = self.info.get('status', 'unknown')
        if status != 'running':
            return status
        age, period = store.heartbeat_age(self.info)
        if age <= SETTINGS.crashed_after * period + 1 or store.locked(self.path):
            return 'running'
        return 'crashed'

    @property
    def experiment(self) -> str | None:
        """
            Name of the experiment the run belongs to, as given to its `Recorder`, or None.
        """
        return self.info.get('experiment')

    @property
    def step(self) -> int:
        """
            Number of steps written so far.
        """
        return max(int(self.info.get('step', 0)), self._progress.get('step', 0))

    def config(self) -> SparkConfig:
        """
            Reads the configuration of the model recorded, from ``model.scfg``.

            Returns
            -------
            SparkConfig
                Configuration of the model.
        """
        from spark.core.config import SparkConfig
        return SparkConfig.from_file(str(self.path / 'model.scfg'))

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    def scalar_keys(self) -> list[str]:
        """
            Returns the names of every scalar series, in sorted order.

            The series of summaries and deltas are named ``<measurements>/<address>/<reduction>``,
            also for runs that stored them with the mode of the probe.
        """
        return sorted({canonical_scalar_key(key) for (key,) in self._query('SELECT name FROM keys')})

    def tag_of(self, key: str) -> str | None:
        """
            Returns the tag a scalar series was written per, or None for steps.

            A series logged with `Recorder.log` has the tag given to it. A summary has the tag its
            measurements are grouped by, also in runs written before the tags of the series were
            kept, where a logged series reads as per step.

            Parameters
            ----------
            key : str
                Name of the series.

            Returns
            -------
            str or None
                The name of the tag, or None for steps and for an unknown series.
        """
        stored = self._stored_key(key)
        if stored is None:
            return None
        if self._keeps_tags is None:
            self._keeps_tags = any(row[1] == 'tag' for row in self._query('PRAGMA table_info(keys)'))
        if self._keeps_tags:
            return self._query('SELECT tag FROM keys WHERE name = ?', (stored,))[0][0]
        canonical = canonical_scalar_key(stored)
        measurements = self.measurements.get(canonical.split('/', 1)[0])
        if measurements is None or not isinstance(measurements.group, str):
            return None
        written = {scalar_key(measurements.name, probe.key, reduction) for probe in measurements.probes for reduction in getattr(probe, 'reduce', ())}
        return measurements.group if canonical in written else None

    def _stored_key(self, key: str) -> str | None:
        """
            Returns the name a series is stored under, given in either form, or None.
        """
        for form in scalar_key_forms(key):
            if self._query('SELECT 1 FROM keys WHERE name = ?', (form,)):
                return form
        return None

    def scalar(self, key: str, points: int | None = None, until: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
            Returns the steps and values of one scalar series, ordered by step.

            An unknown ``key`` gives empty arrays.

            Parameters
            ----------
            key : str
                Name of the series.
            points : int, optional
                Approximate maximum number of rows returned. A longer series is reduced to its
                envelope. The envelope holds the rows with the lowest and the highest value in each
                of ``points // 2`` equal bins of steps, and the first NaN of each bin holding one.
            until : int, optional
                Row id past which rows are left out, as `scalar_rows` returns it.

            Returns
            -------
            steps : ndarray of int64
                Step of each row.
            values : ndarray of float64
                Value of each row. A value stored as NULL (NaN) reads as NaN.
        """
        steps, values, _ = self._scalar(key, points, until)
        return steps, values

    def _scalar(self, key: str, points: int | None, until: int | None) -> tuple[np.ndarray, np.ndarray, bool]:
        """
            Returns the arrays of `scalar` and whether the series was reduced to its envelope.
        """
        reduced = False
        with self.reading():
            found = self._query('SELECT id FROM keys WHERE name = ?', (self._stored_key(key) or key,))
            if not found:
                return np.zeros(0, np.int64), np.zeros(0, np.float64), False
            key_id = found[0][0]
            # One bound for every page, so that the pages read one state of the series.
            bound = self._last('scalars') if until is None else int(until)
            count = None
            if points is not None:
                count = self._query(
                    'SELECT COUNT(*) FROM (SELECT 1 FROM scalars WHERE key = ? AND rowid <= ? LIMIT ?)', (key_id, bound, int(points) + 1),
                )[0][0]
            if count is None or count <= points:
                rows = [(t, value) for t, _, value in self._series(key_id, bound)]
            else:
                reduced = True
                rows = self._envelope(key_id, bound, int(points))
        data = np.array(rows, dtype=np.float64).reshape(-1, 2)
        return data[:, 0].astype(np.int64), data[:, 1], reduced

    def _series(self, key_id: int, bound: int) -> list[tuple]:
        """
            Reads every row of a series up to row id ``bound``, as ``(t, rowid, value)``.

            Rows are ordered by step and row id, and read `SETTINGS.page_rows` rows per statement.
        """
        columns, size = 'SELECT t, rowid, value FROM scalars WHERE key = ? AND rowid <= ?', SETTINGS.page_rows
        rows = self._query(f'{columns} ORDER BY t, rowid LIMIT ?', (key_id, bound, size))
        page = rows
        while len(page) == size:
            t, rowid = page[-1][0], page[-1][1]
            # The rest of the step read last, then the steps after it.
            page = self._query(f'{columns} AND t = ? AND rowid > ? ORDER BY rowid LIMIT ?', (key_id, bound, t, rowid, size))
            if len(page) < size:
                page += self._query(f'{columns} AND t > ? ORDER BY t, rowid LIMIT ?', (key_id, bound, t, size - len(page)))
            rows += page
        return rows

    def _envelope(self, key_id: int, bound: int, points: int) -> list[tuple]:
        """
            Computes the envelope of a series up to row id ``bound``, as ``(t, value)`` rows.

            The envelope holds the rows with the lowest and the highest value in each of ``points // 2``
            equal bins of steps, and the first NaN of each bin holding one. The bins are read in ranges,
            three statements per range. There are about 16 ranges, or about one per
            ``4 * SETTINGS.page_rows`` row ids up to ``bound`` when that is more, and at most one per
            bin.
        """
        where = 'key = ? AND rowid <= ?'
        # One aggregate per statement, each read from an end of the index.
        first = self._query(f'SELECT MIN(t) FROM scalars WHERE {where}', (key_id, bound))[0][0]
        last = self._query(f'SELECT MAX(t) FROM scalars WHERE {where}', (key_id, bound))[0][0]
        spans = max(points // 2, 1)
        width = (last - first) // spans + 1
        ranges = min(spans, max(16, math.ceil(bound / (4 * SETTINGS.page_rows))))
        per_range = -(-spans // ranges)
        kept = set()
        for start in range(first, last + 1, per_range * width):
            args = (first, width, key_id, bound, start, start + per_range * width)
            for aggregate in ('MIN(value)', 'MAX(value)'):
                # With one MIN or MAX, the bare column t is read from the row holding it.
                query = f'SELECT (t - ?) / ? AS span, {aggregate}, t FROM scalars WHERE {where} AND t >= ? AND t < ? AND value IS NOT NULL GROUP BY span'
                kept.update((t, value) for _, value, t in self._query(query, args))
            query = f'SELECT (t - ?) / ? AS span, MIN(t) FROM scalars WHERE {where} AND t >= ? AND t < ? AND value IS NULL GROUP BY span'
            kept.update((t, None) for _, t in self._query(query, args))
        return sorted(kept, key=lambda row: (row[0], row[1] is None))

    def scalar_at(self, key: str, t: int) -> float | None:
        """
            Returns the last value of a scalar series at or before step ``t``.

            Parameters
            ----------
            key : str
                Name of the series.
            t : int
                Step.

            Returns
            -------
            float or None
                The value last written at the latest step up to ``t``. NaN for a value stored as
                NULL. None when the series has no value up to ``t``, or does not exist.
        """
        with self.reading():
            rows = self._query(
                'SELECT value FROM scalars WHERE key = (SELECT id FROM keys WHERE name = ?) AND t <= ? ORDER BY t DESC, rowid DESC LIMIT 1',
                (self._stored_key(key) or key, int(t)),
            )
        if not rows:
            return None
        return float('nan') if rows[0][0] is None else float(rows[0][0])

    def scalars(self, prefix: str = '') -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
            Returns every scalar series whose name starts with ``prefix``.

            Parameters
            ----------
            prefix : str, default ''
                Start of the names of the series returned.

            Returns
            -------
            dict of str to tuple of ndarray
                Steps and values of each series, as `scalar` returns them, by name.
        """
        return {key: self.scalar(key) for key in self.scalar_keys() if key.startswith(prefix)}

    def scalar_rows(self, after: int = 0, keys: tp.Collection[str] | None = None) -> tuple[int, dict[str, np.ndarray]]:
        """
            Returns the scalar rows with a row id greater than ``after``, by series.

            Parameters
            ----------
            after : int, default 0
                Row id after which rows are returned.
            keys : collection of str, optional
                Names of the series returned. All of them by default.

            Returns
            -------
            last : int
                Row id of the last row of the table, or ``after`` when the table has no newer row.
            rows : dict of str to ndarray
                ``(rows, 2)`` float64 arrays of steps and values, ordered by step, by name. Series
                with no new row are left out. A value stored as NULL (NaN) reads as NaN.

            Notes
            -----
            Every row after ``after`` is read, with any ``keys``. The rows are filtered by series on
            the host.
        """
        with self.reading():
            last = self._last('scalars')
            if last <= after or (keys is not None and not keys):
                return max(last, after), {}
            names = {i: canonical_scalar_key(name) for i, name in self._query('SELECT id, name FROM keys')}
            if keys is not None:
                keys = {canonical_scalar_key(key) for key in keys}
            wanted = set(names) if keys is None else {i for i, name in names.items() if name in keys}
            grouped: dict[int, list[tuple]] = {}
            # Filtered page by page: only the rows of the series asked for are kept.
            for page in self._pages('SELECT key, t, value FROM scalars WHERE rowid > ? AND rowid <= ? ORDER BY rowid', (), after, last):
                for key, t, value in page:
                    if key in wanted:
                        grouped.setdefault(key, []).append((t, value))
        out = {}
        for key, found in grouped.items():
            data = np.array(found, dtype=np.float64).reshape(-1, 2)
            out[names[key]] = data[np.argsort(data[:, 0], kind='stable')]
        return last, out

    def events(self, kind: str | None = None) -> list[dict[str, tp.Any]]:
        """
            Returns the events of the run, ordered by step.

            Parameters
            ----------
            kind : str, optional
                Kind of the events returned. All kinds by default.

            Returns
            -------
            list of dict
                One dictionary per event, its payload with ``t``, ``kind`` and ``wall`` (wall-clock
                time in seconds since the epoch).
        """
        rows = self._sorted_rows('events', 't, rowid, kind, payload, wall', None if kind is None else 'kind', kind)
        return [{**json.loads(payload), 't': t, 'kind': k, 'wall': wall} for t, _, k, payload, wall in rows]

    def warnings(self) -> list[dict[str, tp.Any]]:
        """
            Returns the warnings the recorder gave while recording, ordered by step.

            Each is a `RecordingWarning` given to the loop, and dropped what it concerns, such as the
            frames of a raw stream no measurements declare.

            Returns
            -------
            list of dict
                The ``warning`` events: ``message``, the step ``t`` and ``wall``, and what the
                warning concerns (``raw``, ``measurements``, ``scalar`` or ``event``).
        """
        return self.events('warning')

    def tags(self, key: str | None = None) -> list[dict[str, tp.Any]]:
        """
            Returns the tags set during the run, ordered by step.

            Parameters
            ----------
            key : str, optional
                Name of the tags returned. All of them by default.

            Returns
            -------
            list of dict
                One dictionary per value set, with ``t``, ``key``, ``value`` (decoded from JSON) and
                ``wall`` (wall-clock time in seconds since the epoch).
        """
        rows = self._sorted_rows('tags', 't, rowid, key, value, wall', None if key is None else 'key', key)
        return [{'t': t, 'key': k, 'value': json.loads(value), 'wall': wall} for t, _, k, value, wall in rows]

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    def windows(self, name: str | None = None) -> list[Window]:
        """
            Returns the window files of the run, ordered by measurements and number.

            Parameters
            ----------
            name : str, optional
                Name of the measurements. All of them by default.

            Returns
            -------
            list of Window
                One entry per window file written.
        """
        columns = 'measurements, window, t0, t1, spans, groups, file, raw'
        rows = self._sorted_rows('windows', columns, None if name is None else 'measurements', name)
        return [Window(r, n, t0, t1, c, g, self.path / f, _raw(m)) for r, n, t0, t1, c, g, f, m in rows]

    def window(self, name: str, number: int) -> dict[str, np.ndarray]:
        """
            Reads the arrays of one window file.

            The keys are:

            * ``span_t0`` and ``span_steps``, the first step and the number of steps of each span.
            * ``<probe key>`` and ``<probe key>#t``, the rows of a trace or raster and their steps.
              Rasters are bool arrays of ``(steps, units)``.
            * ``group_t0`` and ``group_steps``, the first step recorded of each group and the steps
              it covers, when the window holds groups.
            * ``<probe key>#<reduction>`` for summaries and deltas, and ``<probe key>`` for
              snapshots, one row per group.
            * ``raw:<name>`` and ``raw:<name>#t``, the frames of a raw stream and their steps.

            A group whose steps were not all recorded covers fewer steps than the group of the
            measurements.

            Parameters
            ----------
            name : str
                Name of the measurements.
            number : int
                Number of the window within the measurements.

            Returns
            -------
            dict of str to ndarray
                Arrays by key.

            Raises
            ------
            ValueError
                When the measurements ``name`` have no window ``number``.
        """
        rows = self._query('SELECT file FROM windows WHERE measurements = ? AND window = ?', (name, int(number)))
        if not rows:
            raise ValueError(f'No window {number} in measurements "{name}".')
        return Window(name, int(number), 0, 0, 0, 0, self.path / rows[0][0]).read()

    def timeline(self, name: str) -> dict[str, np.ndarray]:
        """
            Reads every window of a set of measurements, joined along the first axis.

            The rows of every record follow each other, on the steps of the run. `read` gives them
            one record at a time.

            Parameters
            ----------
            name : str
                Name of the measurements.

            Returns
            -------
            dict of str to ndarray
                Arrays by key, with the keys of `window`. Empty when the measurements have no
                window.

            Raises
            ------
            ValueError
                When the windows hold different probes, as after a resume with other measurements.
        """
        loaded = [window.read() for window in self.windows(name)]
        # A window may hold rows of traces without groups, or groups without rows: each kind is compared apart.
        rows = lambda key: split_key(key)[1] in ('trace', 'raster')
        grouped = lambda key: split_key(key)[1] in ('summary', 'snapshot', 'delta')
        kinds = (
            {frozenset(filter(rows, arrays)) for arrays in loaded if len(arrays['span_t0'])},
            {frozenset(filter(grouped, arrays)) for arrays in loaded if 'group_t0' in arrays},
        )
        if any(len(kind) > 1 for kind in kinds):
            raise ValueError(f'The windows of "{name}" hold different probes. Read them one by one with `window`.')
        keys = sorted({key for arrays in loaded for key in arrays})
        return {key: np.concatenate([arrays[key] for arrays in loaded if key in arrays], axis=0) for key in keys}

    def read(self, name: str) -> dict[tp.Any, Record]:
        """
            Reads what a set of measurements recorded, one record per group.

            Measurements grouped by a tag give one record per value of the tag, such as one per
            episode, keyed by the value. Measurements grouped by steps give one record per group,
            keyed by its number. Measurements without a group give one record per stretch of steps
            recorded one after the other, keyed by its first step. Only what was recorded has a
            record.

            Parameters
            ----------
            name : str
                Name of the measurements.

            Returns
            -------
            dict of object to Record
                Records by key, ordered by step. Empty when the measurements have no window.

            Raises
            ------
            ValueError
                When the windows hold different probes, as after a resume with other measurements.

            Notes
            -----
            A tag that takes a value again after another one starts another record. Its key is
            ``(value, n)``, for the ``n``-th time the value comes back. The steps before the first
            value of the tag are in the record keyed None.

            See Also
            --------
            Record : What a record holds.
            timeline : The rows of every record, on the steps of the run.

            Examples
            --------
            >>> episodes = run.read('episode')
            >>> t, potential = episodes[20].traces['first_pool.soma.potential']
            >>> run.read('summary')[20].summaries['first_pool.soma:spikes'].active_fraction
        """
        flat = self.timeline(name)
        if not flat or not len(flat.get('span_t0', ())):
            return {}
        group = self.measurements[name].group
        spans = np.stack([flat['span_t0'], flat['span_t0'] + flat['span_steps']], axis=1).astype(np.int64)
        starts, ends, keys = self._bounds(group, spans)

        def locate(steps: np.ndarray) -> np.ndarray:
            return np.searchsorted(starts, np.asarray(steps, np.int64), side='right') - 1

        # The steps each record covers, from the calls that recorded it.
        first, stop = np.full(len(starts), np.iinfo(np.int64).max), np.full(len(starts), -1, np.int64)
        for a, b in spans:
            index = int(locate(a))
            while index < len(starts) and starts[index] < b:
                first[index] = min(first[index], max(a, starts[index]))
                stop[index] = max(stop[index], min(b, ends[index]))
                index += 1
        arrays: dict[int, dict[str, np.ndarray]] = {int(i): {} for i in np.flatnonzero(stop >= 0)}
        rows = lambda key: split_key(key)[1] in ('trace', 'raster', 'raw')
        for key, values in flat.items():
            if key.endswith('#t') or not rows(key) or f'{key}#t' not in flat:
                continue
            t = flat[f'{key}#t']
            index = locate(t)
            order = np.argsort(index, kind='stable')
            index, t, values = index[order], t[order], values[order]
            cuts = np.flatnonzero(np.diff(index)) + 1
            for part, part_t, part_values in zip(np.split(index, cuts), np.split(t, cuts), np.split(values, cuts)):
                if len(part) and int(part[0]) in arrays:
                    arrays[int(part[0])][key] = part_values
                    arrays[int(part[0])][f'{key}#t'] = part_t - starts[part[0]]
        if 'group_t0' in flat:
            index = locate(flat['group_t0'])
            grouped = [key for key in flat if split_key(key)[1] in ('summary', 'snapshot', 'delta')]
            for record in np.unique(index):
                if int(record) not in arrays:
                    continue
                at = np.flatnonzero(index == record)
                for key in grouped:
                    arrays[int(record)][key] = flat[key][at[0]] if len(at) == 1 else flat[key][at]
        return {
            keys[i]: Record.from_arrays(keys[i], int(starts[i]), np.arange(first[i] - starts[i], stop[i] - starts[i]), values)
            for i, values in sorted(arrays.items())
        }

    def _bounds(self, group: str | int | None, spans: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tp.Any]]:
        """
            Returns the first step, the step after the last one and the key of every record.

            By the changes of the tag ``group``, by the groups of ``group`` steps, or by the stretches
            of ``spans`` that follow each other. Sorted by step.
        """
        if isinstance(group, str):
            changes: list[tuple[int, tp.Any]] = []
            for tag in self.tags(group):
                if not changes or tag['value'] != changes[-1][1]:
                    changes.append((int(tag['t']), tag['value']))
            if not changes or changes[0][0] > 0:
                changes.insert(0, (0, None))
            seen: dict[str, int] = {}
            keys = []
            for _, value in changes:
                name = json.dumps(value, sort_keys=True, default=str)
                keys.append(value if name not in seen else (value, seen[name]))
                seen[name] = seen.get(name, 0) + 1
            starts = np.array([t for t, _ in changes], np.int64)
            return starts, np.append(starts[1:], np.iinfo(np.int64).max), keys
        if isinstance(group, int):
            numbers = sorted({n for a, b in spans for n in range(int(a) // group, (int(b) - 1) // group + 1)})
            starts = np.array(numbers, np.int64) * group
            return starts, starts + group, numbers
        stretches: list[list[int]] = []
        for a, b in sorted(spans.tolist()):
            if stretches and a <= stretches[-1][1]:
                stretches[-1][1] = max(stretches[-1][1], b)
            else:
                stretches.append([a, b])
        starts = np.array([a for a, _ in stretches], np.int64)
        return starts, np.array([b for _, b in stretches], np.int64), [int(a) for a, _ in stretches]

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    def record(self, name: str, steps: int = 1) -> str:
        """
            Asks the recorder writing this run to record a set of measurements, as
            `Recorder.record`.

            The request is written as a file of ``requests/``. It can be written from any process,
            on any host, with write access to the run directory. The recorder reads requests every
            half second and applies them before its next call.

            Parameters
            ----------
            name : str
                Name of the measurements.
            steps : int, default 1
                Steps to record them for.

            Returns
            -------
            str
                Id of the request, as listed by `requests`.

            Raises
            ------
            ValueError
                When the run is not being written, has no measurements ``name``, or ``steps`` is not
                a positive integer.
        """
        if name not in self.measurements:
            raise ValueError(f'No measurements "{name}". Expected one of: {", ".join(self.measurements)}.')
        steps = integer(steps, 'steps', lowest=1, highest=store.MAX_STEPS)
        if self.refresh().status != 'running':
            raise ValueError(f'"{self.path.name}" is not being written.')
        return store.write_request(self.path, 'record', {'measurements': name, 'steps': int(steps)})

    def requests(self) -> list[dict[str, tp.Any]]:
        """
            Returns the requests made with `record`, oldest first.

            Returns
            -------
            list of dict
                One dictionary per request, its payload with ``id``, ``kind``, ``status``, ``wall``
                and ``handled``. ``status`` is one of:

                * ``'pending'`` until the recorder reads the request.
                * ``'received'`` or ``'rejected'`` once the recorder reads it.
                * ``'applied'`` once the measurements are recorded.
                * ``'expired'`` for a request the recorder can no longer apply, left when it closes
                  or when the run is resumed.
        """
        # Files first: a request the recorder lists and removes meanwhile is then in the index.
        files = store.pending_requests(self.path)
        rows = self._query('SELECT id, kind, payload, wall, status, handled FROM requests')
        found = {i: {**store.from_json(p, dict, {}), 'id': i, 'kind': k, 'status': st, 'wall': w, 'handled': h} for i, k, p, w, st, h in rows}
        for file in files:
            if file.stem not in found:
                try:
                    request = store.read_request(file)
                except OSError:
                    request = store.blank_request(file)
                found[file.stem] = {
                    **request['payload'], 'id': request['id'], 'kind': request['kind'], 'status': 'pending',
                    'wall': request['wall'], 'handled': None,
                }
        return [found[i] for i in sorted(found)]

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    def checkpoints(self) -> list[int]:
        """
            Returns the steps of the checkpoints written completely, in increasing order.

            A checkpoint is written aside and moved in place once complete, so a file
            ``checkpoints/<step>.spark`` is a complete checkpoint.
        """
        return store.checkpoint_steps(self.path)

    def restore(self, step: int | None = None) -> Controller:
        """
            Returns the model saved by a checkpoint of the run, with `Controller.from_checkpoint`.

            Parameters
            ----------
            step : int, optional
                Step of the checkpoint. The last one by default.

            Returns
            -------
            Controller
                The model, built from the configuration saved with the checkpoint and holding its
                state.

            Raises
            ------
            FileNotFoundError
                When the run has no checkpoint, or none at ``step``.
        """
        from spark.nn.controllers.base import Controller
        steps = self.checkpoints()
        if not steps:
            raise FileNotFoundError(f'No checkpoint in "{self.path}".')
        step = steps[-1] if step is None else int(step)
        if step not in steps:
            raise FileNotFoundError(f'No checkpoint at step {step}. Checkpoints: {steps}.')
        return Controller.from_checkpoint(store.checkpoint_file(self.path, step), verbose=False)

    def __repr__(self) -> str:
        count = len(self.warnings())
        warned = f', {count} warning{"s" if count > 1 else ""}' if count else ''
        return f'Run("{self.path.name}", {self.status}, step {self.step}, measurements {list(self.measurements)}{warned})'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _raw(text: str | None) -> tuple[str, ...]:
    """
        Returns the raw streams listed in the ``raw`` column of a window.

        Empty when the column is not a JSON list. Entries other than strings are left out.
    """
    return tuple(n for n in store.from_json(text, list, []) if isinstance(n, str))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def load(path: str | pathlib.Path) -> Run:
    """
        Opens a run written by a `Recorder`.

        Parameters
        ----------
        path : str or path-like
            Directory of the run.

        Returns
        -------
        Run
            The run, opened for reading.

        Raises
        ------
        FileNotFoundError
            When ``path`` holds no ``run.json``.
        ValueError
            When the index was written by another version of the tables.

        Warns
        -----
        RecordingWarning
            When the recorder gave warnings while recording, with the first of them. `Run.warnings`
            lists them all.

        See Also
        --------
        runs : Opens every run of a directory.
        Run : A run written by a `Recorder`, opened for reading.
    """
    from spark.recording.recorder import RecordingWarning
    run = Run(path)
    given = run.warnings()
    if given:
        more = f' And {len(given) - 1} more, listed by `Run.warnings`.' if len(given) > 1 else ''
        warnings.warn(f'The run "{run.path.name}" was recorded with warnings. {given[0]["message"]}{more}', RecordingWarning, stacklevel=2)
    return run

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _created(run: Run) -> float:
    """
        Returns the time a run was created, as a POSIX timestamp.

        Times written on hosts of other time zones compare correctly. Minus infinity when
        ``created`` cannot be read.
    """
    try:
        return store.timestamp(str(run.info.get('created')))
    except (TypeError, ValueError, OverflowError):
        return float('-inf')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def runs(root: str | pathlib.Path = 'runs', experiment: str | None = None) -> list[Run]:
    """
        Opens every run of a directory, or those of one experiment.

        Directories without ``run.json``, and hidden ones such as the runs being created, are
        skipped. Directories that cannot be read as a run are skipped with a warning.

        Parameters
        ----------
        root : str or path-like, default 'runs'
            Directory of the runs.
        experiment : str, optional
            Name of an experiment, as given to `Recorder`. Only its runs are opened.

        Returns
        -------
        list of Run
            The runs, oldest first by the time they were created, then by name. Empty when ``root``
            does not exist.

        See Also
        --------
        load : Opens one run.
        Run : A run written by a `Recorder`, opened for reading.
    """
    root = pathlib.Path(root)
    if not root.exists():
        return []
    found = []
    for path in sorted(root.iterdir()):
        if path.name.startswith('.') or not (path / 'run.json').exists():
            continue                                                    # runs being created are hidden
        try:
            run = Run(path)
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
            warnings.warn(f'"{path}" is not a readable run: {type(error).__name__}: {error}')
            continue
        if experiment is None or run.experiment == experiment:
            found.append(run)
    return sorted(found, key=lambda run: (_created(run), run.path.name))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
