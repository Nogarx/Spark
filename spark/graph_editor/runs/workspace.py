#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import os
import re
import json
import time
import hashlib
import pathlib
import datetime
import warnings
import collections

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor

from spark.recording.run import Run, runs as list_runs
from spark.graph_editor.styles.run_viewer import THEME
from spark.graph_editor.runs.data import POINTS, READ_ERRORS

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

STEP, WALL = 'step', 'wall'
"""
    The spaces every series can be drawn in besides the integer tags of its run: steps, and wall
    time in seconds since the first row of the run.
"""

NATURAL = 'natural'
"""
    The space chosen to draw every series in the space it was written per.
"""

BUCKETS = 500
"""
    Buckets of the x axis in which the runs of a group are averaged, in steps and wall time.
"""

SUFFIX = '.exploration.json'
"""
    Ending of the files of explorations.
"""

_DIRECTORY = re.compile(r'(?P<stamp>\d{8}-\d{6})_(?P<name>.+)_[0-9a-f]+')
"""
    Name of the directory of a run created without a run id, ``<yyyymmdd>-<hhmmss>_<name>_<id>``.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def identity(run: Run) -> tuple[str, datetime.datetime | None]:
    """
        Returns the name given to the `Recorder` of a run and the time the run was created.

        Read from ``run.json``, else from the name of its directory, `_DIRECTORY`. The time is the
        local time of the host that created the run, or None when it cannot be read.
    """
    match = _DIRECTORY.fullmatch(run.path.name)
    name = run.info.get('name')
    if not isinstance(name, str) or not name:
        name = match['name'] if match else run.path.name
    try:
        created = datetime.datetime.fromisoformat(str(run.info.get('created')))
    except ValueError:
        created = datetime.datetime.strptime(match['stamp'], '%Y%m%d-%H%M%S') if match else None
    return name, created

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def run_names(runs: tp.Sequence[Run]) -> dict[str, str]:
    """
        Returns the name every run is shown by: the name given to its `Recorder` and the minute it
        was created, as ``'cartpole · 10-01 21:23'``.

        Runs that would share a name are told apart by the second they were created, then by the
        order they were created in, as ``'cartpole · 10-01 21:23:05 (2)'``.

        Parameters
        ----------
        runs : sequence of Run
            The runs, oldest first.

        Returns
        -------
        dict of str to str
            The names, by path of run.
    """
    found = {str(run.path): identity(run) for run in runs}
    shown = lambda form: {path: f'{name} · {created:{form}}' if created else name for path, (name, created) in found.items()}
    names, seconds = shown('%m-%d %H:%M'), shown('%m-%d %H:%M:%S')
    taken = collections.Counter(names.values())
    names = {path: seconds[path] if taken[name] > 1 else name for path, name in names.items()}
    taken, counted = collections.Counter(names.values()), collections.Counter()
    for path, name in list(names.items()):
        if taken[name] > 1:
            counted[name] += 1
            names[path] = f'{name} ({counted[name]})'
    return names

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Project(QObject):
    """
        The runs of a directory, looked for again as new ones appear.

        The experiment of a run is the one given to its `Recorder`, unless set in the viewer with
        `set_experiment`, which writes nothing to the run. Fields describe the runs for searching
        and grouping them: ``experiment``, ``name``, and every key of their parameters.

        Parameters
        ----------
        root : str or path-like
            Directory of the runs.
        parent : QObject, optional
            Parent object.

        Attributes
        ----------
        root : pathlib.Path
            Directory of the runs.
        runs : list of Run
            The runs, oldest first.
        overrides : dict of str to str or None
            Experiments set in the viewer, by path of run. None takes a run out of its experiment.
    """

    changed = Signal()

    def __init__(self, root: str | pathlib.Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.root = pathlib.Path(root).resolve()
        self.runs: list[Run] = []
        self.overrides: dict[str, str | None] = {}
        self._names: dict[str, str] = {}
        self.refresh()

    def refresh(self) -> bool:
        """
            Opens the runs added to the directory since, and reads again those being written.

            Returns whether a run was added. Emits `changed` then. A directory that cannot be listed
            adds none.
        """
        known = {str(run.path) for run in self.runs}
        with warnings.catch_warnings():
            # A run being created or damaged is skipped, as `runs` warns.
            warnings.simplefilter('ignore')
            try:
                found = list_runs(self.root) if self.root.is_dir() else []
            except OSError:
                found = []
        added = [run for run in found if str(run.path) not in known]
        for run in self.runs:
            if run.info.get('status') == 'running':
                try:
                    run.refresh()
                except READ_ERRORS:
                    pass
        if added:
            order = {str(run.path): index for index, run in enumerate(found)}
            self.runs = sorted(self.runs + added, key=lambda run: order.get(str(run.path), len(order)))
            self._names = run_names(self.runs)
            self.changed.emit()
        return bool(added)

    def name(self, run: Run) -> str:
        """
            Returns the name ``run`` is shown by, as `run_names` gives it.
        """
        name = self._names.get(str(run.path))
        return name if name is not None else run_names([run])[str(run.path)]

    def run(self, path: str | pathlib.Path) -> Run | None:
        """
            Returns the run at ``path``, or None when the directory holds none there.
        """
        path = str(pathlib.Path(path).resolve())
        return next((run for run in self.runs if str(run.path.resolve()) == path), None)

    def experiment(self, run: Run) -> str | None:
        """
            Returns the experiment of ``run``: the one set in the viewer, else the recorded one.
        """
        key = str(run.path)
        return self.overrides[key] if key in self.overrides else run.experiment

    def set_experiment(self, run: Run, name: str | None) -> None:
        """
            Sets the experiment of ``run`` in the viewer, or takes it out of any for None.
        """
        self.overrides[str(run.path)] = name.strip() if name and name.strip() else None
        self.changed.emit()

    def state(self) -> dict[str, tp.Any]:
        """
            Returns the experiments set in the viewer, by name of run, as an `Exploration` keeps them.
        """
        return {'experiments': {pathlib.Path(path).name: name for path, name in self.overrides.items()}}

    def restore(self, state: dict[str, tp.Any]) -> None:
        """
            Sets the experiments of ``state`` to the runs of the directory with those names.
        """
        experiments = state.get('experiments') or {}
        overrides = {
            str(run.path): experiments[run.path.name] for run in self.runs
            if run.path.name in experiments and (experiments[run.path.name] is None or isinstance(experiments[run.path.name], str))
        }
        if overrides != self.overrides:
            self.overrides = overrides
            self.changed.emit()

    def fields(self) -> list[str]:
        """
            Returns the fields of the runs: ``experiment``, ``name``, then the keys of their
            parameters in sorted order.
        """
        keys = set()
        for run in self.runs:
            keys.update(run.hparams)
        return ['experiment', 'name', *sorted(keys)]

    def value(self, run: Run, field: str) -> tp.Any:
        """
            Returns the value of a field of ``run``, None when it has none.
        """
        if field == 'experiment':
            return self.experiment(run)
        if field == 'name':
            return run.info.get('name')
        return run.hparams.get(field)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def axis_label(space: str) -> str:
    """
        Returns the name of the horizontal axis of ``space``: ``'step'``, ``'wall time (s)'``, or
        the name of the tag.
    """
    return {STEP: 'step', WALL: 'wall time (s)'}.get(space, space)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Spaces:
    """
        The spaces the series of a run are drawn in, and the mapping of their rows between them.

        A space is ``'step'``, ``'wall'``, or the name of an integer tag of the run, such as
        ``'episode'``. A row at a step holds, in a tag space, the value the tag held at that step,
        and in wall time, the seconds since the first row of the run. The natural space of a series
        is the tag it was written per (`Run.tag_of`), or steps.

        Parameters
        ----------
        run : Run
            The run.
    """

    def __init__(self, run: Run) -> None:
        self.run = run
        self._timelines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._natural: dict[str, str] = {}
        self._wall: tuple[np.ndarray, np.ndarray] | None = None

    def natural(self, key: str) -> str:
        """
            Returns the space ``key`` was written per: the name of its tag, or ``'step'``.
        """
        if key not in self._natural:
            try:
                tag = self.run.tag_of(key)
            except READ_ERRORS:
                tag = None
            self._natural[key] = tag or STEP
        return self._natural[key]

    def tags(self) -> list[str]:
        """
            Returns the names of the tags of the run set to integers, in sorted order.
        """
        try:
            found = self.run.tags()
        except READ_ERRORS:
            return []
        return sorted({row['key'] for row in found if _integral(row['value'])})

    def timeline(self, tag: str) -> tuple[np.ndarray, np.ndarray]:
        """
            Returns the steps at which ``tag`` took a new integer value, and the values, ordered by
            step. A value set again as it was is not a new one.

            Read once; call `forget` for a run still being written.
        """
        if tag not in self._timelines:
            try:
                rows = [row for row in self.run.tags(tag) if _integral(row['value'])]
            except READ_ERRORS:
                rows = []
            steps = np.array([row['t'] for row in rows], np.int64)
            values = np.array([int(row['value']) for row in rows], np.int64)
            new = np.r_[True, values[1:] != values[:-1]] if len(values) else np.zeros(0, bool)
            self._timelines[tag] = (steps[new], values[new])
        return self._timelines[tag]

    def forget(self) -> None:
        """
            Drops what was read, to read it again, as for a run being written.
        """
        self._timelines.clear()
        self._wall = None

    def _wall_timeline(self) -> tuple[np.ndarray, np.ndarray]:
        """
            Returns steps and the wall time of the first scalar row at each, in seconds since the
            first row of the run.
        """
        if self._wall is None:
            try:
                rows = self.run._query('SELECT t, MIN(wall) FROM scalars GROUP BY t ORDER BY t')
            except READ_ERRORS:
                rows = []
            steps = np.array([t for t, _ in rows], np.float64)
            walls = np.array([w for _, w in rows], np.float64)
            self._wall = (steps, walls - walls[0] if len(walls) else walls)
        return self._wall

    def to_space(self, steps: np.ndarray, space: str) -> np.ndarray:
        """
            Returns the position of rows at ``steps`` in ``space``, NaN for rows it holds no value
            at, such as rows before a tag is first set.
        """
        steps = np.asarray(steps, np.float64)
        if space == STEP:
            return steps
        if space == WALL:
            known, walls = self._wall_timeline()
            return np.interp(steps, known, walls) if len(known) else np.full(len(steps), np.nan)
        known, values = self.timeline(space)
        index = np.searchsorted(known, steps, side='right') - 1
        found = np.full(len(steps), np.nan)
        inside = index >= 0
        found[inside] = values[index[inside]]
        return found

    def to_steps(self, x: float, space: str) -> tuple[int, int] | None:
        """
            Returns the steps ``[first, end)`` holding the position ``x`` of ``space``, or None.

            A value of a tag holds from the first step it is set at to the step it next changes;
            the last value, to the last step of the run. A step, or a wall time, holds one step.
        """
        if space == STEP:
            return int(round(x)), int(round(x)) + 1
        if space == WALL:
            known, walls = self._wall_timeline()
            if not len(known):
                return None
            step = int(round(np.interp(x, walls, known)))
            return step, step + 1
        known, values = self.timeline(space)
        where = np.flatnonzero(values == int(round(x)))
        if not len(where):
            return None
        first = int(known[where[0]])
        later = np.flatnonzero((known > first) & (values != values[where[0]]))
        end = int(known[later[0]]) if len(later) else max(int(self.run.step), first + 1)
        return first, end

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _integral(value: tp.Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SeriesStore:
    """
        Scalar series read from runs, by run, name and space, kept until read again.

        A series is read as its envelope past `POINTS` rows. Series of a run being written are read
        again at most every `REREAD` seconds. A reader set with `set_reader` reads the series of a
        run instead, such as those the viewer keeps up to date for the run it shows. In a tag
        space, the rows holding the same value of the tag are averaged into one.
    """

    REREAD = 30.0
    """
        Seconds after which the series of a run being written are read again.
    """

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]] = {}
        self._keys: dict[str, tuple[list[str], float]] = {}
        self._spaces: dict[str, Spaces] = {}
        self._readers: dict[str, tp.Callable[[str], tuple[np.ndarray, np.ndarray]]] = {}

    @staticmethod
    def _path(run: Run) -> str:
        # Runs are known by their resolved paths: the same run opened as given and as listed is one run.
        return str(run.path.resolve())

    def set_reader(self, run: Run, reader: tp.Callable[[str], tuple[np.ndarray, np.ndarray]] | None) -> None:
        """
            Reads the series of ``run`` with ``reader``, called with the name of a series, or as
            by default for None.
        """
        if reader is None:
            self._readers.pop(self._path(run), None)
        else:
            self._readers[self._path(run)] = reader

    def spaces(self, run: Run) -> Spaces:
        """
            Returns the spaces of ``run``.
        """
        key = self._path(run)
        if key not in self._spaces:
            self._spaces[key] = Spaces(run)
        return self._spaces[key]

    def _stale(self, run: Run, read: float) -> bool:
        return run.info.get('status') == 'running' and time.monotonic() - read > self.REREAD

    def keys(self, run: Run) -> list[str]:
        """
            Returns the names of the scalar series of ``run``, or none when they cannot be read.
        """
        cached = self._keys.get(self._path(run))
        if cached is None or self._stale(run, cached[1]):
            if cached is not None:
                self.spaces(run).forget()
            try:
                keys = run.scalar_keys()
            except READ_ERRORS:
                keys = []
            cached = self._keys[self._path(run)] = (keys, time.monotonic())
        return cached[0]

    def rows(self, run: Run, key: str) -> tuple[np.ndarray, np.ndarray]:
        """
            Returns the steps and values of a series of ``run``, empty when it has none.
        """
        reader = self._readers.get(self._path(run))
        if reader is not None:
            return reader(key)
        cached = self._rows.get((self._path(run), key))
        if cached is None or self._stale(run, cached[2]):
            if cached is not None:
                # Read again from a run being written: its tags may have moved on too.
                self.spaces(run).forget()
            try:
                steps, values = run.refresh().scalar(key, points=POINTS)
            except READ_ERRORS:
                steps, values = np.zeros(0, np.int64), np.zeros(0, np.float64)
            cached = self._rows[(self._path(run), key)] = (steps, values, time.monotonic())
        return cached[0], cached[1]

    def series(self, run: Run, key: str, space: str = NATURAL) -> tuple[np.ndarray, np.ndarray]:
        """
            Returns a series of ``run`` in ``space``, its natural space by default, as positions
            and values ordered by position.

            Rows without a position in the space are left out. In a tag space, the rows holding
            the same value are averaged into one.
        """
        steps, values = self.rows(run, key)
        spaces = self.spaces(run)
        space = spaces.natural(key) if space == NATURAL else space
        x = spaces.to_space(steps, space)
        keep = np.isfinite(x)
        x, y = x[keep], np.asarray(values, np.float64)[keep]
        if space in (STEP, WALL) or not len(x):
            return x, y
        positions, inverse = np.unique(x, return_inverse=True)
        with np.errstate(invalid='ignore'):
            finite = np.isfinite(y)
            sums = np.bincount(inverse, weights=np.where(finite, y, 0.0), minlength=len(positions))
            counts = np.bincount(inverse, weights=finite.astype(np.float64), minlength=len(positions))
            return positions, np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    def forget(self, run: Run) -> None:
        """
            Drops what was read of ``run``, read again when next asked for.
        """
        path = self._path(run)
        self._keys.pop(path, None)
        for key in [key for key in self._rows if key[0] == path]:
            del self._rows[key]
        if path in self._spaces:
            self._spaces[path].forget()

    def release(self) -> None:
        """
            Drops every series read.
        """
        self._rows.clear()
        self._keys.clear()
        self._spaces.clear()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def smooth(y: np.ndarray, weight: float) -> np.ndarray:
    """
        Returns ``y`` smoothed by an exponential moving average of weight ``weight``, from 0 (none)
        to below 1, corrected for its start as by wandb. NaN are kept and skipped.
    """
    y = np.asarray(y, np.float64)
    if weight <= 0 or not len(y):
        return y
    smoothed = np.full(len(y), np.nan)
    last, debias = 0.0, 0.0
    for index, value in enumerate(y):
        if not np.isfinite(value):
            continue
        last = weight * last + (1 - weight) * value
        debias = weight * debias + (1 - weight)
        smoothed[index] = last / debias
    return smoothed

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Band(tp.NamedTuple):
    """
        The series of the runs of a group taken together, at positions ``x``: their mean, lowest
        and highest value, and the number of runs with a value there.
    """
    x: np.ndarray
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    count: np.ndarray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def aggregate(series: tp.Sequence[tuple[np.ndarray, np.ndarray]], buckets: int | None = BUCKETS) -> Band:
    """
        Returns the series of several runs taken together.

        With ``buckets`` None, as in a tag space, the runs line up on their positions: every
        position of any run, each run counting where it has a value. Otherwise, the values of each
        run are averaged within each of ``buckets`` equal buckets spanning the runs, positioned at
        their middles. The mean, lowest and highest value at a position are taken over the runs with
        a value there.
    """
    series = [(np.asarray(x, np.float64), np.asarray(y, np.float64)) for x, y in series]
    series = [(x[np.isfinite(y)], y[np.isfinite(y)]) for x, y in series]
    series = [(x, y) for x, y in series if len(x)]
    if not series:
        empty = np.zeros(0)
        return Band(empty, empty, empty, empty, np.zeros(0, np.int64))
    if buckets is None:
        x = np.unique(np.concatenate([x for x, _ in series]))
        stacked = np.full((len(series), len(x)), np.nan)
        for row, (xs, ys) in zip(stacked, series):
            row[np.searchsorted(x, xs)] = ys
    else:
        lo, hi = min(float(x[0]) for x, _ in series), max(float(x[-1]) for x, _ in series)
        edges = np.linspace(lo, hi if hi > lo else lo + 1, buckets + 1)
        x = (edges[:-1] + edges[1:]) / 2
        stacked = np.full((len(series), buckets), np.nan)
        for row, (xs, ys) in zip(stacked, series):
            index = np.clip(np.searchsorted(edges, xs, side='right') - 1, 0, buckets - 1)
            sums, counts = np.bincount(index, ys, buckets), np.bincount(index, None, buckets)
            row[counts > 0] = sums[counts > 0] / counts[counts > 0]
    count = np.isfinite(stacked).sum(axis=0)
    with warnings.catch_warnings():
        # Positions no run has a value at give NaN, and warn of it.
        warnings.simplefilter('ignore', RuntimeWarning)
        mean, low, high = np.nanmean(stacked, axis=0), np.nanmin(stacked, axis=0), np.nanmax(stacked, axis=0)
    keep = count > 0
    return Band(x[keep], mean[keep], low[keep], high[keep], count[keep])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Line(tp.NamedTuple):
    """
        What is drawn for a visible run, or for a group of visible runs.
    """
    label: str
    color: QColor
    runs: list[Run]
    group: bool

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Selection(QObject):
    """
        The runs of a project shown, their colors and their groups.

        A run is visible or not. Grouped by fields of the project, such as ``experiment``, the runs
        sharing their values form a group, drawn as one line: the mean of its visible runs over the
        band from their lowest to their highest value. A run of no group, when a field has no value
        for it, is drawn on its own. Colors are given to runs in the order of the project, and to
        groups in the order they are met. Emits `changed` when what is drawn changes.

        Parameters
        ----------
        project : Project
            The runs.
        parent : QObject, optional
            Parent object.

        Attributes
        ----------
        visible : set of str
            Paths of the runs shown.
        group_by : tuple of str
            Fields grouped by, none for no groups.
        members : bool
            Whether the runs of a group are drawn too, faintly.
        smoothing : float
            Weight of the exponential moving average of every series, from 0 to 0.99.
        space : str
            Space every series is drawn in: `NATURAL`, ``'step'``, ``'wall'`` or a tag.
    """

    changed = Signal()

    NEWEST = 10
    """
        Runs shown by default: every run up to this number, else the newest ones.
    """

    def __init__(self, project: Project, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.visible: set[str] = {str(run.path) for run in project.runs[-self.NEWEST:]}
        self.group_by: tuple[str, ...] = ('experiment',) if any(project.experiment(run) for run in project.runs) else ()
        self.members = False
        self.smoothing = 0.0
        self.space = NATURAL
        self._colors: dict[str, QColor] = {}
        self._known = {str(run.path) for run in project.runs}
        project.changed.connect(self._on_project)

    def _on_project(self) -> None:
        # Runs added to the directory are shown.
        paths = {str(run.path) for run in self.project.runs}
        added, self._known = paths - self._known, paths
        self.visible = {path for path in self.visible if path in paths} | added
        self.changed.emit()

    # Changes.

    def set_visible(self, paths: tp.Iterable[str | pathlib.Path], visible: bool) -> None:
        """
            Shows or hides the runs at ``paths``.
        """
        paths = {str(pathlib.Path(path)) for path in paths}
        before = set(self.visible)
        self.visible = (self.visible | paths) if visible else (self.visible - paths)
        if self.visible != before:
            self.changed.emit()

    def show_only(self, paths: tp.Iterable[str | pathlib.Path]) -> None:
        """
            Shows the runs at ``paths``, and hides the others.
        """
        paths = {str(pathlib.Path(path)) for path in paths}
        if paths != self.visible:
            self.visible = paths
            self.changed.emit()

    def show_newest(self, count: int = NEWEST) -> None:
        """
            Shows the ``count`` newest runs, and hides the others.
        """
        self.show_only(str(run.path) for run in self.project.runs[-count:])

    def set_group_by(self, fields: tp.Sequence[str]) -> None:
        """
            Groups the runs by ``fields``, or not for none.
        """
        if tuple(fields) != self.group_by:
            self.group_by = tuple(fields)
            self.changed.emit()

    def set_color(self, key: str, color: QColor) -> None:
        """
            Sets the color of a run, by path, or of a group, by label.
        """
        self._colors[key] = QColor(color)
        self.changed.emit()

    def set_settings(self, *, members: bool | None = None, smoothing: float | None = None, space: str | None = None) -> None:
        """
            Sets how the lines are drawn; the values not given are kept.
        """
        before = (self.members, self.smoothing, self.space)
        self.members = self.members if members is None else bool(members)
        self.smoothing = self.smoothing if smoothing is None else min(max(float(smoothing), 0.0), 0.99)
        self.space = self.space if space is None else space
        if (self.members, self.smoothing, self.space) != before:
            self.changed.emit()

    def state(self) -> dict[str, tp.Any]:
        """
            Returns what is shown and how, with runs by name, as an `Exploration` keeps it.
        """
        name = lambda path: pathlib.Path(path).name
        return {
            'visible': sorted(name(path) for path in self.visible),
            'known': sorted(name(path) for path in self._known),
            'group_by': list(self.group_by),
            'colors': {key if key.startswith('group:') else f'run:{name(key)}': color.name() for key, color in self._colors.items()},
            'members': self.members,
            'smoothing': self.smoothing,
            'space': self.space,
        }

    def restore(self, state: dict[str, tp.Any]) -> None:
        """
            Shows what ``state`` shows, as it shows it. Runs added to the directory since are shown;
            runs gone are left out.
        """
        paths = {run.path.name: str(run.path) for run in self.project.runs}
        known = set(state.get('known', paths))
        self.visible = {paths[name] for name in state.get('visible', ()) if name in paths}
        self.visible |= {path for name, path in paths.items() if name not in known}
        self.group_by = tuple(field for field in state.get('group_by', ()) if isinstance(field, str))
        self._colors = {}
        for key, value in (state.get('colors') or {}).items():
            if key.startswith('run:'):
                if key[4:] not in paths:
                    continue
                key = paths[key[4:]]
            found = QColor(value)
            if found.isValid():
                self._colors[key] = found
        self.members = bool(state.get('members', self.members))
        self.smoothing = min(max(float(state.get('smoothing', self.smoothing)), 0.0), 0.99)
        self.space = str(state.get('space', self.space))
        self.changed.emit()

    # Reading.

    def color(self, key: str) -> QColor:
        """
            Returns the color of a run, by path, or of a group, by label: the run colours of `THEME`
            in order, repeated past the last.
        """
        if key not in self._colors:
            self._colors[key] = QColor(THEME.run_colors[len(self._colors) % len(THEME.run_colors)])
        return QColor(self._colors[key])

    def label(self, run: Run) -> str:
        """
            Returns the label of a run: its experiment and the name it is shown by (`Project.name`).
        """
        experiment, name = self.project.experiment(run), self.project.name(run)
        return f'{experiment} · {name}' if experiment and 'experiment' not in self.group_by else name

    def groups(self) -> list[tuple[str, list[Run]]]:
        """
            Returns the groups of every run of the project, visible or not, as their labels and
            runs in the order of the project. A run of no group is a group of its own, labelled by
            an empty string.
        """
        if not self.group_by:
            return [('', [run]) for run in self.project.runs]
        found: dict[tuple, list[Run]] = {}
        alone: list[tuple[str, list[Run]]] = []
        for run in self.project.runs:
            values = tuple(self.project.value(run, field) for field in self.group_by)
            if all(value is None for value in values):
                alone.append(('', [run]))
                continue
            found.setdefault(values, []).append(run)
        return [(self._group_label(values), runs) for values, runs in found.items()] + alone

    def _group_label(self, values: tuple) -> str:
        if len(values) == 1:
            return str(values[0])
        return ', '.join(f'{field}={value}' for field, value in zip(self.group_by, values))

    def lines(self) -> list[Line]:
        """
            Returns what is drawn: a line per group with a visible run, and per visible run of no
            group, in the order of the project.
        """
        lines = []
        for label, runs in self.groups():
            shown = [run for run in runs if str(run.path) in self.visible]
            if not shown:
                continue
            if label:
                lines.append(Line(f'{label} · {len(shown)} run' + ('s' if len(shown) > 1 else ''), self.color(f'group:{label}'), shown, True))
            else:
                lines.append(Line(self.label(shown[0]), self.color(str(shown[0].path)), shown, False))
        return lines

    def space_of(self, store: SeriesStore, run: Run, key: str) -> str:
        """
            Returns the space a series of ``run`` is drawn in: its natural one, or the one chosen.
        """
        return store.spaces(run).natural(key) if self.space == NATURAL else self.space

    def draw(self, store: SeriesStore, line: Line, key: str, space: str | None = None) -> tuple[str, Band, list[tuple[np.ndarray, np.ndarray]]]:
        """
            Returns the space of a line for the series ``key``, its band, and the series of each of
            its runs, smoothed.

            A run's band is its series, as its mean, lowest and highest value. A group's runs line
            up in a tag space, and are averaged in `BUCKETS` buckets otherwise. The runs of a line
            are drawn in ``space``, or else in the space of its first run with the series.
        """
        found = [(run, self.space_of(store, run, key)) for run in line.runs if len(store.rows(run, key)[0])]
        if not found:
            empty = np.zeros(0)
            return space or STEP, Band(empty, empty, empty, empty, np.zeros(0, np.int64)), []
        space = space or found[0][1]
        series = []
        for run, _ in found:
            x, y = store.series(run, key, space)
            series.append((x, smooth(y, self.smoothing)))
        if len(series) == 1:
            x, y = series[0]
            return space, Band(x, y, y, y, np.ones(len(x), np.int64)), series
        return space, aggregate(series, None if space not in (STEP, WALL) else BUCKETS), series

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def explorations_path() -> pathlib.Path:
    """
        Returns the folder of the explorations the viewer keeps, next to the model library of the
        editor.
    """
    from spark.graph_editor.models.model_library import default_path
    return default_path().parent / 'explorations'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def within_runs(path: str | pathlib.Path) -> bool:
    """
        Returns whether the file ``path`` is within a run, or within a directory holding runs.
    """
    folder = pathlib.Path(path).expanduser().resolve().parent
    for directory in (folder, *folder.parents):
        if (directory / 'run.json').exists():
            return True
        try:
            if any((child / 'run.json').exists() for child in directory.iterdir() if child.is_dir()):
                return True
        except OSError:
            continue
    return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Exploration:
    """
        The state of the exploration of a directory of runs, kept in a file of its own.

        The viewer keeps one per directory in `explorations_path`, and saves others where asked, but
        never within a directory of runs: what the viewer writes cannot take the place of what a
        recorder wrote. The state is a dictionary of JSON values, written whole.

        Parameters
        ----------
        path : str or path-like
            The file.

        Attributes
        ----------
        path : pathlib.Path
            The file.
    """

    VERSION = 1
    """
        Version of the state written, read back only by the same version.
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        self.path = pathlib.Path(path).expanduser()

    @classmethod
    def of(cls, root: str | pathlib.Path) -> Exploration:
        """
            Returns the exploration the viewer keeps for the directory of runs ``root``.
        """
        root = pathlib.Path(root).resolve()
        digest = hashlib.sha1(str(root).encode()).hexdigest()[:10]
        name = re.sub(r'[^A-Za-z0-9_.-]+', '_', root.name) or 'root'
        return cls(explorations_path() / f'{name}-{digest}{SUFFIX}')

    def read(self) -> dict[str, tp.Any] | None:
        """
            Returns the state written, or None when there is none, of another version, or unreadable.
        """
        try:
            state = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        return state if isinstance(state, dict) and state.get('version') == self.VERSION else None

    def write(self, state: dict[str, tp.Any]) -> None:
        """
            Writes ``state``, in place of what the file held.

            Raises
            ------
            ValueError
                When the file is within a run, or within a directory holding runs.
            OSError
                When the file cannot be written.
        """
        if within_runs(self.path):
            raise ValueError(f'"{self.path}" is within a directory of runs, where explorations are not written.')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        written = self.path.with_name(self.path.name + '.tmp')
        written.write_text(json.dumps({**state, 'version': self.VERSION}, indent=1))
        os.replace(written, self.path)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
