#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import numpy as np

from spark.recording.probe import ProbeMode

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

KINDS = ('traces', 'rasters', 'raw', 'summaries', 'deltas', 'snapshots')
"""
    Data types holded by a `Record`.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def split_key(name: str) -> tuple[str, str | None, str | None]:
    """
        Splits the name of an array of a window file into its address, mode and part.

        ``'<address>@<mode>#<part>'`` gives ``(address, mode, part)``, the part being ``'t'`` for
        the steps of rows, or the name of a reduction. ``'raw:<name>#t'`` gives
        ``(name, 'raw', 't')``. A name without a part gives a part of None, and the other names,
        such as ``'span_t0'``, give ``(name, None, None)``.
    """
    if name.startswith('raw:'):
        stream = name[len('raw:'):]
        return (stream[:-len('#t')], 'raw', 't') if stream.endswith('#t') else (stream, 'raw', None)
    base, hashed, part = name.partition('#')
    address, at, mode = base.rpartition('@')
    if not at or mode not in ProbeMode:
        return name, None, None
    return address, mode, part if hashed else None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Rows(tp.NamedTuple):
    """
        The rows of a trace, a raster or a raw stream within a record, and their steps.

        Unpacks as ``t, values``.

        Attributes
        ----------
        t : ndarray
            Step of each row, counted from the start of the record.
        values : ndarray
            One row per step of ``t``. Rasters hold one bool per unit.
    """
    t: np.ndarray
    values: np.ndarray

    def __repr__(self) -> str:
        return f'Rows({len(self.t)} rows, {np.shape(self.values)[1:]} {np.asarray(self.values).dtype})'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Reductions(tp.Mapping[str, tp.Any]):
    """
        The reductions of a summary or a delta over one group, by name.

        Read as attributes (``summary.active_fraction``) or as keys (``summary['active_fraction']``).
    """

    def __init__(self, values: dict[str, tp.Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, name: str) -> tp.Any:
        return self._values[name]

    def __getattr__(self, name: str) -> tp.Any:
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(f'No reduction "{name}". Recorded: {", ".join(self._values) or "none"}.') from None

    def __iter__(self) -> tp.Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __dir__(self) -> list[str]:
        return [*super().__dir__(), *self._values]

    def keys(self) -> tp.KeysView[str]:
        return self._values.keys()

    def values(self) -> tp.ValuesView[tp.Any]:
        return self._values.values()

    def items(self) -> tp.ItemsView[str, tp.Any]:
        return self._values.items()

    def __repr__(self) -> str:
        return f'Reductions({", ".join(f"{name}={_describe(value)}" for name, value in self._values.items())})'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _describe(value: tp.Any) -> str:
    """
        Returns a recorded value as a short text, used for `Reductions` and `Record` to print it.
    """
    value = np.asarray(value)
    return f'{value.item():.6g}' if value.ndim == 0 and value.dtype.kind in 'fiub' else f'{value.shape} {value.dtype}'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Record:
    """
        Object holding a collection of measurements recorded over one group of steps (`Measurements.group`).

        The data is held by type, addressed by the type of the probe:

        * `traces`, `rasters` and `raw` (by stream name): `Rows`, the rows and their steps.
        * `summaries` and `deltas`: `Reductions`, the reductions of the group by name.
        * `snapshots`: the value at the end of the group, an array.

        Attributes
        ----------
        key : object
            For a group by tag, the value of the tag: ``(value, n)`` the ``n``-th time the value
            comes back, None for the steps before the tag is first set. For a group of ``n`` steps,
            its number ``g``, of the steps ``[g * n, (g + 1) * n)``. Without a group, the first step
            of the stretch.
        t0 : int
            Step of the run at which the group starts, or the stretch.
        t : ndarray
            Steps recorded, counted from `t0`. A group recorded from its middle starts past 0.
        traces, rasters, raw : dict of str to Rows
        summaries, deltas : dict of str to Reductions
        snapshots : dict of str to ndarray

        Notes
        -----
        Printed, or shown by a notebook, a record lists what it holds.

        See Also
        --------
        Run.read : The records of a set of measurements.

        Examples
        --------
        >>> episode = run.read('episode')[20]
        >>> t, potential = episode.traces['first_pool.soma.potential']
        >>> episode.summaries['first_pool.soma:spikes'].active_fraction
    """

    def __init__(
            self,
            key: tp.Any,
            t0: int,
            t: np.ndarray,
            *,
            traces: dict[str, Rows] | None = None,
            rasters: dict[str, Rows] | None = None,
            raw: dict[str, Rows] | None = None,
            summaries: dict[str, Reductions] | None = None,
            deltas: dict[str, Reductions] | None = None,
            snapshots: dict[str, np.ndarray] | None = None,
        ) -> None:
        self.key = key
        self.t0 = int(t0)
        self.t = np.asarray(t)
        self.traces = traces or {}
        self.rasters = rasters or {}
        self.raw = raw or {}
        self.summaries = summaries or {}
        self.deltas = deltas or {}
        self.snapshots = snapshots or {}

    @classmethod
    def from_arrays(cls, key: tp.Any, t0: int, t: np.ndarray, arrays: dict[str, np.ndarray]) -> Record:
        """
            Builds a record from arrays named as in a window file.

            ``<address>@trace`` and ``<address>@raster`` with their steps in ``<...>#t``,
            ``raw:<name>`` with ``raw:<name>#t``, ``<address>@summary#<reduction>``,
            ``<address>@delta#<reduction>`` and ``<address>@snapshot``. The steps are counted from
            ``t0``.
        """
        kinds: dict[str, dict] = {kind: {} for kind in KINDS}
        reductions: dict[str, dict[str, dict]] = {'summary': {}, 'delta': {}}
        for name, value in arrays.items():
            address, mode, part = split_key(name)
            if part == 't':
                continue
            if mode in ('trace', 'raster', 'raw'):
                kinds['raw' if mode == 'raw' else f'{mode}s'][address] = Rows(arrays[f'{name}#t'], value)
            elif mode == 'snapshot':
                kinds['snapshots'][address] = value
            elif mode in reductions:
                reductions[mode].setdefault(address, {})[part] = value
        kinds['summaries'] = {address: Reductions(values) for address, values in reductions['summary'].items()}
        kinds['deltas'] = {address: Reductions(values) for address, values in reductions['delta'].items()}
        return cls(key, t0, t, **kinds)

    @property
    def steps(self) -> int:
        """
            Number of steps recorded.
        """
        return len(self.t)

    def __repr__(self) -> str:
        held = ', '.join(f'{len(getattr(self, kind))} {kind}' for kind in KINDS if getattr(self, kind))
        return f'Record({self.key!r}, t0={self.t0}, steps={self.steps}{", " + held if held else ""})'

    def __str__(self) -> str:
        first, stop = (int(self.t[0]), int(self.t[-1]) + 1) if self.steps else (0, 0)
        lines = [f'Record {self.key!r}: steps {self.t0 + first} to {self.t0 + stop} of the run ({self.steps} steps)']
        for kind in KINDS:
            entries = getattr(self, kind)
            if not entries:
                continue
            width = max(len(name) for name in entries)
            lines.append(f'  {kind}')
            for name, entry in entries.items():
                if isinstance(entry, Rows):
                    text = f'{np.shape(entry.values)} {np.asarray(entry.values).dtype}'
                elif isinstance(entry, Reductions):
                    text = ', '.join(entry)
                else:
                    text = _describe(entry)
                lines.append(f'    {name:<{width}}  {text}')
        return '\n'.join(lines)

    def _repr_pretty_(self, printer: tp.Any, cycle: bool) -> None:
        printer.text(repr(self) if cycle else str(self))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
