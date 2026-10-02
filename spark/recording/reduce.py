#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
if tp.TYPE_CHECKING:
    from spark.nn.controllers.base import Controller

import math
import functools
import numpy as np
import jax
import jax.numpy as jnp
from spark.recording.probe import (
    Probe, SummaryProbe, TraceProbe, RasterProbe, SnapshotProbe, DeltaProbe, 
    GROUPED_PROBES, BOUNDARY_PROBES, get_recorded_array, resolve,
    check_units,
)
from spark.recording.settings import SETTINGS

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

Layout = tuple[tuple[tp.Any, np.dtype, tuple[int, ...], int, int], ...]
"""
    Layout of a byte buffer. One ``(key, dtype, shape, offset, size)`` entry per value, with
    ``offset`` and ``size`` in bytes.
"""

HISTOGRAM_LOW_BITS = 30
"""
    Bits of the low word of the histogram counts kept over a call. The counts are two int32 words,
    ``high * 2 ** 30 + low``, combined as int64 on the host.
"""

_SIGNIFICAND = {jnp.dtype(jnp.float16): (11, -13), jnp.dtype(jnp.bfloat16): (8, -125)}
"""
    Bits of the significand, and lowest exponent of a normal value, of the narrow floating dtypes, as
    `_round_to` rounds to them.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _group_steps(probe: Probe) -> int | None:
    """
        Returns the steps of each group of ``probe`` when it is grouped by a number of steps, or None.
    """
    return probe.group if isinstance(probe, GROUPED_PROBES) and isinstance(probe.group, int) else None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def moduli(probes: tuple[Probe, ...]) -> tuple[int, ...]:
    """
        Returns the group sizes and strides of ``probes``, in steps.

        `Start` holds the first step of a call modulo each of them. `recorded_scan` aligns groups and
        strides on the steps of the run with these phases.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of a call.

        Returns
        -------
        tuple of int
            The distinct group sizes of the probes grouped by steps, and the strides above 1 of
            traces and rasters, sorted.
    """
    found = set()
    for probe in probes:
        if _group_steps(probe):
            found.add(probe.group)
        elif isinstance(probe, (TraceProbe, RasterProbe)) and probe.stride > 1:
            found.add(probe.stride)
    return tuple(sorted(found))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
class Start:
    """
        Position of a recorded call on the steps of the run.

        A pytree whose leaf is ``phases`` and whose static part is ``split``. `start_of` builds it.

        Parameters
        ----------
        phases : array
            First step of the call modulo each of `moduli`, as int32.
        split : bool, default True
            Whether the call may cross the end of a group of steps.

        Notes
        -----
        Calls that differ only in ``phases`` share one compiled program. Each value of ``split``
        compiles to its own program. With ``split`` False, a summary keeps one set of statistics,
        and snapshots and deltas keep no values at the ends of groups within the call.
    """

    def __init__(self, phases: jax.Array | np.ndarray, split: bool = True) -> None:
        self.phases = phases
        self.split = bool(split)

    def tree_flatten(self) -> tuple[tuple, bool]:
        return (self.phases,), self.split

    @classmethod
    def tree_unflatten(cls, split: bool, children: tuple) -> Start:
        return cls(children[0], split)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def start_of(probes: tuple[Probe, ...], step: int, steps: int | None = None) -> Start:
    """
        Returns the position of a recorded call in the groups and strides of ``probes``.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the call.
        step : int
            Step of the run at which the call starts.
        steps : int, optional
            Steps of the call. Without it, the call may cross the end of a group.

        Returns
        -------
        Start
            Phases of ``step`` modulo each of `moduli`, and whether the call crosses the end of a
            group of steps.

        Notes
        -----
        The phases are computed on the host from ``step`` as a Python integer, exact for any step.
    """
    step = int(step)
    phases = np.asarray([step % m for m in moduli(probes)], dtype=np.int32)
    split = steps is None or any(step % k + int(steps) > k for k in map(_group_steps, probes) if k)
    return Start(phases, split)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def may_split(probes: tuple[Probe, ...], step: int, steps: int) -> bool:
    """
        Returns whether calls of ``steps`` steps may cross the end of a group of ``probes``.

        The calls considered start at the steps ``step + k * steps``, for every ``k >= 0``. Calls
        whose length divides every group never cross one when ``step`` is a multiple of that length.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the calls.
        step : int
            Step of the run at which the first call starts.
        steps : int
            Steps of every call.

        Returns
        -------
        bool
            Whether one of those calls crosses the end of a group of steps.
    """
    for k in filter(None, map(_group_steps, probes)):
        common = math.gcd(int(steps), k)
        if int(steps) + int(step) % common > common:
            return True
    return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def warmup_starts(probes: tuple[Probe, ...], step: int, steps: int) -> tuple[Start, ...]:
    """
        Returns the starts for which calls of ``steps`` steps from ``step`` compile apart.

        A call that may cross the end of a group compiles apart from one that does not
        (`Start.split`). The second is given only when such calls may cross one (`may_split`).

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the calls.
        step : int
            Step of the run at which the first call starts.
        steps : int
            Steps of every call.

        Returns
        -------
        tuple of Start
            The starts to compile the calls with. Their phases are those of step 0.
    """
    phases = start_of(probes, 0).phases
    return tuple(Start(phases, split) for split in ((False, True) if may_split(probes, step, steps) else (False,)))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _phases(start: Start | jax.Array | None) -> jax.Array | None:
    return start.phases if isinstance(start, Start) else start

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _split(start: Start | jax.Array | None) -> bool:
    """
        Returns whether a call may cross the end of a group, as ``start`` gives it.

        True when ``start`` is not a `Start`.
    """
    return start.split if isinstance(start, Start) else True

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _phase(probes: tuple[Probe, ...], start: Start | jax.Array | None, modulus: int) -> jax.Array | int:
    """
        Returns the phase of ``modulus`` in ``start``, or 0 without ``start``.
    """
    phases = _phases(start)
    if phases is None:
        return 0
    return phases[moduli(probes).index(modulus)]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def slots(probe: Probe, steps: int) -> int:
    """
        Returns the number of groups a call can touch, for a summary grouped by steps.

        Parameters
        ----------
        probe : Probe
            Probe of the call.
        steps : int
            Steps of the call.

        Returns
        -------
        int
            Largest number of groups a call of ``steps`` steps holds steps of, over all first steps.
            1 for other probes.
    """
    if not isinstance(probe, SummaryProbe) or not isinstance(probe.group, int):
        return 1
    return (int(steps) + probe.group - 2) // probe.group + 1

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def ends(probe: Probe, steps: int) -> int:
    """
        Returns the number of group ends within a call, for a snapshot or delta grouped by steps.

        Parameters
        ----------
        probe : Probe
            Probe of the call.
        steps : int
            Steps of the call.

        Returns
        -------
        int
            Largest number of ends of groups at the steps of a call of ``steps`` steps, over all
            first steps. 0 for other probes.
    """
    if not isinstance(probe, BOUNDARY_PROBES) or not isinstance(probe.group, int):
        return 0
    return -(-int(steps) // probe.group)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _contiguous(units: tuple[int, ...]) -> bool:
    return units == tuple(range(units[0], units[0] + len(units)))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _select(x: jax.Array, units: tuple[int, ...] | None, address: str, lead: int = 0) -> jax.Array:
    """
        Selects the entries ``units`` of ``x``, flattened after its first ``lead`` axes.

        Returns ``x`` unchanged when ``units`` is None. Raises ValueError for an index past the
        entries.
    """
    if units is None:
        return x
    flat = x.reshape(*x.shape[:lead], -1)
    check_units(flat.shape[-1], units, address)
    if _contiguous(units):
        return flat[..., units[0]:units[0] + len(units)]
    return flat[..., jnp.asarray(units, dtype=jnp.int32)]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _selected_per_step(units: tuple[int, ...] | None, size: int) -> bool:
    """
        Returns whether the ``units`` of a trace or raster are selected on every step.

        True for contiguous ``units``, for at least ``size`` units, and, from `SETTINGS.select_per_step`
        entries, for at most a quarter of the ``size`` entries. When False, ``len(units) < size``.
        `_units_of_rows` relies on it to tell the two cases apart by the width of the rows.
    """
    if units is None:
        return False
    return _contiguous(units) or len(units) >= size or (size >= SETTINGS.select_per_step and 4 * len(units) <= size)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _units_of_rows(probe: TraceProbe | RasterProbe, rows: jax.Array) -> jax.Array:
    """
        Selects the ``units`` of stacked trace or raster rows, when not selected on every step.
    """
    if probe.units is None or rows.shape[-1] == len(probe.units):
        return rows
    return _select(rows, probe.units, probe.address, lead=1)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _histogram(flat: jax.Array, bins: int, value_range: tuple[float, float]) -> jax.Array:
    """
        Counts the values of ``flat`` in ``bins`` equal bins over ``value_range``, as int32.

        The edges are float32 and the upper edge is included. Values outside the range, and NaN, are
        not counted.

        Notes
        -----
        While `_compared` holds, every value is compared with every edge in one reduction. Otherwise
        each value is counted in its bin from `_bin_index`, by scattering, or by sorting past
        `SETTINGS.histogram_scatter_limit` values on backends other than the CPU (`_counts_by_sorting`).
    """
    return _histogram_by(flat.reshape(1, -1), bins, value_range, jnp.zeros((1,), jnp.int32), 1)[0]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _histogram_by(rows: jax.Array, bins: int, value_range: tuple[float, float], segments: jax.Array, count: int) -> jax.Array:
    """
        Counts the values of each row of ``rows`` in the bins of `_histogram`, summed by segment.

        Row ``i`` counts in segment ``segments[i]``. Returns int32 counts of shape
        ``(count, bins)``.
    """
    lower, upper = _bounds(bins, value_range)
    x = rows.reshape(rows.shape[0], -1).astype(jnp.float32)
    if _compared(bins, x.size):
        hits = (x[:, None, :] >= jnp.asarray(lower)[None, :, None]) & (x[:, None, :] < jnp.asarray(upper)[None, :, None])
        return jnp.zeros((count, bins), jnp.int32).at[segments].add(jnp.sum(hits, axis=2, dtype=jnp.int32))
    index = _bin_index(x, bins, value_range)
    # One bin -1 per segment for the values outside.
    combined = (segments[:, None] * (bins + 1) + index + 1).reshape(-1)
    size = count * (bins + 1)
    if not _counts_by_sorting(x.size):
        counts = jnp.bincount(combined, length=size)
    else:
        ends = jnp.searchsorted(jnp.sort(combined), jnp.arange(size, dtype=jnp.int32), side='right', method='scan_unrolled')
        counts = jnp.diff(ends, prepend=0)
    return counts.reshape(count, bins + 1)[:, 1:].astype(jnp.int32)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _bounds(bins: int, value_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """
        Returns the lower and upper float32 edges of each bin.

        The last upper edge is the next float32 above the end of the range. The end of the range is
        counted in the last bin.
    """
    lo, hi = value_range
    edges = np.linspace(lo, hi, bins + 1, dtype=np.float32)
    return edges[:-1], np.concatenate([edges[1:-1], [np.nextafter(np.float32(hi), np.float32(np.inf))]])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _compared(bins: int, size: int) -> bool:
    """
        Returns whether a histogram compares ``size`` values with the edges of ``bins`` bins.

        True when ``bins * size`` is at most `SETTINGS.histogram_compare_limit`, or ``bins`` at most
        `SETTINGS.histogram_compare_bins`.
    """
    return bins * size <= SETTINGS.histogram_compare_limit or bins <= SETTINGS.histogram_compare_bins

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _bin_index(x: jax.Array, bins: int, value_range: tuple[float, float]) -> jax.Array:
    """
        Returns the bin of each float32 value of ``x`` in the bins of `_histogram`.

        Values outside the range, and NaN, are in bin -1.

        Notes
        -----
        The bin is computed from the position of the value in the range, then corrected by at most
        one against the edges, as NumPy does. This requires at most ``2 ** 20`` bins, a finite range
        width and scale, a scale that is not subnormal, and a narrowest bin wider than four float32
        spacings at the ends of the range. Otherwise the bin is found by a binary search among the
        edges.
    """
    lo, hi = value_range
    edges = np.linspace(lo, hi, bins + 1, dtype=np.float32)
    lower, upper = _bounds(bins, value_range)
    with np.errstate(over='ignore', divide='ignore', under='ignore'):
        width = np.float32(edges[-1]) - np.float32(edges[0])
        scale = np.float32(bins / (np.float64(edges[-1]) - np.float64(edges[0])))
    # The position is off by less than half a bin when the narrowest bin spans four float32 spacings, the
    # width of the range is a float32, and the scale is not subnormal, which processors may read as zero.
    spacing = np.spacing(np.float32(max(abs(edges[0]), abs(edges[-1]))))
    exact = np.isfinite(width) and np.isfinite(scale) and scale >= np.finfo(np.float32).tiny
    if bins <= 2 ** 20 and exact and np.diff(edges.astype(np.float64)).min() > 4 * spacing:
        at = jnp.asarray(edges)
        index = jnp.clip(jnp.floor((x - edges[0]) * scale), 0, bins - 1).astype(jnp.int32)
        index = jnp.where(x < at[index], index - 1, index)
        index = jnp.where((x >= at[jnp.minimum(index + 1, bins)]) & (index < bins - 1), index + 1, index)
        index = jnp.where((x >= edges[0]) & (x <= edges[-1]), index, -1)
    else:
        # Edges at or below each value: b + 1 inside bin b, 0 below, bins + 1 above and for NaN.
        position = jnp.searchsorted(jnp.asarray(np.concatenate([lower, upper[-1:]])), x, side='right')
        index = jnp.where(position > bins, 0, position) - 1
    return index.astype(jnp.int32)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _counts_by_sorting(size: int) -> bool:
    """
        Returns whether ``size`` values are counted by sorting their bin indices.

        True past `SETTINGS.histogram_scatter_limit` values, on backends other than the CPU. Otherwise
        the values are scattered into their bins.
    """
    return size > SETTINGS.histogram_scatter_limit and jax.default_backend() != 'cpu'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _add_counts(low: jax.Array, high: jax.Array, counts: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
        Adds int32 ``counts`` to counts held as ``high * 2 ** HISTOGRAM_LOW_BITS + low``.
    """
    total = low.astype(jnp.uint32) + counts.astype(jnp.uint32)
    mask = (1 << HISTOGRAM_LOW_BITS) - 1
    return (total & mask).astype(jnp.int32), high + (total >> HISTOGRAM_LOW_BITS).astype(jnp.int32)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _as_bytes(x: jax.Array) -> jax.Array:
    """
        Returns ``x`` as a flat uint8 array.

        Bool entries take one bit each, the first entry in the highest bit. Other dtypes are
        bitcast.
    """
    flat = x.reshape(-1)
    if flat.dtype == jnp.bool_:
        return jnp.packbits(flat)
    return jax.lax.bitcast_convert_type(flat, jnp.uint8).reshape(-1)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _from_bytes(data: jax.Array, dtype: np.dtype, shape: tuple[int, ...]) -> jax.Array:
    """
        Reverses `_as_bytes` along the last axis of ``data``.

        Leading axes are kept. The last axis becomes ``shape``, in ``dtype``.
    """
    lead = data.shape[:-1]
    if dtype == np.bool_:
        return jnp.unpackbits(data, axis=-1, count=int(np.prod(shape))).astype(jnp.bool_).reshape(*lead, *shape)
    return jax.lax.bitcast_convert_type(data.reshape(*lead, -1, dtype.itemsize), dtype).reshape(*lead, *shape)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _layout(values: dict[tp.Hashable, jax.Array], by_dtype: bool = False) -> tuple[list[jax.Array], Layout]:
    """
        Returns the byte views of ``values``, as `_as_bytes` gives them, and their layout.

        With ``by_dtype``, values are ordered by dtype, wider types first. The entries of one dtype
        are then contiguous and aligned to their item size.
    """
    items = list(values.items())
    if by_dtype:
        items.sort(key=lambda item: (-np.dtype(item[1].dtype).itemsize, np.dtype(item[1].dtype).str))
    parts, layout, offset = [], [], 0
    for path, value in items:
        data = _as_bytes(value)
        parts.append(data)
        layout.append((path, np.dtype(value.dtype), tuple(value.shape), offset, int(data.size)))
        offset += int(data.size)
    return parts, tuple(layout)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _rounded(value: jax.Array) -> jax.Array:
    """
        Returns ``value`` as float32, rounded to its dtype first when that is float16 or bfloat16.

        Rounding is half to even, subnormals included, with overflow to infinity.

        Notes
        -----
        XLA may keep a float16 result in float32 inside a fused kernel
        (``xla_allow_excess_precision``). The model stores the rounded value. Backends that flush
        subnormals to zero, such as XLA on the CPU for bfloat16, read them as zero.
    """
    dtype = jnp.dtype(value.dtype)
    x = value.astype(jnp.float32)
    return _round_to(x, dtype) if dtype in _SIGNIFICAND else x

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _round_to(x: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """
        Rounds float32 ``x`` to the nearest value of ``dtype`` (float16 or bfloat16), as float32.
    """
    bits, lowest = _SIGNIFICAND[jnp.dtype(dtype)]
    _, exponent = jnp.frexp(x)
    scale = jnp.maximum(exponent, lowest) - bits
    rounded = jnp.ldexp(jnp.round(jnp.ldexp(x, -scale)), scale)
    largest = float(jnp.finfo(dtype).max)
    return jnp.where(jnp.abs(rounded) > largest, jnp.copysign(jnp.inf, x), rounded)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def step_value(probe: Probe, value: tp.Any) -> jax.Array:
    """
        Returns what one step contributes to the record of a per-step probe.

        Parameters
        ----------
        probe : Probe
            A probe read on every step.
        value : SparkPayload or Variable or array
            The value read on this step.

        Returns
        -------
        array
            One of the following, by mode.

            * ``summary``: the value, flat.
            * ``raster``: whether each entry is nonzero, flat.
            * ``trace``: the value, flat when ``units`` is given.

            Traces and rasters keep only their ``units`` when these are selected on every step
            (`SETTINGS.select_per_step`).

        Raises
        ------
        ValueError
            When the probe is not read per step, a summary has no entries, a histogram has
            ``2 ** 31`` entries or more, or ``units`` asks for an index past the entries.

        Notes
        -----
        A float16 or bfloat16 value of a raster is rounded to its dtype before the test
        (`_rounded`).
    """
    x = get_recorded_array(value)
    if isinstance(probe, SummaryProbe):
        if x.size == 0:
            raise ValueError(f'"{probe.address}" has no entries to summarize.')
        if 'hist' in probe.reduce and x.size >= 2 ** 31:
            raise ValueError(f'"{probe.address}" has {x.size} entries; a histogram counts at most 2 ** 31 - 1 per step.')
        return x.reshape(-1)
    if not isinstance(probe, (TraceProbe, RasterProbe)):
        raise ValueError(f'"{probe.key}" is not read per step.')
    check_units(x.size, probe.units, probe.address)
    if isinstance(probe, RasterProbe):
        x = (_rounded(x) if jnp.dtype(x.dtype) in _SIGNIFICAND else x) != 0
    elif probe.units is None:
        return x
    flat = x.reshape(-1)
    return _select(flat, probe.units, probe.address) if _selected_per_step(probe.units, flat.size) else flat

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
class StepRecords:
    """
        Values of one step that a call stacks, packed into one byte row.

        `recorded_scan` packs the values of the traces, the rasters, and the summaries whose histogram
        is counted after the call (`uses_rows`). A pytree whose only leaf is the row. The layout is
        static. Returned from the body of ``jax.lax.scan``, the rows of all the steps stack into one
        output of shape ``(steps, bytes)``.

        Parameters
        ----------
        row : array
            uint8 row, or rows stacked along leading axes.
        layout : Layout
            The values of the row, along its last axis.

        Notes
        -----
        Bool values take one bit per entry.
    """

    def __init__(self, row: jax.Array, layout: Layout) -> None:
        self.row = row
        self.layout = layout

    def tree_flatten(self) -> tuple[tuple, Layout]:
        return (self.row,), self.layout

    @classmethod
    def tree_unflatten(cls, layout: Layout, children: tuple) -> StepRecords:
        return cls(children[0], layout)

    def unpack(self) -> dict[str, jax.Array]:
        """
            Unpacks the row into its values.

            Returns
            -------
            dict of str to array
                Values by `Probe.key`, with the leading axes of ``row`` kept.
        """
        return {
            key: _from_bytes(self.row[..., offset:offset + size], dtype, shape)
            for key, dtype, shape, offset, size in self.layout
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def pack_step(values: dict[str, jax.Array]) -> StepRecords | dict:
    """
        Packs the values of one step into a `StepRecords`.

        Parameters
        ----------
        values : dict of str to array
            Values of the step, by `Probe.key`.

        Returns
        -------
        StepRecords or dict
            The values in one row, ordered by dtype, wider types first. ``values`` unchanged when
            empty.
    """
    if not values:
        return values
    parts, layout = _layout(values, by_dtype=True)
    return StepRecords(jnp.concatenate(parts), layout)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def init_spaced(probes: tuple[Probe, ...], values: dict[str, tp.Any], steps: int) -> tuple:
    """
        Creates the buffers of the traces and rasters that write only the steps they keep.

        These are the traces and rasters with a ``stride`` above 1 whose values over the call would take
        more than `SETTINGS.spaced_rows_limit` bytes. The buffers and a step counter are carried through
        the steps of the call. The other traces and rasters with a stride are stacked with the rows of
        every step and strided after the call.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the call.
        values : dict of str to array or ShapeDtypeStruct
            What `ProbeContext.values` gives on one step, or its shapes.
        steps : int
            Steps of the call.

        Returns
        -------
        tuple
            ``(step, buffers)``. ``step`` is the int32 step counter. ``buffers`` holds, by
            `Probe.key`, one row per step the call can keep and a spare row. An empty tuple when no
            probe needs a buffer.
    """
    buffers = {}
    for probe in probes:
        if isinstance(probe, (TraceProbe, RasterProbe)) and probe.stride > 1:
            value = values[probe.key]
            if steps * int(np.prod(value.shape)) * np.dtype(value.dtype).itemsize > SETTINGS.spaced_rows_limit:
                buffers[probe.key] = jnp.zeros((-(-steps // probe.stride) + 1, *value.shape), value.dtype)
    return (jnp.zeros((), jnp.int32), buffers) if buffers else ()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def write_spaced(probes: tuple[Probe, ...], spaced: tuple, values: dict[str, jax.Array], firsts: dict[int, tp.Any] | None = None) -> tuple:
    """
        Writes the values of one step to the buffers of `init_spaced`.

        Step ``i`` of the call writes row ``ceil((i - first) / stride)``, where ``first`` is the
        first step the stride keeps. A step kept is the last to write its row. The steps after the
        last one kept write the next row. That row is the spare row, or the last row of the record
        when the call keeps fewer than ``ceil(steps / stride)`` steps.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the call.
        spaced : tuple
            As given by `init_spaced`, or by the previous step.
        values : dict of str to array
            What `ProbeContext.values` gave on this step.
        firsts : dict of int to int or array, optional
            First step kept by each stride, as `first_kept` gives it. 0 for a stride not in it.

        Returns
        -------
        tuple
            The step counter, advanced by one, and the updated buffers. ``spaced`` unchanged when
            empty.
    """
    if not spaced:
        return spaced
    step, buffers = spaced
    written = {}
    for probe in probes:
        if probe.key in buffers:
            first = (firsts or {}).get(probe.stride, 0)
            row = (step - first + probe.stride - 1) // probe.stride
            written[probe.key] = jax.lax.dynamic_update_index_in_dim(buffers[probe.key], values[probe.key], row, 0)
    return step + 1, written

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def first_kept(stride: int, phase: jax.Array | int) -> jax.Array | int:
    """
        Returns the index within a call of the first step a stride keeps.

        A stride keeps the steps ``t`` of the run with ``t % stride == 0``.

        Parameters
        ----------
        stride : int
            Stride of a trace or raster.
        phase : int or array
            First step of the call modulo ``stride``.

        Returns
        -------
        int or array
            Index of that step within the call, from 0. It may lie past the end of a short call.
    """
    return (stride - phase) % stride

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def read_boundary(model: Controller, probes: tuple[Probe, ...]) -> dict[str, jax.Array]:
    """
        Reads the values of the snapshot and delta probes from the current state of the model.

        `recorded_scan` reads them before and after the steps of a call, and passes them to `finalize`
        as ``start`` and ``end``. It also reads those of `capture` after every step.

        Parameters
        ----------
        model : Controller
            The model.
        probes : tuple of Probe
            Probes of the call. Probes read on every step are skipped.

        Returns
        -------
        dict of str to array
            One entry per ``snapshot`` or ``delta`` probe, by `Probe.key`.

        Raises
        ------
        ValueError
            When a module path is not found, or the value of a delta has no entries.
        TypeError
            When an attribute holds no array.
    """
    values = {}
    for probe in probes:
        if probe.per_step:
            continue
        value = get_recorded_array(getattr(resolve(model, probe.path, probe.address), probe.name))
        if isinstance(probe, DeltaProbe) and value.size == 0:
            raise ValueError(f'"{probe.address}" has no entries to take the change of.')
        values[probe.key] = value
    return values

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def init_captures(probes: tuple[Probe, ...], values: dict[str, tp.Any], steps: int) -> dict[str, jax.Array]:
    """
        Creates the buffers of the values at the ends of groups within a call.

        One buffer per snapshot or delta grouped by steps. The buffers are carried through the steps
        of the call.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the call.
        values : dict of str to array or ShapeDtypeStruct
            What `read_boundary` gives, or its shapes.
        steps : int
            Steps of the call.

        Returns
        -------
        dict of str to array
            Buffers by `Probe.key`, of shape ``(ends + 1, ...)``. One row per end of a group the
            call can hold (`ends`), and a spare row the other steps write. A snapshot keeps only its
            ``units``.
    """
    buffers = {}
    for probe in probes:
        count = ends(probe, steps)
        if count:
            value = _boundary_value(probe, values[probe.key])
            buffers[probe.key] = jnp.zeros((count + 1, *value.shape), value.dtype)
    return buffers

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _boundary_value(probe: SnapshotProbe | DeltaProbe, value: tp.Any) -> tp.Any:
    """
        Returns what a snapshot or delta keeps of ``value``.

        A snapshot keeps its ``units``, and a delta the whole value. ``value`` may be a
        ``jax.ShapeDtypeStruct``.
    """
    if isinstance(probe, SnapshotProbe) and probe.units is not None:
        if isinstance(value, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct((len(probe.units),), value.dtype)
        return _select(value, probe.units, probe.address)
    return value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def capture(
        probes: tuple[Probe, ...],
        buffers: dict[str, jax.Array],
        values: dict[str, jax.Array],
        index: jax.Array,
        start: jax.Array | None,
    ) -> dict[str, jax.Array]:
    """
        Writes the values after step ``index`` of a call to the buffers of `init_captures`.

        The step ending the ``e``-th group of the call, from 0, writes row ``e``. The other steps
        write the spare row.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the call.
        buffers : dict of str to array
            As given by `init_captures`, or by the previous step.
        values : dict of str to array
            What `read_boundary` gives after the step.
        index : array
            Step within the call, from 0.
        start : Start or array or None
            Where the call starts, as `start_of` gives it.

        Returns
        -------
        dict of str to array
            The updated buffers.

        Notes
        -----
        Every step writes its value, without a branch.
    """
    captured = {}
    for probe in probes:
        if probe.key not in buffers:
            continue
        k, buffer = probe.group, buffers[probe.key]
        position = _phase(probes, start, k) + index + 1
        row = jnp.where(position % k == 0, position // k - 1, buffer.shape[0] - 1)
        value = _boundary_value(probe, values[probe.key]).astype(buffer.dtype)
        captured[probe.key] = jax.lax.dynamic_update_index_in_dim(buffer, value, row, 0)
    return captured

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def uses_rows(probe: Probe, accumulators: dict | None = None) -> bool:
    """
        Returns whether a call stacks the values of every step of ``probe`` in its rows.

        True for traces and rasters. Given ``accumulators``, also true for summaries whose histogram
        is counted after the call.

        Parameters
        ----------
        probe : Probe
            Probe of the call.
        accumulators : dict, optional
            As given by `init_accumulators`.

        Returns
        -------
        bool
    """
    if isinstance(probe, (TraceProbe, RasterProbe)):
        return True
    return (
        isinstance(probe, SummaryProbe) and 'hist' in probe.reduce and accumulators is not None
        and 'hist' not in accumulators.get(probe.key, {})
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def accumulated_fields(probe: Probe) -> tuple[str, ...]:
    """
        Returns the names of the running statistics a summary probe keeps.

        ``n`` counts the steps. ``mean``, ``m2``, ``min``, ``max`` and ``active`` are kept per unit.
        ``hist`` and ``hist_high`` are the two words of the histogram counts.

        Parameters
        ----------
        probe : Probe
            Probe of the call.

        Returns
        -------
        tuple of str
            Names of the statistics the reductions of ``probe`` need. Empty for other modes.
    """
    if not isinstance(probe, SummaryProbe):
        return ()
    need = set(probe.reduce)
    # Steps summarized, which merging the partials of one group across calls weighs by.
    fields = ['n']
    if need & {'mean', 'std'}:
        fields.append('mean')
    if 'std' in need:
        fields.append('m2')
    if 'min' in need:
        fields.append('min')
    if 'max' in need:
        fields.append('max')
    if need & {'active_fraction', 'active_fraction_per_unit', 'inactive_unit_fraction'}:
        fields.append('active')
    if 'hist' in need:
        fields += ['hist', 'hist_high']
    return tuple(fields)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _ordered_dtype(dtype: tp.Any) -> np.dtype:
    """
        Returns the dtype in which ``min`` and ``max`` are kept.

        Float32 for floating types, uint8 for bool, and the dtype itself otherwise.
    """
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return np.dtype(np.uint8)
    if jnp.issubdtype(dtype, jnp.floating):
        return np.dtype(np.float32)
    return dtype

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _extremes(dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """
        Returns the identity values of ``min`` and ``max`` in ``dtype``, float32 or an integer type.
    """
    if dtype == np.float32:
        return np.array(np.inf, dtype), np.array(-np.inf, dtype)
    info = np.iinfo(dtype)
    return np.array(info.max, dtype), np.array(info.min, dtype)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def init_accumulators(probes: tuple[Probe, ...], values: dict[str, tp.Any], steps: int | None = None, split: bool = True) -> dict[str, dict[str, jax.Array]]:
    """
        Creates the initial running statistics of the summary probes.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the call.
        values : dict of str to array or ShapeDtypeStruct
            What `ProbeContext.values` gives on one step, or its shapes as ``jax.eval_shape`` gives
            them.
        steps : int, optional
            Steps of the call. Without it, every histogram is counted step by step, and a summary
            keeps one set of statistics.
        split : bool, default True
            Whether the call may cross the end of a group. When False, a summary keeps one set of
            statistics.

        Returns
        -------
        dict of str to dict of str to array
            One entry per summary probe, by `Probe.key`, with the fields of `accumulated_fields`.

        Notes
        -----
        A histogram whose values over the call take at most `SETTINGS.histogram_rows_limit` bytes is
        counted after the call from its rows (`uses_rows`), and keeps no ``hist`` statistics. A summary
        grouped by steps, in a call that may cross the end of a group, keeps one set of statistics for
        each group the call can touch (`slots`), along a leading axis.
    """
    accumulators = {}
    for probe in probes:
        fields = accumulated_fields(probe)
        shape = tuple(values[probe.key].shape) if fields else ()
        itemsize = np.dtype(values[probe.key].dtype).itemsize if fields else 0
        if 'hist' in fields and steps is not None and steps * int(np.prod(shape)) * itemsize <= SETTINGS.histogram_rows_limit:
            fields = tuple(f for f in fields if f not in ('hist', 'hist_high'))
        if not fields:
            continue
        dtype = _ordered_dtype(values[probe.key].dtype)
        low, high = _extremes(dtype)
        lead = (slots(probe, steps),) if split and steps is not None and slots(probe, steps) > 1 else ()
        acc = {}
        if 'n' in fields:
            acc['n'] = jnp.zeros(lead, jnp.int32)
        if 'mean' in fields:
            acc['mean'] = jnp.zeros((*lead, *shape), jnp.float32)
        if 'm2' in fields:
            acc['m2'] = jnp.zeros((*lead, *shape), jnp.float32)
        if 'min' in fields:
            acc['min'] = jnp.full((*lead, *shape), low)
        if 'max' in fields:
            acc['max'] = jnp.full((*lead, *shape), high)
        if 'active' in fields:
            acc['active'] = jnp.zeros((*lead, *shape), jnp.int32)
        if 'hist' in fields:
            acc['hist'] = jnp.zeros((*lead, probe.bins), jnp.int32)
            acc['hist_high'] = jnp.zeros((*lead, probe.bins), jnp.int32)
        accumulators[probe.key] = acc
    return accumulators

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def accumulate(
        probes: tuple[Probe, ...],
        accumulators: dict[str, dict[str, jax.Array]],
        values: dict[str, jax.Array],
        index: jax.Array | None = None,
        start: jax.Array | None = None,
    ) -> dict[str, dict[str, jax.Array]]:
    """
        Updates the running statistics of the summary probes with the values of one step.

        The updates are elementwise, except for histograms, which count the values of the step. Mean
        and spread follow Welford's algorithm. A summary grouped by steps updates the statistics of
        the group holding the step.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the call.
        accumulators : dict
            As given by `init_accumulators`, or by the previous step.
        values : dict of str to array
            What `ProbeContext.values` gave on this step.
        index : array, optional
            Step within the call, from 0. Needed by summaries grouped by steps.
        start : Start or array, optional
            Where the call starts, as `start_of` gives it.

        Returns
        -------
        dict
            The updated statistics.

        Notes
        -----
        Floating values are rounded to their dtype first (`_rounded`). With at most
        `SETTINGS.masked_slots` groups, the statistics of every group are updated, masked to the group
        of the step. With more, those of the group of the step are read and written at a dynamic index.
    """
    updated = {}
    for probe in probes:
        if probe.key not in accumulators:
            continue
        acc, value = accumulators[probe.key], values[probe.key]
        x = _rounded(value) if jnp.issubdtype(value.dtype, jnp.floating) else value
        if acc['n'].ndim == 0:
            updated[probe.key] = _update(probe, acc, x)
            continue
        slot = (_phase(probes, start, probe.group) + index) // probe.group
        if acc['n'].shape[0] <= SETTINGS.masked_slots:
            updated[probe.key] = _update(probe, acc, x, jnp.arange(acc['n'].shape[0]) == slot)
            continue
        new = _update(probe, {k: jax.lax.dynamic_index_in_dim(v, slot, 0, keepdims=False) for k, v in acc.items()}, x)
        updated[probe.key] = {k: jax.lax.dynamic_update_index_in_dim(acc[k], v, slot, 0) for k, v in new.items()}
    return updated

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _update(probe: SummaryProbe, acc: dict[str, jax.Array], x: jax.Array, mask: jax.Array | None = None) -> dict[str, jax.Array]:
    """
        Updates the statistics ``acc`` of one group with the values ``x`` of one step.

        With ``mask``, ``acc`` holds every group along a leading axis. The groups where ``mask`` is
        true are updated, and the others kept.
    """
    if mask is None:
        pick = lambda new, old: new
        n = acc['n'] + 1
    else:
        lead = lambda a: mask.reshape(mask.shape + (1,) * (a.ndim - 1))
        pick = lambda new, old: jnp.where(lead(old), new, old)
        n = acc['n'] + mask.astype(jnp.int32)
    new = {'n': n}
    if 'mean' in acc:
        wide = x.astype(jnp.float32)
        delta = wide - acc['mean']
        count = n.reshape(n.shape + (1,) * (acc['mean'].ndim - n.ndim)).astype(jnp.float32)
        mean = pick(acc['mean'] + delta / jnp.maximum(count, 1.0), acc['mean'])
        new['mean'] = mean
        if 'm2' in acc:
            new['m2'] = pick(acc['m2'] + delta * (wide - mean), acc['m2'])
    if 'min' in acc:
        new['min'] = pick(jnp.minimum(acc['min'], x.astype(acc['min'].dtype)), acc['min'])
    if 'max' in acc:
        new['max'] = pick(jnp.maximum(acc['max'], x.astype(acc['max'].dtype)), acc['max'])
    if 'active' in acc:
        new['active'] = pick(acc['active'] + (x != 0).astype(jnp.int32), acc['active'])
    if 'hist' in acc:
        counts = _histogram(x.reshape(-1), probe.bins, probe.range)
        if mask is not None:
            counts = counts * mask[:, None].astype(jnp.int32)
        new['hist'], new['hist_high'] = _add_counts(acc['hist'], acc['hist_high'], counts)
    return new

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _summary_complete(probe: SummaryProbe, partials: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
        Computes the reductions of a summary probe from its partials, on the host.

        ``std`` pools the spread within each unit, around its own mean, and the spread of the unit
        means. Raises ValueError for partials of no steps.
    """
    need = set(probe.reduce)
    out = {}
    steps = int(partials['n']) if 'n' in partials else None
    if steps == 0:
        raise ValueError(f'"{probe.key}" summarizes no steps.')
    if need & {'mean', 'std'}:
        unit_mean = partials['mean'].astype(np.float64)
        mean = np.mean(unit_mean)
        if 'mean' in need:
            out['mean'] = np.float32(mean)
        if 'std' in need:
            # Spread within each unit, around its own mean, plus the spread of the unit means.
            within = np.sum(partials['m2'], dtype=np.float64)
            variance = (within + steps * np.sum((unit_mean - mean) ** 2)) / (steps * unit_mean.size)
            out['std'] = np.float32(np.sqrt(max(variance, 0.0)))
    if 'min' in need:
        out['min'] = np.min(partials['min'])
    if 'max' in need:
        out['max'] = np.max(partials['max'])
    if need & {'active_fraction', 'active_fraction_per_unit', 'inactive_unit_fraction'}:
        active = partials['active']
        if 'active_fraction' in need:
            out['active_fraction'] = np.float32(np.sum(active, dtype=np.int64) / (steps * active.size))
        if 'active_fraction_per_unit' in need:
            out['active_fraction_per_unit'] = (active / steps).astype(np.float32)
        if 'inactive_unit_fraction' in need:
            out['inactive_unit_fraction'] = np.float32(np.mean(active == 0))
    if 'hist' in need:
        out['hist'] = _hist_counts(partials)
    return {name: out[name] for name in probe.reduce}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def merge_summaries(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """
        Merges the partials of a summary over several calls, on the host.

        * Step counts and ``active`` counts are added.
        * Means are weighted by steps.
        * Spreads follow the parallel form of Welford's algorithm.
        * ``min`` and ``max`` take the extremes of the parts.
        * Histogram counts are combined as int64.

        Parameters
        ----------
        parts : list of dict of str to array
            Partials of one group from each call, on the host.

        Returns
        -------
        dict of str to array
            Partials over all their steps, as `_summary_complete` takes them. A single part is
            returned as is.
    """
    if len(parts) == 1:
        return parts[0]
    steps = np.array([int(part['n']) for part in parts], np.int64)
    total = int(steps.sum())
    out: dict[str, np.ndarray] = {'n': np.int64(total)}
    first = parts[0]
    if 'mean' in first:
        means = np.stack([np.asarray(part['mean'], np.float64) for part in parts])
        weights = steps.reshape(-1, *([1] * (means.ndim - 1))).astype(np.float64)
        mean = (weights * means).sum(axis=0) / max(total, 1)
        out['mean'] = mean
        if 'm2' in first:
            within = np.stack([np.asarray(part['m2'], np.float64) for part in parts]).sum(axis=0)
            out['m2'] = within + (weights * (means - mean) ** 2).sum(axis=0)
    if 'min' in first:
        out['min'] = np.min(np.stack([part['min'] for part in parts]), axis=0)
    if 'max' in first:
        out['max'] = np.max(np.stack([part['max'] for part in parts]), axis=0)
    if 'active' in first:
        out['active'] = np.stack([np.asarray(part['active'], np.int64) for part in parts]).sum(axis=0)
    if 'hist' in first:
        out['hist'] = sum(_hist_counts(part) for part in parts)
        out['hist_high'] = np.zeros_like(out['hist'])
    return out

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _hist_counts(partials: dict[str, np.ndarray]) -> np.ndarray:
    """
        Returns the int64 counts of a histogram from its two int32 words.

        A missing ``hist_high`` counts as zero.
    """
    high = np.asarray(partials.get('hist_high', 0)).astype(np.int64)
    return (high << HISTOGRAM_LOW_BITS) + np.asarray(partials['hist']).astype(np.int64)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _delta_partials(probe: DeltaProbe, start: jax.Array, end: jax.Array) -> dict[str, jax.Array]:
    """
        Computes the partials of a delta probe from its values ``start`` and ``end``.

        The change ``end - start`` is taken in float32. A row is one index of its first axis, or the
        whole change when it has fewer than two axes. Each entry is present with its reduction.

        * ``full`` (reduction ``full``): the change.
        * ``squares`` (reduction ``norm``): the sum of squares of each row.
        * ``abs`` (reduction ``mean_abs``): the mean absolute value of each row.
    """
    change = end.astype(jnp.float32) - start.astype(jnp.float32)
    out = {}
    if 'full' in probe.reduce:
        out['full'] = change
    rows = change.reshape(change.shape[0] if change.ndim > 1 else 1, -1)
    if 'norm' in probe.reduce:
        out['squares'] = jnp.sum(rows * rows, axis=1)
    if 'mean_abs' in probe.reduce:
        out['abs'] = jnp.mean(jnp.abs(rows), axis=1)
    return out

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _delta_complete(probe: DeltaProbe, partials: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
        Computes the reductions of a delta probe from its partials, on the host.

        ``norm`` is the square root of the summed squares. ``mean_abs`` is the mean of the row
        means, equal to the mean over all entries.
    """
    out = {}
    if 'full' in probe.reduce:
        out['full'] = np.asarray(partials['full'])
    if 'norm' in probe.reduce:
        out['norm'] = np.float32(np.sqrt(np.sum(partials['squares'], dtype=np.float64)))
    if 'mean_abs' in probe.reduce:
        out['mean_abs'] = np.float32(np.mean(partials['abs'], dtype=np.float64))
    return {name: out[name] for name in probe.reduce}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def finalize(
        probes: tuple[Probe, ...],
        rows: StepRecords | dict[str, jax.Array],
        accumulators: dict[str, dict[str, jax.Array]],
        start: dict[str, jax.Array] | None = None,
        end: dict[str, jax.Array] | None = None,
        spaced: tuple = (),
        phases: Start | jax.Array | None = None,
    ) -> dict[str, jax.Array | dict[str, jax.Array]]:
    """
        Builds the records of a call on the device, to move to the host.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the call.
        rows : StepRecords or dict of str to array
            Rows of every step, as `pack_step` or `ProbeContext.collect` gives them, stacked along a
            leading step axis by ``jax.lax.scan``.
        accumulators : dict
            Running statistics after the last step.
        start, end : dict of str to array, optional
            What `read_boundary` returned before and after the call. The ungrouped ``delta`` needs
            both, and the ungrouped ``snapshot`` needs ``end``.
        spaced : tuple, optional
            Buffers of `write_spaced` after the last step. Traces and rasters with a stride that are
            not in them are read from ``rows``.
        phases : Start or array, optional
            Where the call starts, as `start_of` gives it. Groups and strides are aligned on the
            steps of the run. Without it, the call starts a group and every stride.

        Returns
        -------
        dict
            One entry per probe moved to the host, by `Probe.key`.

            * ``trace`` and ``raster``: the final records. With a stride, ``ceil(steps / stride)``
              rows. Rows past the steps kept hold the last step of the call.
            * ``summary``: the running statistics (`accumulated_fields`), with a leading axis of
              groups for a summary with several groups (`slots`).
            * ungrouped ``delta``: the partials per row (`_delta_partials`).
            * ungrouped ``snapshot``: the value, restricted to ``units``.

            `complete` finishes them on the host. Snapshots and deltas with a group are kept on the
            device by `held`.
    """
    if isinstance(rows, StepRecords):
        rows = rows.unpack()
    start = start or {}
    end = end or {}
    records = {}
    for probe in probes:
        key = probe.key
        if isinstance(probe, SummaryProbe):
            partials = dict(accumulators.get(key, {}))
            if 'hist' in probe.reduce and 'hist' not in partials:
                values = _rounded(rows[key])
                count = slots(probe, values.shape[0]) if _split(phases) else 1
                if count == 1:
                    partials['hist'] = _histogram(values.reshape(-1), probe.bins, probe.range)
                else:
                    segments = (_phase(probes, phases, probe.group) + jnp.arange(values.shape[0], dtype=jnp.int32)) // probe.group
                    partials['hist'] = _histogram_by(values, probe.bins, probe.range, segments, count)
            records[key] = partials
        elif isinstance(probe, (TraceProbe, RasterProbe)):
            if spaced and key in spaced[1]:
                kept = spaced[1][key][:-1]
            elif probe.stride == 1:
                kept = rows[key]
            elif phases is None:
                kept = rows[key][::probe.stride]
            else:
                steps = rows[key].shape[0]
                first = first_kept(probe.stride, _phase(probes, phases, probe.stride))
                at = jnp.minimum(first + probe.stride * jnp.arange(-(-steps // probe.stride), dtype=jnp.int32), steps - 1)
                kept = rows[key][at]
            records[key] = _units_of_rows(probe, kept)
        elif probe.group is not None:
            continue
        elif isinstance(probe, SnapshotProbe):
            records[key] = _select(end[key], probe.units, probe.address)
        elif isinstance(probe, DeltaProbe):
            records[key] = _delta_partials(probe, start[key], end[key])
    return records

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def held(
        probes: tuple[Probe, ...],
        start: dict[str, jax.Array],
        end: dict[str, jax.Array],
        captured: dict[str, jax.Array],
    ) -> dict[str, dict[str, jax.Array]]:
    """
        Returns the values a call keeps on the device for its snapshots and deltas with a group.

        The recorder reads them once it knows where the groups end.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the call.
        start, end : dict of str to array
            What `read_boundary` returned before and after the call.
        captured : dict of str to array
            Buffers of `capture` after the last step.

        Returns
        -------
        dict of str to dict of str to array
            By `Probe.key`, values with a leading axis of rows.

            * ``last``: the value after the last step, one row. It ends a group at the end of the
              call, a group by tag, or a group cut short.
            * ``ends``: the values at the ends of groups within the call, the last row spare.
              Present for a group of steps in a call that may cross the end of one.
            * ``first``: the value before the first step, one row. Deltas only.
    """
    out = {}
    for probe in probes:
        if not isinstance(probe, BOUNDARY_PROBES) or probe.group is None:
            continue
        # With a leading axis of one row, as the ends: `group_values` takes a row of any of them.
        values = {'last': _boundary_value(probe, end[probe.key])[None]}
        if probe.key in captured:
            values['ends'] = captured[probe.key]
        if isinstance(probe, DeltaProbe):
            values['first'] = start[probe.key][None]
        out[probe.key] = values
    return out

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@functools.partial(jax.jit, static_argnums=0)
def group_values(probes: tuple[Probe, ...], ends: tuple, end_rows: tuple, starts: tuple, start_rows: tuple) -> tuple:
    """
        Computes what the snapshots and deltas of a set of measurements record for one group.

        Runs on the device, from the values `held` kept, in one call.

        Parameters
        ----------
        probes : tuple of Probe
            Snapshots and deltas with a group. Static.
        ends : tuple of array
            For each probe, kept values holding the value after the last step of the group.
        end_rows : tuple of int32
            For each probe, the row of ``ends`` holding that value.
        starts : tuple of array or None
            For each delta, kept values holding the value before the first step of the group. None
            for a snapshot.
        start_rows : tuple of int32
            For each delta, the row of ``starts`` holding that value.

        Returns
        -------
        tuple
            For each probe, the snapshot value, or the delta partials (`_delta_partials`) from the
            value before the group to the value after it.

        Notes
        -----
        Jitted with ``probes`` static. The rows are traced. Compiles once per set of probes and
        shapes.
    """
    out = []
    for probe, end, end_row, start, start_row in zip(probes, ends, end_rows, starts, start_rows):
        value = jax.lax.dynamic_index_in_dim(end, end_row, 0, keepdims=False)
        if isinstance(probe, SnapshotProbe):
            out.append(value)
        else:
            out.append(_delta_partials(probe, jax.lax.dynamic_index_in_dim(start, start_row, 0, keepdims=False), value))
    return tuple(out)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def complete(probes: tuple[Probe, ...], records: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """
        Completes the records `finalize` returned, on the host.

        Parameters
        ----------
        probes : tuple of Probe
            Probes of the call.
        records : dict
            As returned by `finalize`, moved to the host.

        Returns
        -------
        dict
            One entry per probe of ``records``, by `Probe.key`, as NumPy arrays.

            * ``summary``: one entry per reduction, over steps and units. ``mean``, ``std``,
              ``active_fraction`` and ``inactive_unit_fraction`` are float32 scalars. ``min`` and
              ``max`` are scalars in float32 for floating values, uint8 for bool, and the integer
              type of integer values. ``active_fraction_per_unit`` holds one float32 per unit, and
              ``hist`` the int64 counts. A summary with a leading axis of groups gives a list of
              such entries, one per group holding steps of the call.
            * ``trace``: ``(ceil(steps / stride), ...)``, in the dtype read.
            * ``raster``: ``(ceil(steps / stride), units)`` bool.
            * ``snapshot``: the value at the end, restricted to ``units``.
            * ``delta``: one entry per reduction of ``end - start``, in float32.
    """
    out = {}
    for probe in probes:
        if probe.key not in records:
            continue
        record = records[probe.key]
        if isinstance(probe, SummaryProbe):
            partials = {k: np.asarray(v) for k, v in record.items()}
            if np.ndim(partials.get('n', 0)) == 0:
                out[probe.key] = _summary_complete(probe, partials)
            else:
                out[probe.key] = [
                    _summary_complete(probe, {k: v[slot] for k, v in partials.items()})
                    for slot in range(len(partials['n'])) if partials['n'][slot]
                ]
        elif isinstance(probe, DeltaProbe):
            out[probe.key] = _delta_complete(probe, {k: np.asarray(v) for k, v in record.items()})
        else:
            out[probe.key] = np.asarray(record)
    return out

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
class Packed:
    """
        Records of a call, packed into one byte buffer, and the values kept on the device.

        A pytree whose leaves are the buffer and the held values. The layout and the probes are
        static. `recorded_scan` returns one, and `Recorder.push` takes it.

        Parameters
        ----------
        buffer : array
            uint8 buffer.
        layout : Layout
            The records in the buffer, keyed by their path in the nested dictionary of records.
        probes : tuple of Probe
            Probes the records belong to.
        held : dict, optional
            What `held` keeps on the device, by `Probe.key`. Not part of the buffer.

        Notes
        -----
        Returned from a jitted function, the buffer moves to the host in one transfer. Bool records,
        such as rasters, take one bit per entry.

        See Also
        --------
        spark.scan : ``jax.lax.scan``, recorded within a call of a `spark.jit` function.
        Recorder.push : Hands over the records of a call.
    """

    def __init__(self, buffer: jax.Array | np.ndarray, layout: Layout, probes: tuple[Probe, ...], held: dict | None = None) -> None:
        self.buffer = buffer
        self.layout = layout
        self.probes = probes
        self.held = held or {}

    def tree_flatten(self) -> tuple[tuple, tuple]:
        return (self.buffer, self.held), (self.layout, self.probes)

    @classmethod
    def tree_unflatten(cls, aux: tuple, children: tuple) -> Packed:
        return cls(children[0], *aux, held=children[1])

    @property
    def nbytes(self) -> int:
        """
            Size of the buffer in bytes.
        """
        return int(self.buffer.size)

    def unpack(self, completed: bool = True) -> dict[str, tp.Any]:
        """
            Unpacks the records as nested dictionaries of NumPy arrays.

            Blocks until the buffer is on the host.

            Parameters
            ----------
            completed : bool, default True
                Whether to complete the records with `complete`. When False, they are returned as
                `finalize` gave them.

            Returns
            -------
            dict
                Records by `Probe.key`. The held values are not included.
        """
        data = np.asarray(self.buffer)
        records: dict[str, tp.Any] = {}
        for path, dtype, shape, offset, size in self.layout:
            if dtype == np.bool_:
                value = np.unpackbits(data[offset:offset + size], count=int(np.prod(shape))).astype(bool).reshape(shape)
            else:
                value = data[offset:offset + size].view(dtype).reshape(shape)
            node = records
            for name in path[:-1]:
                node = node.setdefault(name, {})
            node[path[-1]] = value
        return complete(self.probes, records) if completed else records

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def pack(probes: tuple[Probe, ...], records: dict[str, tp.Any], kept: dict[str, tp.Any] | None = None) -> Packed | dict:
    """
        Packs the records of a call into one byte buffer.

        Parameters
        ----------
        probes : tuple of Probe
            Probes passed to `finalize`, stored for `Packed.unpack`.
        records : dict
            As returned by `finalize`.
        kept : dict, optional
            As returned by `held`. Kept apart from the buffer.

        Returns
        -------
        Packed or dict
            The records in one buffer, with ``kept``. ``records`` unchanged when both are empty.
    """
    if not records and not kept:
        return records
    leaves = {
        tuple(entry.key for entry in path): leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(records)
    }
    parts, layout = _layout(leaves)
    buffer = jnp.concatenate(parts) if parts else jnp.zeros((0,), jnp.uint8)
    return Packed(buffer, layout, tuple(probes), kept)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
