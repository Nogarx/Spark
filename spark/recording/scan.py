#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import jax
import jax.numpy as jnp

from spark.recording.probe import Probe, TraceProbe, RasterProbe
from spark.recording.probe_context import ProbeContext, ModelCalls
from spark.recording.reduce import (
    read_boundary, init_accumulators, accumulated_fields, accumulate, 
    uses_rows, pack_step, init_spaced, write_spaced, first_kept, init_captures, 
    capture, finalize, held, pack, _phase, _split, Start, Packed,
)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _first(xs: tp.Any) -> tp.Any:
    """
        Returns the first step of ``xs``, or None.
    """
    return None if xs is None else jax.tree.map(lambda a: a[0], xs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def recorded_scan(
        f: tp.Callable[[tp.Any, tp.Any], tuple[tp.Any, tp.Any]],
        init: tp.Any,
        xs: tp.Any = None,
        *,
        steps: int,
        probes: tuple[Probe, ...],
        start: Start | jax.Array | None = None,
        unroll: int | bool = 1,
        pack_steps: bool = True,
        split_transpose: bool = False,
    ) -> tuple[tp.Any, tp.Any, Packed | dict]:
    """
        Scans ``f`` over ``steps`` steps of a model and records ``probes``.

        ``f`` takes and returns what ``jax.lax.scan`` passes to it, and calls the model once per
        step. The model is found by the probe context, where it reports its call. `spark.scan` and
        `Runner` record with it.

        Parameters
        ----------
        f : callable
            ``f(carry, x) -> (carry, y)``, one step of the model.
        init, xs
            As for ``jax.lax.scan``.
        steps : int
            Number of steps, at least 1. With ``xs``, the length of its leading axis.
        probes : tuple of Probe
            What to record. Without probes, ``jax.lax.scan``.
        start : Start or array, optional
            Where the call starts on the steps of the run, as `Recorder.start` gives it.
        unroll : int or bool, default 1
            Passed to ``jax.lax.scan``.
        pack_steps : bool, default True
            Whether to pack the values each step adds to the records into one byte row.
        split_transpose : bool, default False
            Passed to ``jax.lax.scan``.

        Returns
        -------
        carry, ys
            As ``jax.lax.scan`` returns them.
        records : Packed or dict
            The records of the call in one buffer, and the values kept on the device for the
            snapshots and deltas with a group. An empty dictionary without probes.

        Raises
        ------
        RuntimeError
            When ``f`` does not call the model once per step, or when a port a probe asks for is
            not produced during a step.

        Notes
        -----
        The values of snapshots and deltas before the first step and after the last one are read
        from the model as ``f`` builds it from the carry, where it calls the model. ``f`` runs up to
        that call outside the scan, and what it computes there is not used otherwise.
    """
    probes = tuple(probes)
    if not probes:
        carry, ys = jax.lax.scan(f, init, xs, length=steps, unroll=unroll, _split_transpose=split_transpose)
        return carry, ys, {}
    first = _first(xs)
    boundary = tuple(p for p in probes if not p.per_step)
    read = lambda model: read_boundary(model, boundary)
    before = ModelCalls(read).run(f, init, first) if boundary else {}

    def recorded_step(carry: tp.Any, x: tp.Any) -> tuple[tp.Any, tp.Any, ProbeContext]:
        with ProbeContext(probes) as context:
            carry, y = f(carry, x)
        if context.calls != 1:
            raise RuntimeError(
                f'The step function of `spark.scan` called the model {context.calls} times in one step. Each step of a '
                f'recorded scan is one call of the model.'
            )
        return carry, y, context

    spaced_probes = tuple(p for p in probes if isinstance(p, (TraceProbe, RasterProbe)) and p.stride > 1)
    # The first step of the call kept by each stride.
    firsts = {m: first_kept(m, _phase(probes, start, m)) for m in {p.stride for p in spaced_probes}}
    accumulators, spaced = {}, ()
    if spaced_probes or any(accumulated_fields(p) for p in probes):
        value_shapes = jax.eval_shape(lambda carry, xs: recorded_step(carry, _first(xs))[2].values(), init, xs)
        accumulators = init_accumulators(probes, value_shapes, int(steps), _split(start))
        spaced = init_spaced(probes, value_shapes, int(steps))
    # Snapshots and deltas grouped by steps keep the values at the ends of their groups.
    # A call that crosses no end of a group keeps none: the value after the call ends the group.
    captures = init_captures(probes, before, int(steps)) if _split(start) else {}
    captured_probes = tuple(p for p in probes if p.key in captures)

    def step(carry: tuple, x: tp.Any) -> tuple:
        carry, accumulators, spaced, captures, index = carry
        carry, y, context = recorded_step(carry, x)
        values = context.values()
        accumulators = accumulate(probes, accumulators, values, index, start)
        spaced = write_spaced(spaced_probes, spaced, values, firsts)
        if captures:
            captures = capture(probes, captures, read_boundary(context.model, captured_probes), index, start)
        rows = {p.key: values[p.key] for p in probes if uses_rows(p, accumulators) and not (spaced and p.key in spaced[1])}
        return (carry, accumulators, spaced, captures, index + 1), (y, pack_step(rows) if pack_steps else rows)

    initial = (init, accumulators, spaced, captures, jnp.zeros((), jnp.int32))
    (carry, accumulators, spaced, captures, _), (ys, rows) = jax.lax.scan(
        step, initial, xs, length=steps, unroll=unroll, _split_transpose=split_transpose,
    )
    after = ModelCalls(read).run(f, carry, first) if boundary else {}
    records = finalize(probes, rows, accumulators, before, after, spaced, start)
    return carry, ys, pack(probes, records, held(probes, before, after, captures))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
