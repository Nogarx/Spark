#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
if tp.TYPE_CHECKING:
    from spark.nn.controllers.base import Controller

import os
import pathlib
import warnings
import functools

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from spark.core.backend import split, merge
from spark.core.payloads import SparkPayload, SpikeArray
from spark.core.recording_hooks import active_probe_context
from spark.recording.probe import CALL, Probe, validate
from spark.recording.measurements import merge_probes
from spark.recording.scan import recorded_scan, _first
from spark.recording.reduce import start_of, warmup_starts, read_boundary, moduli, _phases, Start, Packed
from spark.recording.recorder import Recorder
from spark.recording.calls import layout_of, check_memory
from spark.recording.utils import integer, hold_interrupt

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

OUTPUTS = ('last', 'all', 'none')
"""
    Accepted values of the ``outputs`` argument of `Runner`.
"""

_STATIC = ('steps', 'probes', 'outputs', 'unroll', 'pack_steps', 'sharding')
"""
    Static arguments of `_run_call`.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _input_formats(model: Controller) -> dict[str, tuple[type[SparkPayload], tp.Any, tuple[int, ...] | None]]:
    """
        Returns the payload type, dtype and shape of every input of a controller.
    """
    from spark.nn.controllers.base import Controller
    formats: dict[str, tuple[type[SparkPayload], tp.Any, tuple[int, ...] | None]] = {}
    for spec in model._modules_specs:
        child = getattr(model, spec.name)
        for input_name, maps in spec.inputs.items():
            for port_map in maps:
                # A shape found elsewhere is kept; an input only seen with other maps so far gets it here.
                if port_map.origin != CALL or formats.get(port_map.port, (None, None, None))[2] is not None:
                    continue
                if isinstance(child, Controller):
                    inner = _input_formats(child).get(input_name)
                    if inner is not None:
                        # With several maps, the input is a part of what the child receives.
                        formats[port_map.port] = inner if len(maps) == 1 else (*inner[:2], None)
                else:
                    port_spec = child.get_input_specs().get(input_name)
                    if port_spec is not None:
                        shape = port_spec.shape
                        known = isinstance(shape, tuple) and all(isinstance(d, int) for d in shape) and len(maps) == 1
                        formats[port_map.port] = (port_spec.payload_type, port_spec.dtype, shape if known else None)
    return formats

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _structure(payload_type: type[SparkPayload], dtype: tp.Any) -> jax.tree_util.PyTreeDef:
    """
        Returns the pytree structure of a payload of ``payload_type``.
    """
    if issubclass(payload_type, SpikeArray):
        return jax.tree.structure(payload_type(jnp.zeros((1,), jnp.uint8)))
    return jax.tree.structure(payload_type(jnp.zeros((1,), dtype or jnp.float32)))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _scan(
        graph: nnx.GraphDef,
        state: nnx.State,
        inputs: dict[str, SparkPayload] | None = None,
        per_step: dict[str, SparkPayload] | None = None,
        *,
        steps: int,
        probes: tuple[Probe, ...] = (),
        outputs: str = 'last',
        unroll: int = 1,
        pack_steps: bool = True,
        start: Start | jax.Array | None = None,
    ) -> tuple[dict[str, SparkPayload] | None, nnx.State, Packed | dict]:
    """
        Advances a model ``steps`` steps and records ``probes``, for `Runner`.

        The recorded scan of a model given by its graph and state. Call it inside a jitted function,
        with ``steps``, ``probes``, ``outputs``, ``unroll`` and ``pack_steps`` static. ``start`` is
        traced.

        Parameters
        ----------
        graph, state : GraphDef, State
            The model, as given by `spark.split`.
        inputs : dict of str to SparkPayload, optional
            Inputs given on every step. `Runner` converts arrays to payloads.
        per_step : dict of str to SparkPayload, optional
            Inputs given step by step, stacked along a leading axis of length ``steps``.
        steps : int
            Number of steps, at least 1.
        probes : tuple of Probe, default ()
            What to record. Without probes, the call traces to the same program as one written
            without them.
        outputs : {'last', 'all', 'none'}, default 'last'
            ``'last'`` returns the outputs of the last step, ``'all'`` those of every step stacked,
            and ``'none'`` nothing.
        unroll : int, default 1
            Passed to ``jax.lax.scan``.
        pack_steps : bool, default True
            Whether to pack the values each step adds to the records into one byte row
            (`StepRecords`).
        start : Start or array, optional
            Where the call starts on the steps of the run, as `Recorder.start` gives it for
            ``probes``. Groups of steps and strides are aligned on the steps of the run. Without it,
            the call starts at step 0.

        Returns
        -------
        outputs : dict of str to SparkPayload or None
            Outputs of the last step, or of every step stacked, as ``outputs`` asks. None for
            ``'none'``.
        state : State
            The state after the last step.
        records : Packed or dict
            The records of the call in one buffer, and the values kept on the device for the
            snapshots and deltas with a group. An empty dictionary without probes.

        Raises
        ------
        ValueError
            When ``start`` does not hold one phase per group size and stride of ``probes``.
        RuntimeError
            When a port a probe asks for is not produced during a step.

        Notes
        -----
        With the state sharded across devices, packing the values of a step gathers them on one
        device. `Runner` passes ``pack_steps=False`` then.
    """
    from spark.nn.controllers.base import Controller
    inputs = inputs or {}
    probes = tuple(probes)
    if start is not None and tuple(jnp.shape(_phases(start))) != (len(moduli(probes)),):
        raise ValueError(
            f'"start" holds {jnp.shape(_phases(start))} phases; the probes need {len(moduli(probes))}. It is given by '
            f'`Recorder.start` for the same probes.'
        )

    def model_step(carry: tuple, x: dict | None) -> tuple:
        state, last = carry
        model = merge(graph, state)
        if not isinstance(model, Controller):
            # A controller reports its calls to the probe context; another module is reported here.
            context = active_probe_context()
            if context is not None:
                context.called(model)
        out = model(**inputs, **(x or {}))
        _, state = split((model))
        return (state, out if outputs == 'last' else None), (out if outputs == 'all' else None)

    last = None
    if outputs == 'last':
        shapes = jax.eval_shape(lambda state, xs: model_step((state, None), _first(xs))[0][1], state, per_step)
        last = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), shapes)
    # Without probes, the scan of a call written without them.
    if probes:
        (state, last), stacked, records = recorded_scan(
            model_step, (state, last), per_step, steps=int(steps), probes=probes, start=start, unroll=unroll, pack_steps=pack_steps,
        )
    else:
        (state, last), stacked = jax.lax.scan(model_step, (state, last), per_step, length=int(steps), unroll=unroll)
        records = {}
    return (last if outputs == 'last' else stacked), state, records

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _run_call(graph, state, inputs, per_step, start, steps, probes, outputs, unroll, pack_steps, sharding):
    """
        Runs `_scan` over the steps of one call.

        The records are constrained to ``sharding`` when it is given.
    """
    outputs, state, records = _scan(
        graph, state, inputs, per_step, steps=steps, probes=probes, outputs=outputs, unroll=unroll, pack_steps=pack_steps,
        start=start,
    )
    if sharding is not None and records:
        records = jax.lax.with_sharding_constraint(records, sharding)
    return outputs, state, records

_donated_call = jax.jit(_run_call, static_argnames=_STATIC, donate_argnames=('state',))
_call = jax.jit(_run_call, static_argnames=_STATIC)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Runner:
    """
        Steps a model and records what its recorder asks for.

        The runner holds its own copy of the model state. Each `run` advances it by one compiled
        ``jax.lax.scan``. The recorder gives the probes before the call and receives the records
        after it, without waiting for them.

        Parameters
        ----------
        model : Controller
            A built model, called once with example inputs.
        recorder : Recorder or str or path-like, optional
            Where the records go. A path creates a `Recorder` there with the default measurements of
            the model (`presets.default`). Without one, nothing is recorded.
        outputs : str, default 'last'
            What `run` returns. ``'last'`` gives the outputs of the last step, ``'all'`` those of
            every step stacked, and ``'none'`` nothing.
        unroll : int, default 1
            Passed to ``jax.lax.scan``.
        donate : bool, default True
            Whether each call reuses the memory of the state it receives. The runner then copies the
            state of ``model`` first. The model given is not modified.

        Attributes
        ----------
        state : State
            The current state, as given by ``spark.split``.
        graph : GraphDef
            The graph of the model, as given by ``spark.split``.
        recorder : Recorder or None
            The recorder given or created, or None.

        Raises
        ------
        ValueError
            When ``model`` is not built, or ``outputs`` is unknown.

        Notes
        -----
        The probes of a recorder given are validated against the model. A recorder given to a second
        runner gives a warning. The second runner starts from the state of the model given, while
        the run goes on from its current step.

        Before its first `run` or `warmup`, the runner traces the model with the probes of every set
        of measurements of the recorder, without compiling it. A probe that cannot record the model
        raises there, before its measurements are first recorded. Inputs are checked against the
        shapes the model was built with.

        With the state sharded across devices, the values of a step are not packed into one row.
        With several processes, the records of a call are replicated on every device, where
        process 0 reads them.

        See Also
        --------
        Recorder : Decides what each call records and writes it to a run.
        spark.jit : Compiles a function whose calls the open recorder records.
        Run : A run written by a `Recorder`, opened for reading.

        Examples
        --------
        >>> runner = spark.recording.Runner(brain, 'runs')
        >>> for episode in range(100):
        ...     runner.recorder.tag(episode=episode)
        ...     outputs = runner.run(50, {'signal': observation})
        >>> runner.close()
    """

    def __init__(
            self,
            model: Controller,
            recorder: Recorder | str | os.PathLike | None = None,
            *,
            outputs: str = 'last',
            unroll: int = 1,
            donate: bool = True,
        ) -> None:
        if not getattr(model, '__built__', False):
            raise ValueError('The model is not built yet. Call it once with example inputs first.')
        if outputs not in OUTPUTS:
            raise ValueError(f'Unknown outputs "{outputs}". Expected one of: {", ".join(OUTPUTS)}.')
        if isinstance(recorder, (str, os.PathLike)):
            recorder = Recorder(recorder, model)
        elif recorder is not None:
            # A recorder made before the model was built has not checked its probes.
            for measurements in recorder.measurements:
                validate(model, measurements.probes)
        if recorder is not None:
            if getattr(recorder, '_runners', 0):
                warnings.warn(
                    'The recorder has a runner already: this one starts from the state of the model given, while the '
                    'run goes on from its current step.'
                )
            recorder._runners = getattr(recorder, '_runners', 0) + 1
        self.recorder = recorder
        self.outputs = outputs
        self.unroll = int(unroll)
        self._formats = _input_formats(model)
        self._structures = {name: _structure(payload_type, dtype) for name, (payload_type, dtype, _) in self._formats.items()}
        self.graph, state = split((model))
        self.state = jax.tree.map(jnp.copy, state) if donate else state
        self._fn = _donated_call if donate else _call
        self._checked = False
        # The probe sets and input shapes compiled for, with the probes they hold.
        self._compiled: dict[tuple, tuple[Probe, ...]] = {}
        self._pack_steps, self._sharding = layout_of(self.state)

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    def _payload(self, name: str, value: tp.Any, per_step: bool = False) -> SparkPayload:
        """
            Converts ``value`` to the payload input ``name`` expects.

            Host values stay NumPy arrays inside the payload and move to the device with the
            arguments of the call.
        """
        if name not in self._formats:
            raise ValueError(f'"{name}" is not an input of the model. Inputs: {", ".join(self._formats) or "none"}.')
        payload_type, dtype, shape = self._formats[name]
        if isinstance(value, SparkPayload):
            if not isinstance(value, payload_type):
                raise ValueError(f'Input "{name}" takes a {payload_type.__name__}, got a {type(value).__name__}.')
            self._check_shape(name, jax.tree.leaves(value)[0].shape, shape, per_step)
            return value
        xp = jnp if isinstance(value, jax.Array) else np
        array = value if isinstance(value, jax.Array) else np.asarray(value)
        self._check_shape(name, array.shape, shape, per_step)
        if issubclass(payload_type, SpikeArray):
            # Spike bit, and the inhibition bit for negative entries, as `SpikeArray` encodes them.
            leaf = (array != 0).astype(xp.uint8) | ((array < 0).astype(xp.uint8) << 1)
        elif dtype is not None and array.dtype != dtype:
            leaf = array.astype(dtype)
        else:
            leaf = array
        return jax.tree.unflatten(self._structures[name], [leaf])

    @staticmethod
    def _check_shape(name: str, shape: tuple[int, ...], expected: tuple[int, ...] | None, per_step: bool) -> None:
        """
            Checks the shape of an input against the one the model was built with.

            A per-step input is checked without its leading axis of steps.
        """
        if per_step:
            if len(shape) == 0:
                raise ValueError(f'Input "{name}" is given per step and has no leading axis of steps.')
            shape = shape[1:]
        if expected is not None and tuple(shape) != tuple(expected):
            raise ValueError(
                f'Input "{name}" has shape {tuple(expected)}{" on every step" if per_step else ""}, as the model was built with; '
                f'got {tuple(shape)}.'
            )

    def _inputs(self, inputs: dict | None, per_step: dict | None, steps: int) -> tuple[dict, dict | None]:
        """
            Converts the inputs of a call to payloads, held and per step.

            Checks that every input is given once, and that per-step inputs have ``steps`` entries.
        """
        held ={k: self._payload(k, v, per_step=False) for k, v in (inputs or {}).items()}
        stacked = {k: self._payload(k, v, per_step=True) for k, v in (per_step or {}).items()} or None
        given = set(held) | set(stacked or ())
        missing = set(self._formats) - given
        if missing:
            raise ValueError(f'Missing inputs: {", ".join(sorted(missing))}.')
        repeated = set(held) & set(stacked or ())
        if repeated:
            raise ValueError(f'Inputs given both held and per step: {", ".join(sorted(repeated))}.')
        for leaf in jax.tree.leaves(stacked):
            if leaf.shape[0] != steps:
                raise ValueError(f'Per-step inputs have {leaf.shape[0]} entries along their first axis, for {steps} steps.')
        return held, stacked

    def _static(self, steps: int, probes: tuple) -> dict[str, tp.Any]:
        """
            Returns the static arguments of a call of ``steps`` steps recording ``probes``.
        """
        return dict(steps=steps, probes=probes, outputs=self.outputs, unroll=self.unroll, pack_steps=self._pack_steps, sharding=self._sharding)

    def _check_probes(self, held: dict, stacked: dict | None, steps: int) -> None:
        """
            Traces a call with the probes of every set of measurements, without compiling it.
        """
        probes = merge_probes(p for r in self.recorder.measurements for p in r.probes)
        if probes:
            call = functools.partial(_run_call, **self._static(steps, probes))
            try:
                jax.eval_shape(call, self.graph, self.state, held, stacked, start_of(probes, 0))
            except Exception as error:
                error.add_note('Raised while tracing every probe of the recorder, before the first call.')
                raise
        self._checked = True

    def run(self, steps: int, inputs: dict[str, tp.Any] | None = None, per_step: dict[str, tp.Any] | None = None) -> dict[str, SparkPayload] | None:
        """
            Advances the model ``steps`` steps, in one call of the compiled scan.

            Parameters
            ----------
            steps : int
                Steps of the call. Each distinct value compiles once per probe set.
            inputs : dict, optional
                Inputs held over the call, by input name. Arrays are converted to the payload type
                and dtype the model expects. For spikes, nonzero entries spike and negative ones are
                inhibitory.
            per_step : dict, optional
                Inputs given step by step, with a leading axis of length ``steps``.

            Returns
            -------
            dict of str to SparkPayload or None
                The outputs, as chosen by ``outputs``. They stay on the device until read.

            Raises
            ------
            ValueError
                When ``steps`` is not a positive integer, or when an input is unknown, missing,
                given both held and per step, or of another shape than the model was built with.

            Notes
            -----
            The call is traced and compiled first, where SIGINT stops it. A SIGINT during the call
            and the hand-over of its records is delivered once both are done. A second SIGINT while
            the recorder waits for room in its queue raises at once.
        """
        steps = integer(steps, 'steps', lowest=1)
        held, stacked = self._inputs(inputs, per_step, steps)
        if self.recorder is not None and not self._checked:
            self._check_probes(held, stacked, steps)
        probes = self.recorder.probes(steps) if self.recorder is not None else ()
        start = self.recorder.start(probes) if self.recorder is not None else start_of((), 0)
        static = self._static(steps, probes)
        try:
            # Traced and compiled first, where an interrupt stops it. The call then takes the state it donates
            # and hands back the new one, which the recorder counts; an interrupt meanwhile waits until both
            # are done, and a second one stops waiting for room in the queue of the writer.
            self._compile(held, stacked, start, static)
            with hold_interrupt(armed=False) as interrupt:
                outputs, self.state, records = self._fn(self.graph, self.state, held, stacked, start, **static)
                if self.recorder is not None:
                    if interrupt is not None:
                        interrupt.arm()
                    self.recorder.push(records, steps)
            if interrupt is not None:
                interrupt.deliver()
        except BaseException:
            if self.recorder is not None:
                self.recorder._pending = None
            raise
        return outputs

    def _compile(self, held: dict[str, SparkPayload], stacked: dict[str, SparkPayload], start: np.ndarray, static: dict[str, tp.Any]) -> tp.Any:
        """
            Compiles the call for a probe set and input shapes not compiled for yet.

            Returns the compiled call, or None when it was compiled before. The call then takes it
            from the cache of JAX.
        """
        leaves, tree = jax.tree.flatten((held, stacked))
        shapes = tuple((np.shape(leaf), getattr(leaf, 'dtype', type(leaf))) for leaf in leaves)
        key = (static['steps'], id(static['probes']), start.split, tree, shapes)
        if key in self._compiled:
            return None
        compiled = self._fn.lower(self.graph, self.state, held, stacked, start, **static).compile()
        self._compiled[key] = static['probes']
        return compiled

    __call__ = run

    def warmup(self, steps: int, inputs: dict[str, tp.Any] | None = None, per_step: dict[str, tp.Any] | None = None) -> int:
        """
            Compiles the calls of the run ahead of it.

            Compiles one call of ``steps`` steps for every probe set of `Recorder.warmup_sets`. The
            state is not advanced.

            Parameters
            ----------
            steps : int
                Steps of every call.
            inputs : dict, optional
                Example inputs held over the call, as for `run`.
            per_step : dict, optional
                Example inputs given step by step, as for `run`.

            Returns
            -------
            int
                Number of sets compiled.

            Notes
            -----
            Warns when a set needs more memory than the device has.
        """
        steps = integer(steps, 'steps', lowest=1)
        held, stacked = self._inputs(inputs, per_step, steps)
        if self.recorder is not None and not self._checked:
            self._check_probes(held, stacked, steps)
        sets = self.recorder.warmup_sets(steps) if self.recorder is not None else [()]
        step = self.recorder.step if self.recorder is not None else 0
        for probes in sets:
            static = self._static(steps, probes)
            # A call crossing the end of a group compiles apart, with the statistics and values of several groups.
            for start in warmup_starts(probes, step, steps):
                compiled = self._compile(held, stacked, start, static) or self._fn.lower(self.graph, self.state, held, stacked, start, **static).compile()
                check_memory(compiled, steps, probes)
        if self.recorder is not None:
            # What the recorder computes on the device when a group ends.
            shapes = lambda probes: jax.eval_shape(lambda state: read_boundary(merge(self.graph, state), probes), self.state)
            self.recorder._warm_groups(shapes, steps)
        return len(sets)

    def checkpoint(self, step: int | None = None) -> pathlib.Path:
        """
            Saves the current state to the run of the recorder, as `Recorder.checkpoint`.

            Parameters
            ----------
            step : int, optional
                Step the checkpoint is filed under. The current step of the recorder by default.

            Returns
            -------
            pathlib.Path
                File of the checkpoint.

            Raises
            ------
            ValueError
                When the runner has no recorder.
        """
        if self.recorder is None:
            raise ValueError('A runner without a recorder has no run to write checkpoints to.')
        return self.recorder.checkpoint(merge(self.graph, self.state), step=step)

    #-------------------------------------------------------------------------------------------------------------------------------------------#

    @property
    def model(self) -> Controller:
        """
            A copy of the model in its current state.
        """
        return merge(self.graph, jax.tree.map(jnp.copy, self.state))

    def close(self) -> None:
        """
            Closes the recorder, writing what is left. Does nothing without a recorder.
        """
        if self.recorder is not None:
            self.recorder.close()

    def __enter__(self) -> Runner:
        return self

    def __exit__(self, kind, error, traceback) -> None:
        if self.recorder is not None:
            self.recorder.__exit__(kind, error, traceback)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
