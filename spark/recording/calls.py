#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
if tp.TYPE_CHECKING:
    from spark.recording.recorder import Recorder

import inspect
import weakref
import warnings
import functools
import contextlib

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from spark.core.backend.transforms import Jit
from spark.core.recording_hooks import TRACED_CALL, PROBE_CONTEXT, set_recording_hooks
from spark.recording.probe import Probe
from spark.recording.probe_context import ModelCalls
from spark.recording.measurements import merge_probes
from spark.recording.scan import recorded_scan, _first
from spark.recording.current import open_recorder
from spark.recording.reduce import start_of, warmup_starts, read_boundary, Start, Packed
from spark.recording.utils import hold_interrupt

# NOTE: `spark.scan` needs ``_trace_ctx`` within another transformation or JAX will report a leaked tracer.
try:
    from jax._src.core import trace_ctx as _trace_ctx
except ImportError:
    _trace_ctx = None

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

_EXTRA = ('_spark_probes', '_spark_start', '_spark_learned')
"""
    The arguments `_recorded_variant` adds to a function.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _qualname(function: Jit) -> str:
    """
        Returns the qualified name of the function ``function`` compiles, for error messages.
    """
    return getattr(function.fun, '__qualname__', str(function.fun))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _current_trace() -> tp.Any:
    """
        Returns the current JAX trace context.
    """
    return None if _trace_ctx is None else _trace_ctx.trace

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def replicated(sharding: jax.sharding.Sharding) -> NamedSharding:
    """
        Returns a sharding that replicates a value on the devices of ``sharding``.

        Use to gracefully expose fragmented arrays (multi-host) to the recorded.
    """
    mesh = getattr(sharding, 'mesh', None)
    if isinstance(mesh, Mesh):
        return NamedSharding(mesh, PartitionSpec())
    return NamedSharding(Mesh(np.array(sharding._device_assignment), ('devices',)), PartitionSpec())

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def check_memory(compiled: tp.Any, steps: int, probes: tuple[Probe, ...]) -> None:
    """
        Memory check that warns when a compiled call needs more memory than available.
    """
    try:
        analysis = compiled.memory_analysis()
        limit = int(jax.local_devices()[0].memory_stats()['bytes_limit'])
        needed = (
            analysis.argument_size_in_bytes + analysis.output_size_in_bytes + analysis.temp_size_in_bytes
            - analysis.alias_size_in_bytes
        )
    except Exception:
        return
    if needed > limit:
        warnings.warn(
            f'A call of {steps} steps recording {len(probes)} probes requires ~{needed / 2 ** 30:.2f} GiB of device memory.'
            f'However the device has only has {limit / 2 ** 30:.2f} GiB. Probes: {", ".join(p.key for p in probes)}.'
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@contextlib.contextmanager
def _outside_calls() -> tp.Generator[None, None, None]:
    """
        Runs the block as if no call were traced and no probe context open.
    """
    call, context = TRACED_CALL.set(None), PROBE_CONTEXT.set(None)
    try:
        yield
    finally:
        PROBE_CONTEXT.reset(context)
        TRACED_CALL.reset(call)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Layout(tp.NamedTuple):
    """
        How the records of the calls of a function are laid out, from where its arguments are.

        Attributes
        ----------
        pack_steps : bool
            Whether the values of a step are packed into one row. Not when the state is sharded
            across devices.
        sharding : NamedSharding or None
            With several processes, where the records are replicated.
    """
    pack_steps: bool = True
    sharding: NamedSharding | None = None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Learned(tp.NamedTuple):
    """
        What the first trace of a function found for one value of its static arguments.

        Attributes
        ----------
        steps : int
            Steps of a call. 0 when the call runs no `spark.scan`.
        layout : Layout
            How the records are laid out.
    """
    steps: int
    layout: Layout = Layout()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Call:
    """
        A call of a `Jit` traced for a recorder.

        Parameters
        ----------
        mode : {'count', 'record', 'boundary'}
            ``'count'`` counts the steps of the `spark.scan` it runs. ``'record'`` records
            ``probes`` on them, from ``start``. ``'boundary'`` reads the values of ``probes`` before
            the steps.
        probes : tuple of Probe
            Probes of the call.
        start : Start, optional
            Where the call starts on the steps of the run.
        learned : Learned, optional
            The steps and layout of the call.
    """

    def __init__(self, mode: str, probes: tuple[Probe, ...] = (), start: Start | None = None, learned: Learned | None = None) -> None:
        self.mode = mode
        self.probes = probes
        self.start = start
        self.learned = learned
        self.counted = 0
        self.scans = 0
        self.shaped = False
        self.records: Packed | dict | None = None
        self.boundary: dict[str, jax.Array] | None = None
        self.trace: tp.Any = None

    def run(self, fun: tp.Callable, args: tp.Sequence, kwargs: dict) -> tp.Any:
        """
            Calls ``fun`` with this call traced.
        """
        self.trace = _current_trace()
        token = TRACED_CALL.set(self)
        try:
            result = fun(*args, **kwargs)
        finally:
            TRACED_CALL.reset(token)
        if self.mode == 'record' and self.records is None:
            raise RuntimeError(
                'The call ran no `spark.scan`, while the first trace of the function found one for the same static '
                'arguments.'
            )
        return result

    def check_trace(self, what: str) -> None:
        """
            Raises when ``what`` runs within another transformation than the traced call.
        """
        if self.trace is not None and _current_trace() is not self.trace:
            raise RuntimeError(
                f'{what} runs within another transformation of the call, such as a scan, `jax.vmap`, `jax.lax.cond` or '
                f'a function compiled apart. A recorded call runs its `spark.scan` directly.'
            )

    def add_scan(self, steps: int, what: str) -> None:
        """
            Counts a scan of ``steps`` steps. A call runs one.
        """
        self.scans += 1
        if self.scans > 1:
            raise RuntimeError(
                f'{what} is the second `spark.scan` of the call. While a recorder is open, a call of a `spark.jit` '
                f'function runs one `spark.scan`, whose steps are the steps of the call.'
            )
        self.counted += steps

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Calls:
    """
        What is kept about the calls of one `Jit`.

        Attributes
        ----------
        learned : dict
            `Learned`, by the static arguments of a call, and by the shapes of the others when
            `shaped`.
        shaped : bool
            Whether a `spark.scan` of the function takes its length from ``xs``.
        recorded : callable
            The function compiled with the probes of the call as a static argument, and the records
            returned beside its result.
        checked : WeakSet of Recorder
            The recorders whose probes were traced with the function.
        compiled : set of tuple
            The calls compiled, by key, probes and whether they may cross the end of a group.
    """

    def __init__(self, function: Jit) -> None:
        self.function = function
        self.learned: dict[tp.Hashable, Learned] = {}
        self.shaped = False
        self.recorded = _recorded_variant(function)
        self.checked: weakref.WeakSet[Recorder] = weakref.WeakSet()
        self.compiled: set[tuple] = set()

    def static(self, args: tuple, kwargs: dict) -> tp.Hashable:
        """
            Returns the static arguments of a call, as a key.
        """
        count = len(args)
        return (
            tuple(args[i % count] for i in self.function.static_argnums if -count <= i < count),
            tuple((name, kwargs[name]) for name in self.function.static_argnames if name in kwargs),
        )

    def arguments(self, args: tuple, kwargs: dict) -> tuple[tp.Hashable, tuple[list, dict], tp.Callable[[list, dict], tuple[list, dict]]]:
        """
            Splits the arguments of a call into its static and dynamic ones.

            Returns the static ones as a key, the dynamic ones, and a function rebuilding the
            arguments from the dynamic ones.
        """
        count = len(args)
        numbers = {i % count for i in self.function.static_argnums if -count <= i < count}
        names = set(self.function.static_argnames)
        static = self.static(args, kwargs)
        dynamic = ([a for i, a in enumerate(args) if i not in numbers], {k: v for k, v in kwargs.items() if k not in names})

        def rebuild(dynamic_args: list, dynamic_kwargs: dict) -> tuple[list, dict]:
            remaining = iter(dynamic_args)
            full = [args[i] if i in numbers else next(remaining) for i in range(count)]
            return full, {**{k: v for k, v in kwargs.items() if k in names}, **dynamic_kwargs}

        return static, dynamic, rebuild

    def key(self, static: tp.Hashable, dynamic: tuple[list, dict]) -> tp.Hashable:
        if not self.shaped:
            return static
        leaves, tree = jax.tree.flatten(dynamic)
        return static, tree, tuple(np.shape(leaf) for leaf in leaves)

    def learn(self, args: tuple, kwargs: dict) -> tuple[tp.Hashable, Learned]:
        """
            Returns the key of a call and its steps and layout, tracing the function when they are not
            known.

            Raises
            ------
            RuntimeError
                When the function calls the model outside `spark.scan`.
        """
        if not self.shaped:
            key = self.static(args, kwargs)
            learned = self.learned.get(key)
            if learned is not None:
                return key, learned
        static, dynamic, rebuild = self.arguments(args, kwargs)
        key = self.key(static, dynamic)
        learned = self.learned.get(key)
        if learned is not None:
            return key, learned
        call, calls = _Call('count'), ModelCalls()

        def counted(dynamic: tuple[list, dict]) -> None:
            with calls:
                call.run(self.function.fun, *rebuild(*dynamic))

        with _outside_calls():
            jax.eval_shape(counted, dynamic)
        if calls.calls:
            raise RuntimeError(
                f'`{_qualname(self.function)}` calls the model outside `spark.scan`. While a '
                f'recorder is open, the steps of a call of a `spark.jit` function are the steps of its `spark.scan`: call the '
                f'model within it, or compile the function with `jax.jit`.'
            )
        self.shaped = self.shaped or call.shaped
        learned = Learned(call.counted, layout_of(dynamic) if call.counted else Layout())
        key = self.key(static, dynamic)
        self.learned[key] = learned
        return key, learned

    def compile(self, key: tp.Hashable, args: tuple, kwargs: dict, probes: tuple[Probe, ...], start: Start | None, learned: Learned) -> tp.Any:
        """
            Compiles a call not compiled yet, and returns the compiled call, or None.

            The call then takes it from the cache of JAX. `_Hooks.call` compiles before it holds
            SIGINT, so that an interrupt stops a long compilation.
        """
        compiled = (key, probes, None if start is None else start.split)
        if compiled in self.compiled:
            return None
        if probes:
            lowered = self.recorded.lower(*args, _spark_probes=probes, _spark_start=start, _spark_learned=learned, **kwargs)
        else:
            lowered = self.function.jitted.lower(*args, **kwargs)
        executable = lowered.compile()
        self.compiled.add(compiled)
        return executable

    def trace(self, args: tuple, kwargs: dict, call: _Call) -> tp.Any:
        """
            Traces a call of the function with ``call``, without compiling it.

            Returns the shapes of the values `spark.scan` read before the steps, for a
            ``'boundary'`` call, or None.
        """
        static, dynamic, rebuild = self.arguments(args, kwargs)

        def traced(dynamic: tuple[list, dict], start: Start | None) -> dict | None:
            call.start = start
            call.run(self.function.fun, *rebuild(*dynamic))
            return call.boundary

        with _outside_calls():
            return jax.eval_shape(traced, dynamic, call.start)

    def check(self, recorder: Recorder, args: tuple, kwargs: dict, learned: Learned) -> None:
        """
            Traces the function with every probe of ``recorder``, once per recorder.
        """
        if recorder in self.checked:
            return
        probes = merge_probes(p for r in recorder.measurements for p in r.probes)
        if probes:
            try:
                self.trace(args, kwargs, _Call('record', probes, start_of(probes, 0), learned))
            except Exception as error:
                error.add_note('Raised while tracing every probe of the recorder, before the first call.')
                raise
        self.checked.add(recorder)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def layout_of(dynamic: tp.Any) -> Layout:
    """
        Returns the layout of the records of a call with the dynamic arguments ``dynamic``, such as
        the state of the model.
    """
    shardings = [
        leaf.sharding for leaf in jax.tree.leaves(dynamic)
        if isinstance(leaf, jax.Array) and not isinstance(leaf, jax.core.Tracer)
    ]
    widest = max(shardings, key=lambda sharding: len(sharding.device_set), default=None)
    if widest is None or len(widest.device_set) <= 1:
        return Layout()
    return Layout(pack_steps=False, sharding=replicated(widest) if jax.process_count() > 1 else None)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _recorded_variant(function: Jit) -> tp.Callable:
    """
        Compiles ``function`` for recorded calls.

        The variant takes the probes of the call, where it starts and what its first trace found as
        keyword arguments, the probes and the steps static. It returns the records beside the
        result.
    """
    fun = function.fun

    def recorded(*args: tp.Any, _spark_probes: tuple[Probe, ...], _spark_start: Start, _spark_learned: Learned, **kwargs: tp.Any) -> tuple:
        call = _Call('record', _spark_probes, _spark_start, _spark_learned)
        result = call.run(fun, args, kwargs)
        records = call.records
        if _spark_learned.layout.sharding is not None and records:
            records = jax.lax.with_sharding_constraint(records, _spark_learned.layout.sharding)
        return result, records

    functools.update_wrapper(recorded, fun)
    try:
        signature = inspect.signature(fun)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        # The same parameters, for the static and donated arguments of JAX, and the three keyword-only ones.
        parameters = list(signature.parameters.values())
        at = next((i for i, p in enumerate(parameters) if p.kind == p.VAR_KEYWORD), len(parameters))
        extra = [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in _EXTRA]
        recorded.__signature__ = signature.replace(parameters=[*parameters[:at], *extra, *parameters[at:]])
    options = {**function.options, 'static_argnums': function.static_argnums}
    options['static_argnames'] = (*function.static_argnames, '_spark_probes', '_spark_learned')
    return jax.jit(recorded, **options)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _calls(function: Jit) -> _Calls:
    calls = function._recording
    if calls is None:
        calls = function._recording = _Calls(function)
    return calls

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Hooks:
    """
        Recorder hooks for `spark.jit` and `spark.scan`.
    """

    def call(self, function: Jit, args: tuple, kwargs: dict) -> tp.Any:
        """
            Runs a call of ``function`` for the open recorder.
        """
        calls = _calls(function)
        outer = TRACED_CALL.get()
        if outer is not None:
            return self._within(calls, outer, args, kwargs)
        recorder = open_recorder()
        if recorder is None:
            return function.jitted(*args, **kwargs)
        key, learned = calls.learn(args, kwargs)
        if not learned.steps:
            return function.jitted(*args, **kwargs)
        calls.check(recorder, args, kwargs, learned)
        probes = recorder.probes(learned.steps)
        try:
            start = recorder.start(probes) if probes else None
            calls.compile(key, args, kwargs, probes, start, learned)
            # From the call to the hand-over of its records, SIGINT waits: the call counts for the recorder once
            # the loop receives its result. It is raised at the next call of the recorder.
            with hold_interrupt() as held:
                if probes:
                    result, records = calls.recorded(*args, _spark_probes=probes, _spark_start=start, _spark_learned=learned, **kwargs)
                else:
                    result, records = function.jitted(*args, **kwargs), {}
                recorder.push(records, learned.steps)
        except BaseException:
            # A call not handed over did not run for the recorder: the next one is decided again.
            recorder._pending = None
            raise
        if held is not None and held.frame is not None:
            recorder._hold_interrupt(held)
        return result

    def _within(self, calls: _Calls, outer: _Call, args: tuple, kwargs: dict) -> tp.Any:
        """
            Runs a call of a `Jit` traced within the call ``outer``, as a part of it.
        """
        _, learned = calls.learn(args, kwargs)
        if not learned.steps or outer.mode == 'boundary':
            with _outside_calls():
                return calls.function.jitted(*args, **kwargs)
        what = f'The call of `{_qualname(calls.function)}`'
        outer.check_trace(what)
        outer.add_scan(learned.steps, what)
        if outer.mode == 'count':
            outer.shaped = outer.shaped or calls.shaped
            with _outside_calls():
                return calls.function.jitted(*args, **kwargs)
        extra = {'_spark_probes': outer.probes, '_spark_start': outer.start, '_spark_learned': Learned(learned.steps, outer.learned.layout)}
        with _outside_calls():
            result, outer.records = calls.recorded(*args, **kwargs, **extra)
        return result

    def scan(self, f: tp.Callable, init: tp.Any, xs: tp.Any, length: int | None, reverse: bool, unroll: int | bool, split_transpose: bool) -> tp.Any:
        """
            Runs a `spark.scan` of the traced call.
        """
        call = TRACED_CALL.get()
        call.check_trace('`spark.scan`')
        if reverse:
            raise ValueError('A recorded `spark.scan` runs the steps of the model in order; `reverse=True` is not recorded.')
        if length is None:
            leaves = jax.tree.leaves(xs)
            if not leaves:
                raise ValueError('`spark.scan` takes `length` when `xs` is None.')
            steps = int(leaves[0].shape[0])
        else:
            steps = int(length)
        call.add_scan(steps, '`spark.scan`')
        if call.mode == 'count':
            call.shaped = call.shaped or length is None
            # The calls of the model within the scan are its steps, and are not counted apart.
            token = PROBE_CONTEXT.set(None)
            try:
                return jax.lax.scan(f, init, xs, length, reverse, unroll, split_transpose)
            finally:
                PROBE_CONTEXT.reset(token)
        if call.mode == 'boundary':
            call.boundary = ModelCalls(lambda model: read_boundary(model, call.probes)).run(f, init, _first(xs))
            return jax.lax.scan(f, init, xs, length, reverse, unroll, split_transpose)
        if steps != call.learned.steps:
            raise RuntimeError(
                f'This call runs {steps} steps, and the first call with the same static arguments ran {call.learned.steps}. '
                f'While a recorder is open, the static arguments of a `spark.jit` function give the steps of its calls.'
            )
        carry, ys, call.records = recorded_scan(
            f, init, xs, steps=steps, probes=call.probes, start=call.start, unroll=unroll,
            pack_steps=call.learned.layout.pack_steps, split_transpose=split_transpose,
        )
        return carry, ys

    def warmup(self, function: Jit, args: tuple, kwargs: dict) -> int:
        """
            Compiles ahead the calls of ``function`` the open recorder is likely to record.
        """
        recorder = open_recorder()
        if recorder is None:
            raise RuntimeError('`warmup` compiles the calls a recorder records; no recorder is open.')
        calls = _calls(function)
        key, learned = calls.learn(args, kwargs)
        if not learned.steps:
            raise ValueError(
                f'`{_qualname(function)}` runs no `spark.scan`: its calls are not calls of the '
                f'recorder.'
            )
        calls.check(recorder, args, kwargs, learned)
        steps, sets = learned.steps, recorder.warmup_sets(learned.steps)
        for probes in sets:
            # A call crossing the end of a group compiles apart, with the statistics and values of several groups.
            for start in warmup_starts(probes, recorder.step, steps) if probes else (None,):
                compiled = calls.compile(key, args, kwargs, probes, start, learned)
                if compiled is not None:
                    check_memory(compiled, steps, probes)
        # What the recorder computes on the device when a group ends.
        shapes = lambda probes: calls.trace(args, kwargs, _Call('boundary', probes, start_of(probes, 0), learned))
        recorder._warm_groups(shapes, steps)
        return len(sets)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

set_recording_hooks(_Hooks())

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
