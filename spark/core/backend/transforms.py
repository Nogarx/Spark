#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import types
import inspect
import functools
import contextvars

import jax
import flax.nnx as nnx
A = tp.TypeVar('A')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

OPEN_RECORDERS: dict = {}
"""
    The open recorders of `spark.recording`, in the order they were opened, with the thread that opened each.
"""

TRACED_CALL: contextvars.ContextVar[tp.Any] = contextvars.ContextVar('spark_traced_call', default=None)
"""
    The call of a `jit` function traced for a recorder, or None.
"""

class RecordingHooks(tp.Protocol):
    """
        What `spark.recording` provides to `jit` and `scan` while a recorder is open.
    """

    def call(self, function: Jit, args: tuple, kwargs: dict) -> tp.Any:
        ...

    def scan(self, f: tp.Callable, init: tp.Any, xs: tp.Any, length: int | None, reverse: bool, unroll: int | bool, split_transpose: bool) -> tp.Any:
        ...

    def warmup(self, function: Jit, args: tuple, kwargs: dict) -> int:
        ...

_hooks: RecordingHooks | None = None

def set_recording_hooks(hooks: RecordingHooks) -> None:
    """
        Installs the hooks of `spark.recording`. Called once, when it is imported.
    """
    global _hooks
    _hooks = hooks

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _as_tuple(value: tp.Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (int, str)):
        return (value,)
    return tuple(value)

def _modules_in(args: tuple, kwargs: dict) -> bool:
    """
        Returns whether a module is among the arguments of a call.
    """
    for value in args:
        if isinstance(value, nnx.Module):
            return True
    for value in kwargs.values():
        if isinstance(value, nnx.Module):
            return True
    return False

def _static_arguments(fun: tp.Callable, options: dict[str, tp.Any]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """
        Returns the positions and names of the static arguments of ``fun``, as ``jax.jit`` infers
        them.

        Given only positions, the names are inferred from the signature of ``fun``, and given only
        names, the positions.
    """
    numbers, names = _as_tuple(options.get('static_argnums')), _as_tuple(options.get('static_argnames'))
    try:
        parameters = list(inspect.signature(fun).parameters.values())
    except (TypeError, ValueError):
        return numbers, names
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if numbers and not names:
        names = tuple(
            positional[n].name for n in numbers
            if -len(positional) <= n < len(positional) and positional[n].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
    elif names and not numbers:
        numbers = tuple(i for i, p in enumerate(positional) if p.name in names)
    return numbers, names

class Jit:
    """
        A function compiled with ``jax.jit``, whose calls an open recorder records.

        `jit` creates it. Without an open recorder, a call is a call of ``jax.jit(fun, **options)``.

        While a recorder of `spark.recording` is open, each call of the function is a call of the
        recorder. Before the call, the recorder decides what to record over its steps, and `scan`
        records it along the way. After the call, the records are handed over to the recorder
        (`Recorder.push`). A call recording nothing runs the program compiled without a recorder.

        The steps of a call are those of the `scan` it runs. They are found by tracing the function
        once for each value of its static arguments. A call running no `scan` is not a call of the
        recorder.

        Called with a module among its arguments, the function is compiled with ``flax.nnx.jit``
        instead, which updates the module in place. Such calls are not recorded: recorded calls take
        the graph and the state of the model, as `spark.split` gives them.

        Parameters
        ----------
        fun : callable
            The function to compile.
        **options
            Passed to ``jax.jit``, such as ``static_argnames`` or ``donate_argnames``.

        Attributes
        ----------
        fun : callable
            The function compiled.
        jitted : callable
            ``jax.jit(fun, **options)``. Attributes not found on the `Jit`, such as ``lower`` or
            ``trace``, are read from it.

        Notes
        -----
        While a recorder is open:

        * The static arguments of a call give its steps, unless a `scan` of the function takes its
          length from ``xs``. The shapes of the arguments are then part of the key too.
        * The model is called within `scan` only. A call of the model outside it raises.
        * A call runs one `scan`, directly: not within another `scan`, ``jax.lax.scan``,
          ``jax.vmap``, ``jax.lax.cond`` or a function compiled apart. A `Jit` called within
          another is traced as part of it.
        * The first call of a function traces it with every probe of the recorder, without
          compiling it. A probe that cannot record the model raises there.
        * An interrupt (Ctrl-C) during a call is held until the call returns its result, and raised
          by the next call of the recorder from the loop, before it does anything. The recorder then
          counts the calls whose result the loop received. A second interrupt raises at once. A
          call is compiled before the interrupt is held.

        See Also
        --------
        jit : Creates a `Jit`.
        scan : ``jax.lax.scan``, recorded within a `Jit` while a recorder is open.
        spark.recording.Recorder : Decides what each call records and writes it to a run.
    """

    def __init__(self, fun: tp.Callable, **options: tp.Any) -> None:
        functools.update_wrapper(self, fun)
        self.fun = fun
        self.options = options
        self.jitted = jax.jit(fun, **options)
        self.static_argnums, self.static_argnames = _static_arguments(fun, options)
        # What `spark.recording` keeps about the calls of the function: its steps and recorded variant.
        self._recording: tp.Any = None
        # ``flax.nnx.jit`` of the function, for calls with modules among their arguments.
        self._nnx: tp.Any = None

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        if _modules_in(args, kwargs):
            if self._nnx is None:
                self._nnx = nnx.jit(self.fun, **self.options)
            return self._nnx(*args, **kwargs)
        if not OPEN_RECORDERS:
            return self.jitted(*args, **kwargs)
        return _hooks.call(self, args, kwargs)

    def warmup(self, *args: tp.Any, **kwargs: tp.Any) -> int:
        """
            Compiles ahead the calls the open recorder is likely to record.

            Takes the arguments of a call. Compiles it for every probe set of
            `Recorder.warmup_sets`, with the steps of the call. Nothing is run.

            Returns
            -------
            int
                Number of probe sets compiled.

            Raises
            ------
            RuntimeError
                When no recorder is open, or the thread opened none while several are open.
            ValueError
                When the call runs no `scan`.

            Notes
            -----
            Warns when a set needs more memory than the device has.
        """
        if not OPEN_RECORDERS:
            raise RuntimeError('`warmup` compiles the calls a recorder records; no recorder is open.')
        return _hooks.warmup(self, args, kwargs)

    def __get__(self, instance: tp.Any, owner: type | None = None) -> tp.Any:
        if instance is None:
            return self
        return types.MethodType(self, instance)

    def __getattr__(self, name: str) -> tp.Any:
        if name == 'jitted':
            raise AttributeError(name)
        return getattr(self.jitted, name)

    def __repr__(self) -> str:
        return f'Jit({getattr(self.fun, "__qualname__", self.fun)!r})'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def jit(fun: tp.Callable | None = None, /, **options: tp.Any) -> Jit | tp.Callable[[tp.Callable], Jit]:
    """
        Compiles a function with ``jax.jit``. Its calls are recorded while a recorder is open.

        Parameters
        ----------
        fun : callable, optional
            The function to compile. Without it, returns a decorator taking ``options``.
        **options
            Passed to ``jax.jit``, such as ``static_argnames`` or ``donate_argnames``.

        Returns
        -------
        Jit
            The compiled function. Without an open recorder, it is ``jax.jit(fun, **options)``, or
            ``flax.nnx.jit(fun, **options)`` for calls with modules among their arguments.

        See Also
        --------
        Jit : How a call is recorded.
        scan : ``jax.lax.scan``, recorded within a `Jit` while a recorder is open.

        Examples
        --------
        >>> @partial(spark.jit, static_argnames=['steps'])
        ... def run(graph, state, steps, **inputs):
        ...     def step(state, _):
        ...         model = spark.merge(graph, state)
        ...         outputs = model(**inputs)
        ...         return spark.split(model)[1], outputs
        ...     return spark.scan(step, state, length=steps)
    """
    if fun is None:
        return functools.partial(jit, **options)
    return Jit(fun, **options)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def scan(
        f: tp.Callable,
        init: tp.Any,
        xs: tp.Any = None,
        length: int | None = None,
        reverse: bool = False,
        unroll: int | bool = 1,
        _split_transpose: bool = False,
    ) -> tuple[tp.Any, tp.Any]:
    """
        ``jax.lax.scan``, recorded within a call of a `Jit` while a recorder is open.

        Otherwise, it is ``jax.lax.scan``, and traces to the same program.

        Each step of the scan is one step of the model: ``f`` calls the model once. Within a
        recorded call, the probes of the call are recorded on every step, and the records are
        returned by the `Jit` to the recorder. What ``f`` returns is unchanged.

        Parameters
        ----------
        f, init, xs, length, reverse, unroll, _split_transpose
            As for ``jax.lax.scan``.

        Returns
        -------
        carry, ys
            As ``jax.lax.scan`` returns them.

        Raises
        ------
        RuntimeError
            Within a recorded call, when ``f`` does not call the model once per step, or when the
            scan runs within another transformation.
        ValueError
            Within a recorded call, with ``reverse``.

        See Also
        --------
        jit : Compiles a function whose calls a recorder records.
    """
    if TRACED_CALL.get() is None:
        return jax.lax.scan(f, init, xs, length, reverse, unroll, _split_transpose)
    return _hooks.scan(f, init, xs, length, reverse, unroll, _split_transpose)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def eval_shape(*args, **kwargs) -> A:
    """
        Wrapper around flax.nnx.eval_shape, to simplify imports.
    """
    return nnx.eval_shape(*args, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def grad(*args, **kwargs) -> (tp.Callable[..., tp.Any] | tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]] ):
    """
        Wrapper around flax.nnx.grad, to simplify imports.
    """
    return nnx.grad(*args, **kwargs)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
