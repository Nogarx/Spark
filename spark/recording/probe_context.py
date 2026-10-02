#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
if tp.TYPE_CHECKING:
    from spark.nn.controllers.base import Controller

import jax
import contextvars
from spark.core.recording_hooks import PROBE_CONTEXT
from spark.recording.probe import Probe, resolve
from spark.recording.reduce import step_value

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class _Scoped:
    """
        What every probe context does for the controllers called within it.

        A probe context is used as a context manager, and is the only one open. It follows the scope
        of the modules called within it, and counts the calls of the model, the outermost
        controller. The values the controllers offer are ignored.

        Attributes
        ----------
        calls : int
            Calls of the model within the context.

        Raises
        ------
        RuntimeError
            When entered while another probe context is open. Probe contexts do not nest.
    """

    def __init__(self) -> None:
        self._scope: list[str] = []
        self._token: contextvars.Token | None = None
        self.calls = 0

    def __enter__(self) -> tp.Self:
        if PROBE_CONTEXT.get() is not None:
            raise RuntimeError('A probe context is already open. Probe contexts do not nest.')
        self._token = PROBE_CONTEXT.set(self)
        return self

    def __exit__(self, *exc_info) -> None:
        PROBE_CONTEXT.reset(self._token)
        self._token = None

    def push(self, name: str) -> None:
        """
            Enters the scope of a module.

            Parameters
            ----------
            name : str
                Name of the module within the current scope.
        """
        self._scope.append(name)

    def pop(self) -> None:
        """
            Leaves the scope of the current module.
        """
        self._scope.pop()

    def called(self, model: Controller) -> None:
        """
            Reports a call of a controller, before its modules run.

            Only the calls at the outermost scope, those of the model, are counted. Controllers
            within the model are called within the scope of their name.

            Parameters
            ----------
            model : Controller
                The controller called.
        """
        if self._scope:
            return
        self.calls += 1
        self._model_called(model)

    def _model_called(self, model: Controller) -> None:
        """
            Receives each call of the model, once counted.
        """

    def offer(self, values: dict[str, tp.Any], suffix: str | None = None) -> None:
        """
            Receives the values a controller offers at the current scope, and ignores them.
        """

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ProbeContext(_Scoped):
    """
        Probe context that captures the values used by a collection of probes, during JIT tracing.

        On model call, controllers expose their values to the open probe context: inputs and the outputs 
        of each of its modules, as well as the top most controller. Note that captured values are 
        JAX tracers, not numbers. 

        It is used as a context manager, and `recorded_scan` opens one around each step it traces.
        `get_probe_targets` opens one with ``capture_all`` to list the ports of a model.

        Parameters
        ----------
        probes : tuple of Probe
            Probes to capture, with distinct keys.
        capture_all : bool, default False
            Whether to keep every port offered, requested or not.

        Attributes
        ----------
        model : Controller or None
            The model called within the context, once called.
        calls : int
            Calls of the model within the context.

        Raises
        ------
        ValueError
            When two probes share a key.
        RuntimeError
            When entered while another probe context is open. Probe contexts do not nest.

        Notes
        -----
        The open probe context is held in a context variable, `spark.core.recording_hooks.PROBE_CONTEXT`.
    """

    def __init__(self, probes: tuple[Probe, ...], capture_all: bool = False) -> None:
        super().__init__()
        self.probes = probes
        self.capture_all = capture_all
        keys = [probe.key for probe in probes]
        if len(set(keys)) != len(keys):
            raise ValueError(f'Probes share the keys {sorted({k for k in keys if keys.count(k) > 1})}. Expected one probe per address and mode.')
        requests: dict[tuple[str, ...], set[str]] = {}
        for probe in probes:
            if probe.kind == 'port':
                requests.setdefault(probe.path, set()).add(probe.name)
        self._requests = {path: tuple(sorted(ports)) for path, ports in requests.items()}
        self._values: dict[tuple[tuple[str, ...], str], tp.Any] = {}
        self.model: Controller | None = None

    @property
    def captured(self) -> dict[tuple[tuple[str, ...], str], tp.Any]:
        """
            Values kept, by ``(path, port)``, in the order they were offered.
        """
        return self._values

    def _model_called(self, model: Controller) -> None:
        if self.model is None:
            self.model = model

    def offer(self, values: dict[str, tp.Any], suffix: str | None = None) -> None:
        """
            Keeps the requested entries of ``values``, produced at the current scope.

            With ``capture_all``, keeps every entry.

            Parameters
            ----------
            values : dict of str to SparkPayload
                Ports produced at the current scope.
            suffix : str, optional
                Appended to the scope. Controllers offer their inputs under ``__call__``.
        """
        path = tuple(self._scope) if suffix is None else (*self._scope, suffix)
        if self.capture_all:
            for port, value in values.items():
                self._values[(path, port)] = value
            return
        ports = self._requests.get(path)
        if ports is None:
            return
        for port in ports:
            if port in values:
                self._values[(path, port)] = values[port]

    def values(self, model: Controller | None = None) -> dict[str, jax.Array]:
        """
            Returns the values of the per-step probes for the step just taken.

            Parameters
            ----------
            model : Controller, optional
                The model called within the probe context. Attributes are read from it. Defaults to
                `model`.

            Returns
            -------
            dict of str to array
                Values by `Probe.key`, as `step_value` gives them.

            Raises
            ------
            RuntimeError
                When a requested port was not produced during the call.
            ValueError
                When an attribute is not found, or a value does not suit its probe (`step_value`).
            TypeError
                When an attribute holds no array.
        """
        model = self.model if model is None else model
        values = {}
        for probe in self.probes:
            if not probe.per_step:
                continue
            if probe.kind == 'port':
                value = self._values.get((probe.path, probe.name))
                if value is None:
                    raise RuntimeError(
                        f'"{probe.address}" was not produced during the call. Ports are captured from the modules '
                        f'of a controller called within the probe context.'
                    )
            else:
                value = getattr(resolve(model, probe.path, probe.address), probe.name)
            values[probe.key] = step_value(probe, value)
        return values

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _Reached(BaseException):
    """
        Signal that stops the tracing of a step function.

        A recorded scan reads some values of the model before its first step and after its last one. 
        While JAX traces the recorded call, `ModelCalls.run` calls the step function once more,
        on the carry of the scan, to get the model the step function builds from it. When the step
        function calls the model, `ModelCalls` reads the values from the model and raises this
        exception: the tracing of the step function stops there, before the step of the model is
        traced, and `run` returns the values. The compiled program holds the values read, and no
        step of the model.

        It derives from BaseException, not Exception: ``except Exception`` blocks in the step
        function do not catch it.

        Attributes
        ----------
        value : Any
            The values read from the model.
    """

    def __init__(self, value: tp.Any) -> None:
        super().__init__()
        self.value = value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ModelCalls(_Scoped):
    """
        Probe context that counts the calls of a model.

        Both uses happen while JAX traces a function:

        * Counting. The first time `spark.jit` traces a function while a recorder is open, it
          traces it within a `ModelCalls` to count the calls of the model outside `spark.scan`.
          The calls within the scan are not counted: no probe context is open there. A call
          outside the scan is an error, as the steps of a recorded call are the steps of its scan.
        * Reading. With ``read``, `run` calls a step function once more, on the carry of a scan.
          At the first call of the model, before any module runs, ``read`` takes values from the
          model and `_Reached` stops the tracing of the step function. A recorded scan reads the
          model before its first step and after its last one this way.

        Only the calls of the outermost model count. Controllers nested in it, such as the neurons
        of a brain, are not counted.

        Parameters
        ----------
        read : callable, optional
            Called with the model at its first call. What it returns is returned by `run`.

        Attributes
        ----------
        calls : int
            Calls of the model within the context.

        Raises
        ------
        RuntimeError
            When entered while another probe context is open.
    """

    def __init__(self, read: tp.Callable[[Controller], tp.Any] | None = None) -> None:
        super().__init__()
        self._read = read

    def _model_called(self, model: Controller) -> None:
        if self._read is not None:
            raise _Reached(self._read(model))

    def run(self, f: tp.Callable, *args: tp.Any) -> tp.Any:
        """
            Calls ``f`` up to its first call of the model, and returns what ``read`` read there.

            Raises
            ------
            RuntimeError
                When ``f`` returns without calling the model.
        """
        try:
            with self:
                f(*args)
        except _Reached as reached:
            return reached.value
        raise RuntimeError(
            'The step function of `spark.scan` returned without calling the model. Each step of a recorded scan is one '
            'call of the model.'
        )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
