#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import zlib
import numpy as np
from math import prod
from spark.core.module import SparkModule
from spark.core.payloads import SpikeArray
from spark.nn.controllers.base import Controller
from spark.nn.components.somas.base import Soma
from spark.nn.components.synapses.base import Synapses
from spark.nn.interfaces.input.base import InputInterface
from spark.nn.interfaces.control.base import ControlInterface
from spark.nn.interfaces.output.base import OutputInterface
from spark.recording.probe import (
    Probe, SummaryProbe, TraceProbe, RasterProbe, SnapshotProbe, 
    DeltaProbe, SummaryReduction, DeltaReduction, CALL, address,
)
from spark.recording.measurements import Measurements
from spark.recording.triggers import Every, Always

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

_KINDS: dict[str, type] = {
    'soma': Soma, 'synapses': Synapses, 'input': InputInterface, 'control': ControlInterface, 'output': OutputInterface,
}
"""
    The base classes of somas, synapses and interfaces, by kind.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _modules(model: SparkModule | Controller, path: tuple[str, ...] = ()) -> tp.Iterator[tuple[tuple[str, ...], tp.Any]]:
    """
        Yields ``(path, module)`` for every module, including the model itself.
    """
    yield path, model
    if isinstance(model, Controller):
        for name in model._modules_names:
            yield from _modules(getattr(model, name), (*path, name))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _output_ports(module: SparkModule | Controller, spikes: bool) -> list[tuple[str, int]]:
    """
        Lists ``(port, size)`` for the outputs of a module.
        `spikes = True` returns only `SpikeArray` ports and viceversa. 
    """
    ports = []
    for port, spec in module.get_output_specs().items():
        if spec.payload_type is None or spec.shape is None:
            continue
        if issubclass(spec.payload_type, SpikeArray) == spikes:
            ports.append((port, prod(spec.shape)))
    return ports

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def sample_units(address: str, size: int, count: int) -> tuple[int, ...] | None:
    """
        Draws ``count`` units out of ``size`` for the probe at ``address``.

        The draw is seeded by ``address``, and is the same in every process.

        Parameters
        ----------
        address : str
            Probe address.
        size : int
            Number of units of the value.
        count : int
            Number of units to draw.

        Returns
        -------
        tuple of int or None
            Sorted flat indices, or None (all units) when ``size`` is not larger than ``count``.
    """
    if size <= count:
        return None
    rng = np.random.default_rng(zlib.crc32(address.encode()))
    return tuple(int(u) for u in np.sort(rng.choice(size, size=count, replace=False)))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _of_kind(model: tp.Any, kind: str) -> list[tuple[tuple[str, ...], tp.Any]]:
    """
        Lists ``(path, module)`` for the modules of the model of one kind of `_KINDS`.
    """
    cls = _KINDS[kind]
    return [(path, module) for path, module in _modules(model) if isinstance(module, cls)]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def summary(model: Controller) -> tuple[Probe, ...]:
    """
        Returns probes of the population statistics of every soma and set of synapses.

        * Every spike output of a soma: ``ACTIVE_FRACTION`` and ``INACTIVE_UNIT_FRACTION``.
        * The membrane potential of a soma: ``MEAN``, ``STD``, ``MIN`` and ``MAX``.
        * The kernel of a set of synapses: the ``NORM`` and ``MEAN_ABS`` of its change over each
          group.

        Parameters
        ----------
        model : Controller
            A built model.

        Returns
        -------
        tuple of Probe
            `SummaryProbe` and `DeltaProbe` probes. The measurements holding them need a group.

        See Also
        --------
        activity : Traces and rasters of a model on every step.
        weights : Snapshots of the weights of a model.
    """
    probes = []
    for path, soma in _of_kind(model, 'soma'):
        for port, _ in _output_ports(soma, spikes=True):
            probes.append(SummaryProbe(
                address(path, port, 'port'),
                reduce=(SummaryReduction.ACTIVE_FRACTION, SummaryReduction.INACTIVE_UNIT_FRACTION),
            ))
        probes.append(SummaryProbe(
            address(path, 'potential', 'attribute'),
            reduce=(SummaryReduction.MEAN, SummaryReduction.STD, SummaryReduction.MIN, SummaryReduction.MAX),
        ))
    for path, _ in _of_kind(model, 'synapses'):
        probes.append(DeltaProbe(
            address(path, 'kernel', 'attribute'), reduce=(DeltaReduction.NORM, DeltaReduction.MEAN_ABS),
        ))
    return tuple(probes)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def activity(model: Controller, trace_units: int = 64, raster_units: int = 4096) -> tuple[Probe, ...]:
    """
        Returns probes tracing the inputs, interfaces and somas of a model on every step.

        * The inputs of the model: a trace of each, whole.
        * Input interfaces: a raster of each spike output.
        * Control and output interfaces: a trace of each output that does not carry spikes.
        * Somas: a raster of each spike output, and a trace of the membrane potential.

        Parameters
        ----------
        model : Controller
            A built model.
        trace_units : int, default 64
            Units traced per soma or interface output, drawn by `sample_units`.
        raster_units : int, default 4096
            Units kept per raster, drawn by `sample_units`.

        Returns
        -------
        tuple of Probe
            `TraceProbe` and `RasterProbe` probes.

        See Also
        --------
        summary : Population statistics of every soma and set of synapses.
        weights : Snapshots of the weights of a model.
    """
    probes = [TraceProbe(address((CALL,), name, 'port')) for name in model.get_controller_inputs()]
    for path, interface in _of_kind(model, 'input'):
        for port, size in _output_ports(interface, spikes=True):
            name = address(path, port, 'port')
            probes.append(RasterProbe(name, units=sample_units(name, size, raster_units)))
    for kind in ('control', 'output'):
        for path, interface in _of_kind(model, kind):
            for port, size in _output_ports(interface, spikes=False):
                name = address(path, port, 'port')
                probes.append(TraceProbe(name, units=sample_units(name, size, trace_units)))
    for path, soma in _of_kind(model, 'soma'):
        for port, size in _output_ports(soma, spikes=True):
            name = address(path, port, 'port')
            probes.append(RasterProbe(name, units=sample_units(name, size, raster_units)))
        name = address(path, 'potential', 'attribute')
        probes.append(TraceProbe(name, units=sample_units(name, soma.potential.value.size, trace_units)))
    return tuple(probes)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def weights(model: Controller) -> tuple[Probe, ...]:
    """
        Returns probes of the weights of every set of synapses.

        Each probe is a `SnapshotProbe` of a whole kernel, after the last step of each group.

        Parameters
        ----------
        model : Controller
            A built model.

        Returns
        -------
        tuple of Probe
            `SnapshotProbe` probes. The measurements holding them need a group.

        See Also
        --------
        summary : Population statistics of every soma and set of synapses.
        activity : Traces and rasters of a model on every step.
    """
    return tuple(SnapshotProbe(address(path, 'kernel', 'attribute')) for path, _ in _of_kind(model, 'synapses'))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def default(
        model: Controller,
        *,
        summary_group: int = 1000,
        activity_every: int = 100000,
        activity_length: int = 1000,
        weights_every: int = 100000,
    ) -> tuple[Measurements, ...]:
    """
        Returns measurements for any model, recorded by triggers.

        * ``'summary'``: the probes of `summary`, recorded on every step, one record per
          ``summary_group`` steps.
        * ``'activity'``: the probes of `activity`, recorded for ``activity_length`` steps out of
          every ``activity_every``.
        * ``'weights'``: the probes of `weights`, one record per ``weights_every`` steps. Only the
          call holding the last step of each group is recorded.

        Measurements without probes are left out.

        Parameters
        ----------
        model : Controller
            A built model.
        summary_group : int, default 1000
            Steps per record of ``'summary'``.
        activity_every : int, default 100_000
            Period of ``'activity'``, in steps.
        activity_length : int, default 1000
            Steps of ``'activity'`` recorded per period.
        weights_every : int, default 100_000
            Steps per record of ``'weights'``.

        Returns
        -------
        tuple of Measurements

        Notes
        -----
        Each set of measurements recorded together compiles once. At most four sets occur.

        See Also
        --------
        Recorder : Uses these measurements when given none.
        Measurements : A named set of probes recorded together.
    """
    measurements = (
        Measurements('summary', summary(model), group=summary_group, trigger=Always()),
        Measurements('activity', activity(model), trigger=Every(activity_every, length=activity_length)),
        Measurements('weights', weights(model), group=weights_every, trigger=Every(weights_every, offset=weights_every - 1)),
    )
    return tuple(r for r in measurements if r.probes)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
