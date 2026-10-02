#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from spark.nn.controllers.base import Controller

import jax
import numpy as np
from math import prod
import dataclasses as dc
from spark.core.payloads import SparkPayload
from spark.core.backend import split, merge, Variable, Module
from spark.recording.probe_context import ProbeContext
from spark.recording.probe import CALL, get_recorded_array, resolve, address

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@dc.dataclass(frozen=True)
class ProbeTarget:
    """
        Description of variable that a Probe can measure.
        
        Similar in spirit to Specs but for Probes.

        Attributes
        ----------
        address : str
            Probe address.
        kind : str
            ``'port'`` or ``'attribute'``.
        shape : tuple of int
            Shape of the value.
        dtype : numpy.dtype
            Dtype of the recorded array. Bool for spike payloads.
        payload : str
            Class name of the value of a port or property. ``'Variable'`` for a variable.
        module : str
            Class name of the module producing the port or holding the attribute. For the inputs of
            a controller, the controller.

        See Also
        --------
        get_probe_targets : Lists every value of a controller that a probe can address.
        Probe : A value read from a controller, and how it is recorded.
    """
    address: str
    kind: str
    shape: tuple[int, ...]
    dtype: np.dtype
    payload: str
    module: str

    @property
    def size(self) -> int:
        """
            Number of entries of the value.
        """
        return prod(self.shape)

    @property
    def spikes(self) -> bool:
        """
            Whether the value is a spike payload.
        """
        return self.payload == 'SpikeArray'

    def __str__(self) -> str:
        return f'{self.address:<48} {self.kind:<9} {str(self.shape):<16} {str(self.dtype):<8} {self.payload:<14} {self.module}'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _attributes(controller: Controller) -> list[tuple[tuple[str, ...], str, str]]:
    """
        Utility method to gather the path of every property and public variable of the controller.
    """
    from spark.nn.controllers.base import Controller

    found = []

    def walk(node: tp.Any, path: tuple[str, ...]) -> None:
        get_properties = getattr(type(node), 'get_properties', None)
        for name in (get_properties() if get_properties else ()):
            found.append((path, name, type(node).__name__))
        if isinstance(node, Controller):
            for child in node._modules_names:
                walk(getattr(node, child), (*path, child))
            return
        for name, value in vars(node).items():
            if name.startswith('_') or name == 'rng':
                continue
            if isinstance(value, Variable):
                found.append((path, name, type(node).__name__))
            elif isinstance(value, Module):
                walk(value, (*path, name))

    walk(controller, ())

    return found

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _check_payloads(inputs: dict[str, tp.Any]) -> None:
    """
        Raises a TypeError when an input is not a `SparkPayload`.
    """
    for name, value in inputs.items():
        if not isinstance(value, SparkPayload):
            raise TypeError(
                f'`get_probe_targets` takes payloads, such as spark.SpikeArray; input "{name}" is a {type(value).__name__}.'
            )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_probe_targets(controller: Controller, inputs: dict[str, SparkPayload]) -> tuple[ProbeTarget, ...]:
    """
        Returns a list of every variable (input, output or property), within the controller, 
        that can be targeted using a Probe, as well as its shape and dtype.

        Parameters
        ----------
        controller : Controller
            The controller to be described.
        inputs : dict[str, SparkPayload]
            Input sample.

        Returns
        -------
        tuple of ProbeTarget
            Tuple of ProbeTargets pointing towards the controller's input/output/attributes,  
            sorted by module path and name, that a Probe can target.

        Raises
        ------
        TypeError
            When an input is not a payload.

        See Also
        --------
        ProbeTarget : A value of a controller that a probe can target.
        Probe : A value read from a controller, and how it is recorded.
        validate : Checks that every probe addresses something the controller produces.

        Examples
        --------
        >>> for target in spark.recording.get_probe_targets(brain, inputs):
        ...     print(target)
    """
    _check_payloads(inputs)
    graph, state = split((controller))
    attributes = _attributes(controller)
    payloads: dict[tuple, str] = {}
    order: list[tuple[tuple[str, ...], str]] = []

    def trace(state, inputs):
        called = merge(graph, state)
        with ProbeContext((), capture_all=True) as capture:
            called(**inputs)
        ports = {}
        for (path, port), value in capture.captured.items():
            order.append((path, port))
            payloads[('port', path, port)] = type(value).__name__
            ports[(path, port)] = get_recorded_array(value)
        values = {}
        for path, name, _ in attributes:
            try:
                value = getattr(resolve(called, path), name)
                values[(path, name)] = get_recorded_array(value)
            except (AttributeError, TypeError, ValueError):
                # Skip non-ArrayLike properties
                continue
            payloads[('attribute', path, name)] = 'Variable' if isinstance(value, Variable) else type(value).__name__
        return ports, values

    ports, values = jax.eval_shape(trace, state, inputs)
    targets = []
    for path, port in order:
        struct = ports[(path, port)]
        holder = resolve(controller, path[:-1] if path and path[-1] == CALL else path)
        targets.append(ProbeTarget(
            address(path, port, 'port'), 'port', tuple(struct.shape), np.dtype(struct.dtype),
            payloads[('port', path, port)], type(holder).__name__,
        ))
    modules = {(path, name): module for path, name, module in attributes}
    for (path, name), struct in sorted(values.items()):
        targets.append(ProbeTarget(
            address(path, name, 'attribute'), 'attribute', tuple(struct.shape), np.dtype(struct.dtype),
            payloads[('attribute', path, name)], modules[(path, name)],
        ))
    return tuple(targets)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
