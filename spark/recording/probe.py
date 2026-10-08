#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from spark.nn.controllers.base import Controller

import abc
import jax
import enum
import numpy as np
import dataclasses as dc
from spark.core.backend import Variable, Module
from spark.core.payloads import SparkPayload, SpikeArray
from spark.recording.utils import integer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

CALL = '__call__'
"""
    Path segment naming the inputs of a controller.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ProbeMode(enum.StrEnum):
    """
        How a probe records its value. 
        
        Each probe is associated with a specific `Probe.mode`.
    """
    SUMMARY = 'summary'
    """
        `SummaryProbe`: reductions of the value over the units and over each group of steps.
    """
    TRACE = 'trace'
    """
        `TraceProbe`: the value on every step.
    """
    RASTER = 'raster'
    """
        `RasterProbe`: whether each unit is active, on every step.
    """
    SNAPSHOT = 'snapshot'
    """
        `SnapshotProbe`: the value of an attribute at the end of each group of steps.
    """
    DELTA = 'delta'
    """
        `DeltaProbe`: reductions of the change of an attribute over each group of steps.
    """

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SummaryReduction(enum.StrEnum):
    """
        `SummaryProbe` requested operations (reductions) on the Probes.
    """
    MEAN = 'mean'
    """
        Mean over the units and the steps.
    """
    STD = 'std'
    """
        Standard deviation over the units and the steps.
    """
    MIN = 'min'
    """
        Minimum over the units and the steps.
    """
    MAX = 'max'
    """
        Maximum over the units and the steps.
    """
    ACTIVE_FRACTION = 'active_fraction'
    """
        Fraction of the units active, averaged over the steps. For spikes, the firing rate per step.
    """
    ACTIVE_FRACTION_PER_UNIT = 'active_fraction_per_unit'
    """
        Fraction of the steps on which each unit is active. One value per unit.
    """
    INACTIVE_UNIT_FRACTION = 'inactive_unit_fraction'
    """
        Fraction of the units inactive in the group.
    """
    HIST = 'hist'
    """
        Counts of the values of the units and the steps in ``bins`` equal bins over ``range``.
    """

    @property
    def scalar(self) -> bool:
        """
            Whether the reduction gives one number per group of steps. ``ACTIVE_FRACTION_PER_UNIT``
            and ``HIST`` give an array. A `Recorder` also writes the scalar reductions to the
            scalars of the run.
        """
        return self not in (SummaryReduction.ACTIVE_FRACTION_PER_UNIT, SummaryReduction.HIST)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class DeltaReduction(enum.StrEnum):
    """
        `DeltaProbe` requested operations (reductions) on the Probes. Unlike `SummaryProbe`, `DeltaProbe`
        compute differences of an attribute over each group of steps.
    """
    FULL = 'full'
    """
        The change of every unit.
    """
    NORM = 'norm'
    """
        Euclidean norm of the change.
    """
    MEAN_ABS = 'mean_abs'
    """
        Mean absolute change over the units.
    """

    @property
    def scalar(self) -> bool:
        """
            Whether the reduction gives one number per group of steps. ``FULL`` gives an array. A
            `Recorder` also writes the scalar reductions to the scalars of the run.
        """
        return self is not DeltaReduction.FULL

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True, eq=False)
class Probe(abc.ABC):
    """
        Specification for a measurement, denoted by a port address and the measurement operator:
        `SummaryProbe`, `TraceProbe`, `RasterProbe`, `SnapshotProbe` or `DeltaProbe`.

        Parameters
        ----------
        address : str
            Address of the value. ``path`` is the dotted chain of module names from the root
            controller.

            * ``path:port``: an output port of a module.
            * ``path.__call__:port``: an input port of a controller, ``__call__:port`` for the root.
            * ``path.name``: an attribute of a module.

            A pattern, as ``*_excitatory.soma:spikes`` or ``**.soma.potential``, stands for a probe
            of every address it matches, with the same fields (see `spark.core.addresses`). It is
            matched when the probes are checked against a built model: by `validate`, and by a
            `Recorder` given the model.

        Attributes
        ----------
        mode : ProbeMode
            Mode of the class of the probe.
        kind : str
            ``'port'`` or ``'attribute'``.
        path : tuple of str
            Path of the module producing the port or holding the attribute.
        name : str
            Port or attribute name.
        key : str
            Name of the entry of the probe in the records, ``address@mode``.

        Raises
        ------
        ValueError
            When the address is malformed, or a field is invalid.

        Notes
        -----
        Probes are frozen and hashable. Probes of the same class with equal fields compare and hash
        equal. A tuple of probes can be a static argument of a jitted function.

        Ports are read as the module produces them. Attributes are read at the end of each step,
        after every module of the step ran.

        See Also
        --------
        Measurements : A named set of probes recorded together.
        get_probe_targets : Lists the values of a controller that a probe can address.
        validate : Checks probes against a controller.
        Recorder : Records the measurements asked for and writes them to a run.
    """
    address: str
    mode: tp.ClassVar[ProbeMode]
    kind: str = dc.field(init=False, repr=False)
    path: tuple[str, ...] = dc.field(init=False, repr=False)
    name: str = dc.field(init=False, repr=False)
    _values: tuple = dc.field(init=False, repr=False)
    _hash: int = dc.field(init=False, repr=False)

    def __post_init__(self) -> None:
        kind, path, name = _parse_address(self.address)
        self._set('kind', kind)
        self._set('path', path)
        self._set('name', name)
        self._prepare()
        values = tuple(self._kwargs().values())
        self._set('_values', values)
        self._set('_hash', hash((self.mode, values)))

    @abc.abstractmethod
    def _prepare(self) -> None:
        """
            Normalizes and checks the fields of the class. Raises a ValueError for an invalid field.
        """

    def _set(self, field: str, value: tp.Any) -> None:
        object.__setattr__(self, field, value)

    def _kwargs(self) -> dict[str, tp.Any]:
        return {field.name: getattr(self, field.name) for field in dc.fields(self) if field.init}

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._values == other._values

    def __hash__(self) -> int:
        return self._hash

    def __reduce__(self) -> tuple:
        # Rebuilt from its fields: the hash of strings differs between processes.
        return (_rebuild, (type(self), self._kwargs()))

    @property
    def key(self) -> str:
        return f'{self.address}@{self.mode}'

    @property
    def per_step(self) -> bool:
        """
            Whether the probe is read on every step (`STEP_PROBES`).
        """
        return isinstance(self, STEP_PROBES)

    def to_dict(self) -> dict[str, tp.Any]:
        """
            Returns the mode and the fields of the probe as JSON types.

            Returns
            -------
            dict
                ``mode``, and the keyword arguments of the class, with lists in place of tuples.
        """
        data = {'mode': self.mode.value}
        for name, value in self._kwargs().items():
            data[name] = list(value) if isinstance(value, tuple) else value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> Probe:
        """
            Rebuilds a probe from `to_dict`.

            Parameters
            ----------
            data : dict
                Mode and fields of a probe, as `to_dict` gives them.

            Returns
            -------
            Probe
                A probe of the class of the mode.

            Raises
            ------
            ValueError
                When the mode is unknown, or is not the mode of the class called.
        """
        fields = dict(data)
        mode = fields.pop('mode', None)
        probe_class = _CLASSES.get(mode)
        if probe_class is None:
            raise ValueError(f'Unknown probe mode {mode!r}. Expected one of: {", ".join(_CLASSES)}.')
        if cls is not Probe and probe_class is not cls:
            raise ValueError(f'{cls.__name__} rebuilds probes of mode "{cls.mode}", got "{mode}".')
        return probe_class(**fields)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True, eq=False)
class SummaryProbe(Probe):
    """
        Reductions of a value over the units and over each group of steps.

        Parameters
        ----------
        address : str
            Address of the value, as in `Probe`.
        reduce : SummaryReduction or str or sequence of them, default ('mean', 'std', 'min', 'max')
            Reductions, as `SummaryReduction` members or their names.

            * ``MEAN``, ``STD``, ``MIN``, ``MAX``: over the units and the steps of the group.
            * ``ACTIVE_FRACTION``: fraction of the units active, averaged over the steps.
            * ``ACTIVE_FRACTION_PER_UNIT``: fraction of the steps on which each unit is active.
            * ``INACTIVE_UNIT_FRACTION``: fraction of the units active on none of the steps.
            * ``HIST``: counts in ``bins`` equal bins over ``range``.
        bins : int, default 32
            Number of bins of ``HIST``. Ignored without it.
        range : tuple of float, optional
            Lower and upper edges of ``HIST``, finite as float32. Required by ``HIST`` and ignored
            without it. Values outside the range, and NaN, are not counted.
        group : int or str, optional
            How the steps are split into groups, one record per group. Set by the `Measurements`
            holding the probe; see `Measurements` for how a group is recorded.

            * A number of steps ``n``: groups of ``n`` steps on the steps of the run, ``[0, n)``,
              ``[n, 2n)``, and so on.
            * The name of a tag, such as ``'episode'``: a new group each time the tag takes a
              different value (`Recorder.tag`).

        Notes
        -----
        A unit is an entry of the value. A unit is active on a step when its value is nonzero.

        Means, spreads and fractions are computed in float32 and counts in int32, whatever the dtype
        of the value. Histogram counts are int64. ``MIN`` and ``MAX`` keep integer dtypes, and are
        float32 for floating values and uint8 for bool.

        A NaN in a value makes ``MEAN``, ``STD``, ``MIN`` and ``MAX`` NaN, and counts as active for
        the fractions. An infinity makes ``MIN`` or ``MAX`` infinite, and ``MEAN`` and ``STD`` NaN.
        A float32 value near 1e19 or beyond can overflow ``STD``.

        Summaries keep device memory in proportion to the value and to the groups a call
        touches. A histogram is counted after the call from the values of every step, or step by
        step when those values take more than `SETTINGS.histogram_rows_limit` bytes.

        Histogram edges are float32, as in ``numpy.histogram`` of float32 values. XLA on the CPU
        reads float32 subnormals (below about 1.2e-38 in magnitude) as zero in fractions and
        histograms. GPUs do not read them as zero.

        Examples
        --------
        >>> SummaryProbe('first_pool:out_spikes', reduce=(
        ...     SummaryReduction.ACTIVE_FRACTION, SummaryReduction.INACTIVE_UNIT_FRACTION,
        ... ))
        >>> SummaryProbe('first_pool.soma.potential', reduce=SummaryReduction.HIST,
        ...              range=(-80.0, 40.0))
    """
    mode: tp.ClassVar[ProbeMode] = ProbeMode.SUMMARY
    _: dc.KW_ONLY
    reduce: tuple[SummaryReduction | str, ...] = ('mean', 'std', 'min', 'max')
    bins: int = 32
    range: tuple[float, float] | None = None
    group: int | str | None = None

    def _prepare(self) -> None:
        self._set('reduce', _reductions(self, self.reduce, SummaryReduction))
        self._set('group', _group(self.group))
        if SummaryReduction.HIST in self.reduce:
            self._set('bins', integer(self.bins, 'bins', lowest=1))
            self._set('range', _histogram_range(self))
        else:
            # Without 'hist', bins and range take their defaults.
            self._set('bins', 32)
            self._set('range', None)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True, eq=False)
class TraceProbe(Probe):
    """
        The value on every step.

        Parameters
        ----------
        address : str
            Address of the value, as in `Probe`.
        units : sequence of int, optional
            Flat indices of the units kept. All units when omitted.
        stride : int, default 1
            Keeps the steps ``t`` of the run with ``t % stride == 0``.

        Notes
        -----
        A unit is an entry of the value. A trace keeps the dtype of the value. A trace of an
        attribute holds the value the next step starts from.

        A trace keeps every step of the call it records. With a stride, it keeps only the steps of
        the stride when every step would take more than `SETTINGS.spaced_rows_limit`
        bytes.

        Examples
        --------
        >>> TraceProbe('first_pool.soma.potential', units=range(64))
        >>> TraceProbe('__call__:signal', stride=10)
    """
    mode: tp.ClassVar[ProbeMode] = ProbeMode.TRACE
    _: dc.KW_ONLY
    units: tuple[int, ...] | None = None
    stride: int = 1

    def _prepare(self) -> None:
        self._set('units', _units(self.units))
        self._set('stride', integer(self.stride, 'stride', lowest=1))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True, eq=False)
class RasterProbe(Probe):
    """
        Whether each unit is active, on every step.

        Parameters
        ----------
        address : str
            Address of the value, as in `Probe`.
        units : sequence of int, optional
            Flat indices of the units kept. All units when omitted.
        stride : int, default 1
            Keeps the steps ``t`` of the run with ``t % stride == 0``.

        Notes
        -----
        A unit is an entry of the value. A unit is active on a step when its value is nonzero. A
        NaN counts as active. A float16 or bfloat16 value is rounded to its dtype before the test.

        A raster is moved and stored as bits, and read as bool. It keeps every step of the call it
        records. With a stride, it keeps only the steps of the stride when every step would take
        more than `SETTINGS.spaced_rows_limit` bytes.

        XLA on the CPU reads float32 subnormals (below about 1.2e-38 in magnitude) as zero. GPUs do
        not read them as zero.

        Examples
        --------
        >>> RasterProbe('first_pool:out_spikes')
        >>> RasterProbe('first_pool:out_spikes', units=range(256))
    """
    mode: tp.ClassVar[ProbeMode] = ProbeMode.RASTER
    _: dc.KW_ONLY
    units: tuple[int, ...] | None = None
    stride: int = 1

    def _prepare(self) -> None:
        self._set('units', _units(self.units))
        self._set('stride', integer(self.stride, 'stride', lowest=1))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True, eq=False)
class SnapshotProbe(Probe):
    """
        The value of an attribute at the end of each group of steps.

        Parameters
        ----------
        address : str
            Address of an attribute, as in `Probe`.
        units : sequence of int, optional
            Flat indices of the units kept. All units when omitted.
        group : int or str, optional
            How the steps are split into groups, one record per group, as for `SummaryProbe`: a
            number of steps or the name of a tag. Set by the `Measurements` holding the probe.

        Notes
        -----
        The value is read after the last step of each group. Until the group ends, a snapshot with a
        group keeps the value after each call on the device, and, for a call crossing the end of a
        group, the value at that end.

        Examples
        --------
        >>> SnapshotProbe('first_pool.synapses.kernel')
    """
    mode: tp.ClassVar[ProbeMode] = ProbeMode.SNAPSHOT
    _: dc.KW_ONLY
    units: tuple[int, ...] | None = None
    group: int | str | None = None

    def _prepare(self) -> None:
        _check_attribute(self)
        self._set('units', _units(self.units))
        self._set('group', _group(self.group))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True, eq=False)
class DeltaProbe(Probe):
    """
        Reductions of the change of an attribute over each group of steps.

        Parameters
        ----------
        address : str
            Address of an attribute, as in `Probe`.
        reduce : DeltaReduction or str or sequence of them, default ('norm',)
            Reductions, as `DeltaReduction` members or their names.

            * ``FULL``: the change of every unit.
            * ``NORM``: the Euclidean norm of the change.
            * ``MEAN_ABS``: the mean absolute change over the units.
        group : int or str, optional
            How the steps are split into groups, one record per group, as for `SummaryProbe`: a
            number of steps or the name of a tag. Set by the `Measurements` holding the probe.

        Notes
        -----
        The change is the value after the last step of the group minus the value before its first
        step recorded, computed in float32. Until the group ends, a delta with a group keeps the
        value before and after each call on the device, and, for a call crossing the end of a group,
        the value at that end.

        Examples
        --------
        >>> DeltaProbe('first_pool.synapses.kernel',
        ...            reduce=(DeltaReduction.NORM, DeltaReduction.MEAN_ABS))
    """
    mode: tp.ClassVar[ProbeMode] = ProbeMode.DELTA
    _: dc.KW_ONLY
    reduce: tuple[DeltaReduction | str, ...] = ('norm',)
    group: int | str | None = None

    def _prepare(self) -> None:
        _check_attribute(self)
        self._set('reduce', _reductions(self, self.reduce, DeltaReduction))
        self._set('group', _group(self.group))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _rebuild(cls: type[Probe], kwargs: dict[str, tp.Any]) -> Probe:
    """
        Rebuilds a pickled probe.
    """
    return cls(**kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _reductions(probe: Probe, value: tp.Any, accepted: type[enum.StrEnum]) -> tuple[str, ...]:
    """
        Returns the reductions computed by the probe.

        Raises a ValueError for none, an unknown one or a repeated one.
    """
    reductions = (value,) if isinstance(value, str) else tuple(value)
    names = tuple(r.value if isinstance(r, enum.Enum) else r for r in reductions)
    if not names:
        raise ValueError(f'The {type(probe).__name__} of "{probe.address}" takes at least one reduction.')
    unknown = [r for r in names if r not in tuple(accepted)]
    if unknown:
        raise ValueError(
            f'Unknown reductions {", ".join(map(repr, unknown))} for the {type(probe).__name__} of "{probe.address}". '
            f'Expected any of: {", ".join(accepted)}.'
        )
    if len(set(names)) != len(names):
        raise ValueError(f'Repeated reductions in {names}.')
    return names

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _units(value: tp.Any) -> tuple[int, ...] | None:
    """
        Returns ``units`` as a tuple of int. 
        
        Raises a ValueError when empty or negative.
    """
    if value is None:
        return None
    units = tuple(int(u) for u in value)
    if len(units) == 0 or min(units) < 0:
        raise ValueError(f'"units" must be non-empty and non-negative, got {units}.')
    return units

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _group(value: tp.Any) -> int | str | None:
    """
        Returns ``group`` as an int, a tag name or None. 
        
        Raises a ValueError for an empty name or a number of steps below 1.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            raise ValueError('"group" names a tag, or counts steps; got an empty name.')
        return value
    return integer(value, 'group', lowest=1)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _histogram_range(probe: SummaryProbe) -> tuple[float, float]:
    """
        Returns the ``range`` of a histogram as two floats. 
        
        Raises a ValueError when missing, or when the edges are not finite and ordered as float32.
    """
    if probe.range is None:
        raise ValueError(f'The "hist" reduction of "{probe.address}" needs a range.')
    if len(probe.range) != 2:
        raise ValueError(f'"range" takes a lower and an upper edge, got {probe.range}.')
    edges = (float(probe.range[0]), float(probe.range[1]))
    with np.errstate(over='ignore'):
        rounded = np.asarray(edges, np.float64).astype(np.float32)
    if not (np.isfinite(rounded).all() and rounded[0] < rounded[1]):
        raise ValueError(f'Invalid range {probe.range}. Expected finite float32 edges, the lower one first.')
    return edges

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _check_attribute(probe: Probe) -> None:
    """
        Raises a ValueError when ``probe`` addresses a port.
    """
    if probe.kind != 'attribute':
        raise ValueError(f'A {type(probe).__name__} reads attributes, got the port "{probe.address}".')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _parse_address(address: str) -> tuple[str, tuple[str, ...], str]:
    """
        Splits an address into its kind, module path and name. 
        
        Raises a ValueError if malformed.
    """
    if not isinstance(address, str) or not address:
        raise ValueError(f'A probe address must be a non-empty string, got {address!r}.')
    if ':' in address:
        path_str, port = address.split(':', 1)
        path = tuple(path_str.split('.'))
        if not port or ':' in port or not path_str or any(not segment for segment in path):
            raise ValueError(f'Invalid port address "{address}". Expected "path:port".')
        if CALL in path[:-1]:
            raise ValueError(f'Invalid port address "{address}". "{CALL}" may only close the path.')
        return 'port', path, port
    segments = tuple(address.split('.'))
    if any(not segment for segment in segments):
        raise ValueError(f'Invalid attribute address "{address}". Expected "path.name".')
    if CALL in segments:
        raise ValueError(f'Invalid attribute address "{address}". Inputs are ports: "{CALL}:port".')
    return 'attribute', segments[:-1], segments[-1]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def address(path: tp.Sequence[str], name: str, kind: str) -> str:
    """
        Returns the address of the port or attribute ``name`` of the module at ``path``.

        Parameters
        ----------
        path : sequence of str
            Module path. Ends with `CALL` for the inputs of a controller.
        name : str
            Port or attribute name.
        kind : str
            ``'port'`` or ``'attribute'``.

        Returns
        -------
        str
            ``path:name`` for a port, ``path.name`` for an attribute.
    """
    if kind == 'port':
        return f'{".".join(path)}:{name}'
    return '.'.join((*path, name))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def probe_addresses(controller: Controller) -> tuple[str, ...]:
    """
        Returns the address of every value of a built controller that a probe can read.

        Parameters
        ----------
        controller : Controller
            A built controller.

        Returns
        -------
        tuple of str
            The inputs of every controller (``path.__call__:port``), the output ports of every module
            of a controller (``path:port``), and every attribute holding an array (``path.name``).

        See Also
        --------
        get_probe_targets : Lists the same values, with their shapes and dtypes, from example inputs.
    """
    from spark.nn.controllers.base import Controller
    from spark.recording.probe_targets import _attributes
    found = []

    def ports(node: Controller, path: tuple[str, ...]) -> None:
        found.extend(address((*path, CALL), port, 'port') for port in node.get_controller_inputs())
        for child in node._modules_names:
            found.extend(address((*path, child), port, 'port') for port in node._modules_output_map[child])
            module = getattr(node, child)
            if isinstance(module, Controller):
                ports(module, (*path, child))

    ports(controller, ())
    for path, name, _ in _attributes(controller):
        try:
            get_recorded_array(getattr(resolve(controller, path), name))
        except (AttributeError, TypeError, ValueError):
            continue
        found.append(address(path, name, 'attribute'))
    return tuple(found)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def expand(controller: Controller, probes: tp.Iterable[Probe]) -> tuple[Probe, ...]:
    """
        Returns the probes with every pattern replaced by a probe of each address it matches.

        Parameters
        ----------
        controller : Controller
            A built controller.
        probes : iterable of Probe
            Probes, whose addresses may be patterns (see `spark.core.addresses`).

        Returns
        -------
        tuple of Probe
            The probes in the order given, each pattern replaced by probes with the same fields, one for
            each address it matches, in the order of `probe_addresses`.

        Raises
        ------
        ValueError
            When a pattern matches nothing.
    """
    from spark.core import addresses
    probes = tuple(probes)
    if not any(addresses.is_pattern(probe.address) for probe in probes):
        return probes
    found = probe_addresses(controller)
    expanded = []
    for probe in probes:
        if not addresses.is_pattern(probe.address):
            expanded.append(probe)
            continue
        matched = addresses.select(probe.address, found)
        if not matched:
            raise ValueError(
                f'"{probe.address}" matches nothing in the {type(controller).__name__}. `spark.recording.probe_addresses` '
                f'lists what a probe can read.'
            )
        expanded.extend(dc.replace(probe, address=match) for match in matched)
    return tuple(expanded)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def validate(controller: Controller, probes: tp.Iterable[Probe]) -> None:
    """
        Checks that every probe is a valid probe (targets an existing variable within the controller).

        The address of each probe must name a module, port or attribute of the controller. ``units`` must
        be within the size of the value, and a `SummaryProbe` or a `DeltaProbe` must have entries to reduce.
        An attribute must hold an array. A pattern is checked as the probes of the addresses it matches,
        and must match one at least.

        Parameters
        ----------
        controller : Controller
            A built controller, called at least once.
        probes : iterable of Probe
            Probes to check, one per key.

        Raises
        ------
        TypeError
            When ``controller`` is not a controller, or a probe is not a `Probe`.
        ValueError
            When the controller is not built, two probes share a key, or a probe does not fit the controller.
            The message lists the names available where the address failed.

        Notes
        -----
        The sizes of controller inputs and of the ports of nested controllers are not known before
        the controller is traced, and are not checked. `Runner` traces every probe of its recorder before
        its first call.

        See Also
        --------
        get_probe_targets : Lists the values of a controller that a probe can address.
    """
    from spark.nn.controllers.base import Controller
    if not isinstance(controller, Controller):
        raise TypeError(f'Probes address the modules of a controller, got {type(controller).__name__}.')
    if not getattr(controller, '__built__', False):
        raise ValueError('The controller is not built yet. Call it once with example inputs first.')
    probes = tuple(probes)
    for probe in probes:
        if not isinstance(probe, Probe):
            raise TypeError(f'Expected a Probe, got {type(probe).__name__}.')
    keys = set()
    for probe in expand(controller, probes):
        if probe.key in keys:
            raise ValueError(f'Probes share the key "{probe.key}". Expected one probe per address and mode.')
        keys.add(probe.key)
        if probe.kind == 'port':
            _validate_port(controller, probe)
        else:
            _validate_attribute(controller, probe)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _children(obj: tp.Any) -> list[str]:
    """
        Lists the names reachable from an object, for error messages.

        The modules of a controller, or the modules, variables and properties of another object.
    """
    names = getattr(obj, '_modules_names', None)
    if names is not None:
        return list(names)
    found = [k for k, v in vars(obj).items() if not k.startswith('__') and isinstance(v, (Module, Variable))]
    get_properties = getattr(type(obj), 'get_properties', None)
    if get_properties is not None:
        found += list(get_properties())
    return sorted(set(found))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def resolve(controller: tp.Any, path: tp.Sequence[str], address: str | None = None) -> tp.Any:
    """
        Retrieves the module pointed by ``path``, within ``controller``.

        Parameters
        ----------
        controller : object
            Object the path starts from, such as a controller.
        path : sequence of str
            Attribute names, followed one after the other.
        address : str, optional
            Address named in the error message. Defaults to the path.

        Returns
        -------
        object
            The object at the end of the path.

        Raises
        ------
        ValueError
            When a segment is not found. The message lists the names available at that level.
    """
    obj = controller
    for depth, segment in enumerate(path):
        try:
            obj = getattr(obj, segment)
        except AttributeError:
            where = '.'.join(path[:depth]) or 'the root'
            raise ValueError(
                f'"{segment}" not found under {where} for "{address or ".".join(path)}". '
                f'Available: {", ".join(_children(obj)) or "none"}.'
            ) from None
    return obj

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _validate_port(controller: Controller, probe: Probe) -> None:
    """
        Validates a port probe against the controller.
    """
    from spark.nn.controllers.base import Controller
    if probe.path[-1] == CALL:
        controller = resolve(controller, probe.path[:-1], probe.address)
        if not isinstance(controller, Controller):
            raise ValueError(f'"{probe.address}" names the inputs of a {type(controller).__name__}, which is not a controller.')
        ports = controller.get_controller_inputs()
        module = None
    else:
        parent = resolve(controller, probe.path[:-1], probe.address)
        if not isinstance(parent, Controller):
            raise ValueError(
                f'"{probe.address}": ports are read from the modules of a controller, and '
                f'"{".".join(probe.path[:-1])}" is a {type(parent).__name__}.'
            )
        module_name = probe.path[-1]
        if module_name not in parent._modules_names:
            where = '.'.join(probe.path[:-1]) or 'the root'
            raise ValueError(
                f'No module "{module_name}" under {where} for "{probe.address}". '
                f'Available: {", ".join(parent._modules_names)}.'
            )
        ports = parent._modules_output_map[module_name]
        module = getattr(parent, module_name)
    if probe.name not in ports:
        raise ValueError(f'No port "{probe.name}" for "{probe.address}". Available: {", ".join(ports) or "none"}.')
    # The size of an output port of a module that is not a controller is known from its specification;
    # other ports are checked when the controller is traced.
    get_specs = getattr(module, 'get_output_specs', None)
    if get_specs is not None and not isinstance(module, Controller):
        shape = getattr(get_specs().get(probe.name), 'shape', None)
        if isinstance(shape, tuple) and all(isinstance(d, int) for d in shape):
            _check_size(int(np.prod(shape)), probe)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _validate_attribute(controller: Controller, probe: Probe) -> None:
    """
        Validates an attribute probe against the controller.
    """
    holder = resolve(controller, probe.path, probe.address)
    try:
        value = getattr(holder, probe.name)
    except AttributeError:
        raise ValueError(
            f'No attribute "{probe.name}" for "{probe.address}". Available: {", ".join(_children(holder)) or "none"}.'
        ) from None
    try:
        array = get_recorded_array(value)
    except TypeError as error:
        raise ValueError(f'"{probe.address}": {error}') from None
    _check_size(array.size, probe)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _check_size(size: int, probe: Probe) -> None:
    """
        Checks ``units`` against the size of the value of a probe. 
        
        Raises a ValueError for a value with no entries.
    """
    if isinstance(probe, (TraceProbe, RasterProbe, SnapshotProbe)):
        check_units(size, probe.units, probe.address)
    if size == 0 and isinstance(probe, (SummaryProbe, DeltaProbe)):
        raise ValueError(f'"{probe.address}" has no entries; a {type(probe).__name__} reduces them.')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_recorded_array(value: tp.Any) -> jax.Array:
    """
        Returns the array a probe reduces.

        Parameters
        ----------
        value : SparkPayload or Variable or array
            Value read from the controller.

        Returns
        -------
        array

        Raises
        ------
        TypeError
            When the value holds no array.
    """
    if isinstance(value, SpikeArray):
        return value.spikes
    if isinstance(value, (SparkPayload, Variable)):
        return value.value
    if isinstance(value, (jax.Array, np.ndarray)):
        return value
    inner = getattr(value, 'value', None)
    if isinstance(inner, (jax.Array, np.ndarray)):
        return inner
    raise TypeError(f'A {type(value).__name__} holds no array to record.')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def check_units(size: int, units: tuple[int, ...] | None, address: str) -> None:
    """
        Checks that ``units`` are indices within a value of ``size`` entries.

        Parameters
        ----------
        size : int
            Number of entries of the value.
        units : tuple of int or None
            Flat indices, or None for all units.
        address : str
            Address named in the error message.

        Raises
        ------
        ValueError
            When an index of ``units`` is not below ``size``.
    """
    if units is not None and max(units) >= size:
        raise ValueError(f'"{address}" has {size} units, "units" asks for index {max(units)}.')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

STEP_PROBES = (SummaryProbe, TraceProbe, RasterProbe)
"""
    Probes read on every step.
"""

BOUNDARY_PROBES = (SnapshotProbe, DeltaProbe)
"""
    Probes read before and after the steps of each group, not on every step. Attributes only.
"""

GROUPED_PROBES = (SummaryProbe, SnapshotProbe, DeltaProbe)
"""
    Probes giving one record per group of steps.
"""

_CLASSES = {cls.mode: cls for cls in (SummaryProbe, TraceProbe, RasterProbe, SnapshotProbe, DeltaProbe)}
"""
    The class of each mode, for `Probe.from_dict`.
"""

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
