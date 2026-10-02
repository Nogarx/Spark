#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import dataclasses as dc
from spark.recording.probe import Probe, ProbeMode, SummaryProbe, DeltaProbe, GROUPED_PROBES, SummaryReduction, DeltaReduction
from spark.recording.triggers import Trigger, Manual
from spark.recording.utils import integer, is_name

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

_SCALAR_MODES = (ProbeMode.SUMMARY, ProbeMode.DELTA)
"""
    Modes of the probes whose reductions are written as scalar series.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True)
class Measurements:
    """
        Named set of probes recorded together.

        The steps recorded are those the trigger or `Recorder.record` ask for. With ``group``, the
        summaries, snapshots and deltas give one record per group of steps.

        Parameters
        ----------
        name : str
            Name of the measurements and of their files. Letters, digits, ``_``, ``.`` and ``-``,
            starting with a letter or a digit.
        probes : sequence of Probe
            One probe per address and mode.
        group : int or str, optional
            How the steps of the run are split into groups for the `SummaryProbe`, `SnapshotProbe`
            and `DeltaProbe` probes, which give one record per group. Required with such probes.

            * A number of steps ``n``: groups of ``n`` steps on the steps of the run, ``[0, n)``,
              ``[n, 2n)``, and so on, whatever step the recording starts at.
            * The name of a tag set with `Recorder.tag`, such as ``'episode'``: a group lasts while
              the tag keeps its value, and a new group starts each time the tag takes a different
              value. A tag going from 1 to 2 and back to 1 gives three groups. The steps before the
              tag is first set are one group.
        trigger : Trigger, default Manual()
            Steps recorded without a call to `Recorder.record`. `Manual` records none.
        raw : sequence of str, optional
            Names of the raw streams given to `Recorder.raw` that are written with these
            measurements.
        lookback : int, default 0
            Steps kept on the device before a recorded step, and written ahead of it.
        views : dict, optional
            How the viewer draws a value, by probe address or raw stream name, as
            ``{'env/frame': {'kind': 'image', 'shape': [84, 84, 3]}}``. Written to the run.

        Raises
        ------
        TypeError
            When a probe is not a `Probe`, or ``trigger`` is not a `Trigger`.
        ValueError
            When the name is invalid, probes share a key, ``group`` is invalid or missing for
            grouped probes, a probe has a group other than ``group``, or ``lookback`` is negative.

        Notes
        -----
        A group is recorded from the first call of the model holding a step asked for to the end of
        the group, past the steps asked for. A group recorded from its middle covers the steps
        recorded only. A group of steps ends with its last step, and a group by tag when the tag
        changes. The group in progress when the recorder closes is written cut short.

        With ``lookback`` above 0, the probes are recorded on every call. The records of at least
        the last ``lookback`` steps stay on the device. When a step is recorded, they are
        transferred and written with it. Raw frames are not kept.

        The measurements of a recorder share the probes of equal key, merged by `merge_probes`. Such
        probes must agree on their group, and on ``bins`` and ``range`` when both count a histogram.
        Probes without reductions must be equal.

        See Also
        --------
        Probe : A value read from a model, and how it is recorded.
        Recorder.record : Records a set of measurements for the next steps.
        Trigger : Base class for triggers, which ask for steps on their own.
        presets.default : Measurements for any model, recorded by triggers.

        Examples
        --------
        >>> Measurements('episode', probes, group='episode', raw=('env/frame',))
        >>> summary = spark.recording.presets.summary(brain)
        >>> Measurements('summary', summary, group=1000, trigger=Always())
        >>> activity = spark.recording.presets.activity(brain)
        >>> Measurements('activity', activity, trigger=Every(100_000, length=1000))
    """
    name: str
    probes: tuple[Probe, ...]
    _: dc.KW_ONLY
    group: int | str | None = None
    trigger: Trigger = Manual()
    raw: tuple[str, ...] = ()
    lookback: int = 0
    views: dict[str, dict] = dc.field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not is_name(self.name):
            raise ValueError(
                f'Invalid measurements name "{self.name}". Expected letters, digits, "_", "." and "-", starting with '
                f'a letter or a digit.'
            )
        probes = tuple(self.probes)
        if not all(isinstance(p, Probe) for p in probes):
            raise TypeError(f'Measurements "{self.name}": every probe must be a Probe.')
        keys = [p.key for p in probes]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f'Measurements "{self.name}": probes share the keys {sorted({k for k in keys if keys.count(k) > 1})}. '
                f'Expected one probe per address and mode.'
            )
        owner = f'Measurements "{self.name}"'
        if self.group is not None:
            if self.group == '':
                raise ValueError(f'{owner}: "group" names a tag, or counts steps; got an empty name.')
            group = self.group if isinstance(self.group, str) else integer(self.group, 'group', lowest=1, owner=owner)
            object.__setattr__(self, 'group', group)
        grouped = [p.key for p in probes if isinstance(p, GROUPED_PROBES)]
        if grouped and self.group is None:
            raise ValueError(
                f'Measurements "{self.name}": {", ".join(grouped)} give one record per group of steps; give "group", a '
                f'number of steps or the name of a tag.'
            )
        mixed = [p.key for p in probes if isinstance(p, GROUPED_PROBES) and p.group is not None and p.group != self.group]
        if mixed:
            raise ValueError(f'Measurements "{self.name}": the probes {", ".join(mixed)} have a group of their own; the measurements sets it.')
        probes = tuple(dc.replace(p, group=self.group) if isinstance(p, GROUPED_PROBES) else p for p in probes)
        object.__setattr__(self, 'probes', probes)
        object.__setattr__(self, 'raw', tuple(self.raw))
        object.__setattr__(self, 'lookback', integer(self.lookback, 'lookback', lowest=0, owner=owner))
        if not isinstance(self.trigger, Trigger):
            raise TypeError(f'{owner}: the trigger must be a Trigger, got {type(self.trigger).__name__}.')

    def to_dict(self) -> dict[str, tp.Any]:
        """
            Returns the fields of the measurements as JSON types.

            Returns
            -------
            dict
                Keyword arguments of `Measurements`, with probes and trigger as their own `to_dict`.
        """
        return {
            'name': self.name,
            'probes': [p.to_dict() for p in self.probes],
            'group': self.group,
            'trigger': self.trigger.to_dict(),
            'raw': list(self.raw),
            'lookback': self.lookback,
            'views': dict(self.views),
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> Measurements:
        """
            Rebuilds measurements from `to_dict`.

            A `When` trigger comes back as `Manual`.

            Parameters
            ----------
            data : dict
                Fields of the measurements, as `to_dict` gives them.

            Returns
            -------
            Measurements
        """
        return cls(
            name=data['name'],
            probes=tuple(Probe.from_dict(p) for p in data['probes']),
            group=data.get('group'),
            trigger=Trigger.from_dict(data['trigger']),
            raw=tuple(data.get('raw', ())),
            lookback=int(data.get('lookback', 0)),
            views=dict(data.get('views', {})),
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def scalar_key(measurements: str, probe_key: str, reduction: str) -> str:
    """
        Returns the name of the scalar series of one reduction of a probe.

        The reductions of summaries and deltas share no name, so the address of the probe and the
        reduction name one series of the measurements.

        Parameters
        ----------
        measurements : str
            Name of the measurements writing the series.
        probe_key : str
            Key of the probe, as `Probe.key`, or its address.
        reduction : str
            Name of the reduction.

        Returns
        -------
        str
            ``<measurements>/<address>/<reduction>``.
    """
    return f'{measurements}/{_address(probe_key)}/{reduction}'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _address(probe_key: str) -> str:
    """
        Returns the address of a probe key.
    """
    address, _, mode = probe_key.rpartition('@')
    return address if address and mode in _SCALAR_MODES else probe_key

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def canonical_scalar_key(name: str) -> str:
    """
        Returns the name of a scalar series in the form `scalar_key`.
    """
    parts = name.split('/')
    if len(parts) >= 3:
        parts[-2] = _address(parts[-2])
    return '/'.join(parts)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def scalar_key_forms(name: str) -> tuple[str, ...]:
    """
        Returns the names a scalar series may be stored under: ``name``.
    """
    forms = [name, canonical_scalar_key(name)]
    parts = forms[-1].split('/')
    if len(parts) >= 3:
        mode = 'summary' if parts[-1] in SummaryReduction else 'delta' if parts[-1] in DeltaReduction else None
        if mode is not None:
            forms.append('/'.join((*parts[:-2], f'{parts[-2]}@{mode}', parts[-1])))
    return tuple(dict.fromkeys(forms))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def merge_probes(probes: tp.Iterable[Probe]) -> tuple[Probe, ...]:
    """
        Merges probes sharing a key.

        Probes sharing a key become one probe. A `SummaryProbe` or a `DeltaProbe` takes the union of
        their reductions, in the order they are first given. 
        
        Parameters
        ----------
        probes : iterable of Probe
            Probes to merge.

        Returns
        -------
        tuple of Probe
            One probe per key, ordered by key.

        Raises
        ------
        ValueError
            When probes sharing a key differ in their group, or both count a histogram and differ in
            ``bins`` or ``range``, or have no reductions and are not equal.
    """
    merged: dict[str, Probe] = {}
    for probe in probes:
        other = merged.get(probe.key)
        if other is None:
            merged[probe.key] = probe
            continue
        if isinstance(probe, (SummaryProbe, DeltaProbe)):
            both = isinstance(probe, SummaryProbe) and 'hist' in other.reduce and 'hist' in probe.reduce
            agree = other.group == probe.group and not (both and (other.bins, other.range) != (probe.bins, probe.range))
        else:
            agree = other == probe
        if not agree:
            raise ValueError(
                f'Two probes of "{probe.key}" differ in more than their reductions: {other} and {probe}. '
                f'Expected measurements sharing a probe to agree on everything else.'
            )
        if isinstance(probe, SummaryProbe):
            reduce = other.reduce + tuple(r for r in probe.reduce if r not in other.reduce)
            histogram = other if 'hist' in other.reduce else probe
            merged[probe.key] = dc.replace(other, reduce=reduce, bins=histogram.bins, range=histogram.range)
        elif isinstance(probe, DeltaProbe):
            merged[probe.key] = dc.replace(other, reduce=other.reduce + tuple(r for r in probe.reduce if r not in other.reduce))
    return tuple(merged[key] for key in sorted(merged))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
