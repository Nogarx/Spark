#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import abc
import bisect
import dataclasses as dc

from spark.recording.utils import integer

if tp.TYPE_CHECKING:
    from spark.recording.records import Record

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@dc.dataclass(frozen=True)
class Trigger(abc.ABC):
    """
        Base class for triggers.

        A trigger decides which calls of the model record a set of measurements, in addition to
        all the explicit recordings invoked by `Recorder.record`.

        A trigger counts steps, or the values of ``tag``. Tags are set by hand with `Recorder.tag`,
        such as ``'episode'``. Without a tag, the trigger counts the steps of the run, which the
        recorder advances on its own.

        Parameters
        ----------
        tag : str, optional
            Integer tag counted, such as ``'episode'``. Steps are counted without one.

        Raises
        ------
        ValueError
            When ``tag`` is not a non-empty string, or is ``'step'``.

        See Also
        --------
        Every : Records ``length`` steps out of every ``n``.
        At : Records ``length`` steps from each of a set of points.
        Between : Records the steps of a range.
        Always : Records every call.
        Manual : Records nothing on its own. The default trigger.
        When : Records steps after a condition holds for the records of other measurements.
        Recorder.record : Records a set of measurements for the next steps.
    """
    tag: str | None = dc.field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.tag is None:
            return
        if not isinstance(self.tag, str) or not self.tag:
            raise ValueError(f'{type(self).__name__}: "tag" must name a tag, got {self.tag!r}.')
        if self.tag == 'step':
            raise ValueError(f'{type(self).__name__}: "step" is not a tag. Steps are counted without "tag".')

    def recorded(self, counters: dict[str, int], spans: dict[str, int] | None = None) -> bool:
        """
            Tests whether a call of the model is recorded.

            Parameters
            ----------
            counters : dict of str to int
                Value of every counter at the start of the call.
            spans : dict of str to int, optional
                Counts of each counter the call covers. One for a counter not given.

            Returns
            -------
            bool
                False when the counter of the trigger is not in ``counters``.
        """
        counter = 'step' if self.tag is None else self.tag
        start = counters.get(counter)
        if start is None:
            return False
        span = max(int((spans or {}).get(counter, 1)), 1)
        return self.covers(int(start), int(start) + span)

    @abc.abstractmethod
    def covers(self, start: int, stop: int) -> bool:
        """
            Tests whether any count in ``[start, stop)`` is recorded.

            Parameters
            ----------
            start : int
                First count.
            stop : int
                Count after the last.

            Returns
            -------
            bool
        """

    def to_dict(self) -> dict[str, tp.Any]:
        """
            Returns the fields of the trigger and its class name as JSON types.

            Returns
            -------
            dict
                The fields, and the class name under ``'kind'``.
        """
        return {
            'kind': type(self).__name__, **dc.asdict(self)
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> Trigger:
        """
            Rebuilds a trigger from `to_dict`.

            `When` triggers and triggers defined outside this module come back as `Manual`, with
            their ``tag``. The field ``'unit'`` of earlier runs is read as ``tag``, ``'step'`` as no
            tag.

            Parameters
            ----------
            data : dict
                Fields of a trigger, and its class name under ``'kind'``.

            Returns
            -------
            Trigger
                A trigger of the class named by ``'kind'``.

            Raises
            ------
            ValueError
                When called on a subclass, and ``'kind'`` names another class.
        """
        classes = {trigger.__name__: trigger for trigger in (Every, At, Between, Always, Manual)}
        fields = dict(data)
        kind = fields.pop('kind')
        if cls is not Trigger and kind != cls.__name__:
            raise ValueError(f'{cls.__name__} rebuilds triggers of kind "{cls.__name__}", got "{kind}".')
        if 'unit' in fields:
            unit = fields.pop('unit')
            fields['tag'] = None if unit == 'step' else unit
        if kind not in classes:
            return Manual(tag=fields.get('tag'))
        return classes[kind](**fields)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True)
class Manual(Trigger):
    """
        Default trigger of `Measurements`.

        Measurements with this trigger require an explicit call to `Recorder.record`.

        See Also
        --------
        Always : Records every call.
        When : Records steps after a condition holds for the records of other measurements.
        Recorder.record : Records a set of measurements for the next steps.
        Run.record : Asks the recorder writing a run to record a set of measurements.

        Examples
        --------
        >>> measurements = Measurements('episode', probes)      # Manual by default
        >>> recorder.record('episode')                          # Records the next call of the model
    """

    def covers(self, start: int, stop: int) -> bool:
        return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True)
class Every(Trigger):
    """
        Trigger recording ``length`` consecutive steps out of every ``n``.

        With ``tag``, it counts the values of the tag instead of steps: ``Every(25, tag='episode')``
        records one episode out of every 25. ``offset`` shifts the pattern: ``Every(1000, length=100,
        offset=50)`` records the steps [50, 150), [1050, 1150), and so on.

        Parameters
        ----------
        n : int
            Period, at least 1.
        length : int, default 1
            Steps recorded per period, at least 1.
        offset : int, default 0
            First step recorded.
        tag : str, optional
            Integer tag counted, such as ``'episode'``. Steps are counted without one.

        Raises
        ------
        ValueError
            When a field is not an integer, or ``n`` or ``length`` is below 1.

        See Also
        --------
        At : Records ``length`` steps from each of a set of points.
        Between : Records the steps of a range.
        When : Records steps after a condition holds for the records of other measurements.
        Recorder.record : Records a set of measurements for the next steps.

        Examples
        --------
        >>> Every(10000, length=500)            # 500 steps out of every 10000
        >>> Every(25, tag='episode')            # One episode out of every 25
    """
    n: int
    length: int = 1
    offset: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, 'n', integer(self.n, 'n', lowest=1, owner=type(self).__name__))
        object.__setattr__(self, 'length', integer(self.length, 'length', lowest=1, owner=type(self).__name__))
        object.__setattr__(self, 'offset', integer(self.offset, 'offset', owner=type(self).__name__))

    def covers(self, start: int, stop: int) -> bool:
        first = max(start, self.offset)
        if first >= stop:
            return False
        phase = (first - self.offset) % self.n
        return phase < self.length or first + self.n - phase < stop

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True)
class At(Trigger):
    """
        Trigger recording ``length`` consecutive steps from each of ``points``.

        With ``tag``, it counts the values of the tag instead of steps: ``At((0, 500), length=10,
        tag='episode')`` records the episodes [0, 10) and [500, 510).

        Parameters
        ----------
        points : sequence of int
            First steps recorded.
        length : int, default 1
            Steps recorded from each point, at least 1.
        tag : str, optional
            Integer tag counted, such as ``'episode'``. Steps are counted without one.

        Raises
        ------
        ValueError
            When a point or ``length`` is not an integer, or ``length`` is below 1.

        See Also
        --------
        Every : Records ``length`` steps out of every ``n``.
        Between : Records the steps of a range.
        Recorder.record : Records a set of measurements for the next steps.
    """
    points: tuple[int, ...]
    length: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, 'points', tuple(integer(p, 'points', owner=type(self).__name__) for p in self.points))
        object.__setattr__(self, 'length', integer(self.length, 'length', lowest=1, owner=type(self).__name__))
        object.__setattr__(self, '_sorted', tuple(sorted(self.points)))

    def covers(self, start: int, stop: int) -> bool:
        # A point p records [start, stop) when start - length < p < stop.
        return bisect.bisect_right(self._sorted, start - self.length) < bisect.bisect_left(self._sorted, stop)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True)
class Between(Trigger):
    """
        Trigger recording the steps from ``start`` up to, not including, ``stop``.

        With ``tag``, it counts the values of the tag instead of steps: ``Between(100, 200,
        tag='episode')`` records the episodes 100 to 199.

        Parameters
        ----------
        start : int, default 0
            First step recorded.
        stop : int, optional
            Step after the last recorded. No end when omitted.
        tag : str, optional
            Integer tag counted, such as ``'episode'``. Steps are counted without one.

        Raises
        ------
        ValueError
            When ``start`` or ``stop`` is not an integer, or ``stop`` is not after ``start``.

        See Also
        --------
        Every : Records ``length`` steps out of every ``n``.
        At : Records ``length`` steps from each of a set of points.
        Always : Records every call.
        Recorder.record : Records a set of measurements for the next steps.
    """
    start: int = 0
    stop: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, 'start', integer(self.start, 'start', owner=type(self).__name__))
        if self.stop is not None:
            object.__setattr__(self, 'stop', integer(self.stop, 'stop', owner=type(self).__name__))
            if self.stop <= self.start:
                raise ValueError(f'"stop" must be after "start", got {self.start} and {self.stop}.')

    def covers(self, start: int, stop: int) -> bool:
        return self.start < stop and (self.stop is None or start < self.stop)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True)
class Always(Trigger):
    """
        Trigger recording every call.

        Parameters
        ----------
        tag : str, optional
            Integer tag counted. With one, the calls for which it is not set to an integer are not
            recorded.

        See Also
        --------
        Between : Records the steps of a range.
        Manual : Records nothing on its own. The default trigger.
        Recorder.record : Records a set of measurements for the next steps.
    """

    def covers(self, start: int, stop: int) -> bool:
        return True

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True)
class When(Trigger):
    """
        Trigger recording ``length`` steps after ``condition`` holds for a record of ``watch``.

        ``condition`` receives the records of the measurements ``watch`` as they are written to the
        run, as `Record`. For measurements with a group, it receives one record per group, with its
        summaries, deltas and snapshots. Otherwise, it receives one per call of the model, with its
        traces and rasters. When the condition holds, the recorder records the measurements from its
        current step at its next call.

        Parameters
        ----------
        condition : callable
            ``condition(record) -> bool``, called for every record of ``watch``.
        watch : str
            Name of the measurements whose records are tested, other than those this trigger
            records.
        length : int, default 1
            Steps recorded each time the condition holds.

        Raises
        ------
        ValueError
            When ``condition`` is not callable, ``watch`` is empty, or ``length`` is below 1.

        Notes
        -----
        The condition runs on the writer thread once the records reach the host. The measurements
        are recorded some steps after the record that met the condition, more when the writer lags.
        The ``record`` event of the run gives the step of that record. `Measurements.lookback` keeps
        steps from before it.

        A condition that raises is no longer tested. The error is written to the run as an ``error``
        event.

        `to_dict` writes the qualified name of ``condition``. `from_dict` rebuilds the trigger as
        `Manual`.

        See Also
        --------
        Every : Records ``length`` steps out of every ``n``.
        Manual : Records nothing on its own. The default trigger.
        Recorder.record : Records a set of measurements for the next steps.

        Examples
        --------
        >>> collapse = lambda record: record.summaries['A_excitatory.soma:spikes'].active_fraction < 1e-3
        >>> trigger = When(collapse, watch='summary', length=5000)
        >>> Measurements('collapse', probes, trigger=trigger, lookback=5000)
    """
    condition: tp.Callable[[Record], bool]
    watch: str
    length: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not callable(self.condition):
            raise ValueError(f'"condition" must be callable, got {type(self.condition).__name__}.')
        if not isinstance(self.watch, str) or not self.watch:
            raise ValueError('"watch" must name measurements.')
        object.__setattr__(self, 'length', integer(self.length, 'length', lowest=1, owner=type(self).__name__))

    def covers(self, start: int, stop: int) -> bool:
        return False

    def to_dict(self) -> dict[str, tp.Any]:
        """
            Returns the fields of the trigger as JSON types, ``condition`` by its qualified name.
        """
        name = getattr(self.condition, '__qualname__', repr(self.condition))
        return {'kind': 'When', 'tag': self.tag, 'watch': self.watch, 'length': self.length, 'condition': name}

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
