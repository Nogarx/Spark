#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
if tp.TYPE_CHECKING:
    from spark.recording.recorder import Recorder

import threading
from spark.core.recording_hooks import OPEN_RECORDERS

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def open_recorder() -> Recorder | None:
    """
        Returns the open recorder of the calling thread, or None when no recorder is open.

        Raises
        ------
        RuntimeError
            When the thread opened no recorder, and several are open in other threads.
    """
    if not OPEN_RECORDERS:
        return None
    thread = threading.current_thread()
    opened = [recorder for recorder, opener in OPEN_RECORDERS.items() if opener is thread]
    if opened:
        return opened[-1]
    if len(OPEN_RECORDERS) == 1:
        return next(iter(OPEN_RECORDERS))
    raise RuntimeError(
        f'This thread opened no recorder, and {len(OPEN_RECORDERS)} are open in other threads. The calls of a thread go to '
        f'the recorder it opened: open the recorder in the thread running the loop. Open: '
        f'{", ".join(str(r.path) for r in OPEN_RECORDERS)}.'
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def log(values: dict[str, tp.Any] | None = None, *, step: int | None = None, tag: str | None = None, **scalars: tp.Any) -> None:
    """
        Writes scalars held on the host to the open recorder, as `Recorder.log`.

        Does nothing when no recorder is open.

        Parameters
        ----------
        values : dict of str to float, optional
            Scalars by name.
        step : int, optional
            Step of the scalars. The current step by default.
        tag : str, optional
            Name of the integer tag the scalars are logged per. Per step without one.
        **scalars : float
            Scalars by name, added to ``values``.

        Raises
        ------
        RuntimeError
            When the thread opened no recorder, and several are open in other threads.

        Examples
        --------
        >>> spark.recording.log({'episode/steps': 212}, tag='episode')
    """
    recorder = open_recorder()
    if recorder is not None:
        recorder.log(values, step=step, tag=tag, **scalars)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def event(kind: str, *, step: int | None = None, **payload: tp.Any) -> None:
    """
        Writes an event to the open recorder, as `Recorder.event`.

        Does nothing when no recorder is open.

        Parameters
        ----------
        kind : str
            Kind of the event.
        step : int, optional
            Step of the event. The current step by default.
        **payload
            Payload of the event, written as JSON.

        Raises
        ------
        RuntimeError
            When the thread opened no recorder, and several are open in other threads.

        Examples
        --------
        >>> spark.recording.event('episode_end', outcome='fell')
    """
    recorder = open_recorder()
    if recorder is not None:
        recorder.event(kind, step=step, **payload)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def tag(**tags: tp.Any) -> None:
    """
        Sets tags on the timeline of the open recorder, as `Recorder.tag`.

        Does nothing when no recorder is open.

        Parameters
        ----------
        **tags
            Values by tag name.

        Raises
        ------
        RuntimeError
            When the thread opened no recorder, and several are open in other threads.

        Examples
        --------
        >>> spark.recording.tag(episode=3)
    """
    recorder = open_recorder()
    if recorder is not None:
        recorder.tag(**tags)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def raw(name: str, frame: tp.Any, *, step: int | None = None) -> None:
    """
        Keeps a frame of a raw stream in the open recorder, as `Recorder.raw`.

        Does nothing when no recorder is open.

        Parameters
        ----------
        name : str
            Name of the raw stream, as declared by `Measurements`.
        frame : array-like
            Array of numbers or bools, such as an observation.
        step : int, optional
            Step of the frame. The current step by default.

        Raises
        ------
        RuntimeError
            When the thread opened no recorder, and several are open in other threads.

        Warns
        -----
        RecordingWarning
            When the open recorder has no measurements declaring ``name``, or the frame is invalid.
            The frame is dropped.

        Examples
        --------
        >>> spark.recording.raw('env/observation', observation)
    """
    recorder = open_recorder()
    if recorder is not None:
        recorder.raw(name, frame, step=step)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def record(name: str, steps: int = 1) -> None:
    """
        Records the measurements ``name`` for the next ``steps`` steps in the open recorder, as
        `Recorder.record`.

        Does nothing when no recorder is open.

        Parameters
        ----------
        name : str
            Name of the measurements.
        steps : int, default 1
            Steps to record them for.

        Raises
        ------
        RuntimeError
            When the thread opened no recorder, and several are open in other threads.

        Warns
        -----
        RecordingWarning
            When the open recorder has no measurements ``name``. Nothing is recorded.

        Examples
        --------
        >>> spark.recording.record('episode')
    """
    recorder = open_recorder()
    if recorder is not None:
        recorder.record(name, steps)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
