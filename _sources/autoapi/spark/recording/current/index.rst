spark.recording.current
=======================

.. py:module:: spark.recording.current


Functions
---------

.. autoapisummary::

   spark.recording.current.open_recorder
   spark.recording.current.log
   spark.recording.current.event
   spark.recording.current.tag
   spark.recording.current.raw
   spark.recording.current.record


Module Contents
---------------

.. py:function:: open_recorder()

   Returns the open recorder of the calling thread, or None when no recorder is open.

   :raises RuntimeError: When the thread opened no recorder, and several are open in other threads.


.. py:function:: log(values = None, *, step = None, tag = None, **scalars)

   Writes scalars held on the host to the open recorder, as `Recorder.log`.

   Does nothing when no recorder is open.

   :param values: Scalars by name.
   :type values: dict of str to float, optional
   :param step: Step of the scalars. The current step by default.
   :type step: int, optional
   :param tag: Name of the integer tag the scalars are logged per. Per step without one.
   :type tag: str, optional
   :param \*\*scalars: Scalars by name, added to ``values``.
   :type \*\*scalars: float

   :raises RuntimeError: When the thread opened no recorder, and several are open in other threads.

   .. rubric:: Examples

   >>> spark.recording.log({'episode/steps': 212}, tag='episode')


.. py:function:: event(kind, *, step = None, **payload)

   Writes an event to the open recorder, as `Recorder.event`.

   Does nothing when no recorder is open.

   :param kind: Kind of the event.
   :type kind: str
   :param step: Step of the event. The current step by default.
   :type step: int, optional
   :param \*\*payload: Payload of the event, written as JSON.

   :raises RuntimeError: When the thread opened no recorder, and several are open in other threads.

   .. rubric:: Examples

   >>> spark.recording.event('episode_end', outcome='fell')


.. py:function:: tag(**tags)

   Sets tags on the timeline of the open recorder, as `Recorder.tag`.

   Does nothing when no recorder is open.

   :param \*\*tags: Values by tag name.

   :raises RuntimeError: When the thread opened no recorder, and several are open in other threads.

   .. rubric:: Examples

   >>> spark.recording.tag(episode=3)


.. py:function:: raw(name, frame, *, step = None)

   Keeps a frame of a raw stream in the open recorder, as `Recorder.raw`.

   Does nothing when no recorder is open.

   :param name: Name of the raw stream, as declared by `Measurements`.
   :type name: str
   :param frame: Array of numbers or bools, such as an observation.
   :type frame: array-like
   :param step: Step of the frame. The current step by default.
   :type step: int, optional

   :raises RuntimeError: When the thread opened no recorder, and several are open in other threads.

   :Warns: **RecordingWarning** -- When the open recorder has no measurements declaring ``name``, or the frame is invalid.
           The frame is dropped.

   .. rubric:: Examples

   >>> spark.recording.raw('env/observation', observation)


.. py:function:: record(name, steps = 1)

   Records the measurements ``name`` for the next ``steps`` steps in the open recorder, as
   `Recorder.record`.

   Does nothing when no recorder is open.

   :param name: Name of the measurements.
   :type name: str
   :param steps: Steps to record them for.
   :type steps: int, default 1

   :raises RuntimeError: When the thread opened no recorder, and several are open in other threads.

   :Warns: **RecordingWarning** -- When the open recorder has no measurements ``name``. Nothing is recorded.

   .. rubric:: Examples

   >>> spark.recording.record('episode')


