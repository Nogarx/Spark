spark.recording.triggers
========================

.. py:module:: spark.recording.triggers


Classes
-------

.. autoapisummary::

   spark.recording.triggers.Trigger
   spark.recording.triggers.Manual
   spark.recording.triggers.Every
   spark.recording.triggers.At
   spark.recording.triggers.Between
   spark.recording.triggers.Always
   spark.recording.triggers.When


Module Contents
---------------

.. py:class:: Trigger

   Bases: :py:obj:`abc.ABC`


   Base class for triggers.

   A trigger decides which calls of the model record a set of measurements, in addition to
   all the explicit recordings invoked by `Recorder.record`.

   A trigger counts steps, or the values of ``tag``. Tags are set by hand with `Recorder.tag`,
   such as ``'episode'``. Without a tag, the trigger counts the steps of the run, which the
   recorder advances on its own.

   :param tag: Integer tag counted, such as ``'episode'``. Steps are counted without one.
   :type tag: str, optional

   :raises ValueError: When ``tag`` is not a non-empty string, or is ``'step'``.

   .. seealso::

      :py:obj:`Every`
          Records ``length`` steps out of every ``n``.

      :py:obj:`At`
          Records ``length`` steps from each of a set of points.

      :py:obj:`Between`
          Records the steps of a range.

      :py:obj:`Always`
          Records every call.

      :py:obj:`Manual`
          Records nothing on its own. The default trigger.

      :py:obj:`When`
          Records steps after a condition holds for the records of other measurements.

      :py:obj:`Recorder.record`
          Records a set of measurements for the next steps.


   .. py:attribute:: tag
      :type:  str | None
      :value: None



   .. py:method:: __post_init__()


   .. py:method:: recorded(counters, spans = None)

      Tests whether a call of the model is recorded.

      :param counters: Value of every counter at the start of the call.
      :type counters: dict of str to int
      :param spans: Counts of each counter the call covers. One for a counter not given.
      :type spans: dict of str to int, optional

      :returns: False when the counter of the trigger is not in ``counters``.
      :rtype: bool



   .. py:method:: covers(start, stop)
      :abstractmethod:


      Tests whether any count in ``[start, stop)`` is recorded.

      :param start: First count.
      :type start: int
      :param stop: Count after the last.
      :type stop: int

      :rtype: bool



   .. py:method:: to_dict()

      Returns the fields of the trigger and its class name as JSON types.

      :returns: The fields, and the class name under ``'kind'``.
      :rtype: dict



   .. py:method:: from_dict(data)
      :classmethod:


      Rebuilds a trigger from `to_dict`.

      `When` triggers and triggers defined outside this module come back as `Manual`, with
      their ``tag``. The field ``'unit'`` of earlier runs is read as ``tag``, ``'step'`` as no
      tag.

      :param data: Fields of a trigger, and its class name under ``'kind'``.
      :type data: dict

      :returns: A trigger of the class named by ``'kind'``.
      :rtype: Trigger

      :raises ValueError: When called on a subclass, and ``'kind'`` names another class.



.. py:class:: Manual

   Bases: :py:obj:`Trigger`


   Default trigger of `Measurements`.

   Measurements with this trigger require an explicit call to `Recorder.record`.

   .. seealso::

      :py:obj:`Always`
          Records every call.

      :py:obj:`When`
          Records steps after a condition holds for the records of other measurements.

      :py:obj:`Recorder.record`
          Records a set of measurements for the next steps.

      :py:obj:`Run.record`
          Asks the recorder writing a run to record a set of measurements.

   .. rubric:: Examples

   >>> measurements = Measurements('episode', probes)      # Manual by default
   >>> recorder.record('episode')                          # Records the next call of the model


   .. py:method:: covers(start, stop)

      Tests whether any count in ``[start, stop)`` is recorded.

      :param start: First count.
      :type start: int
      :param stop: Count after the last.
      :type stop: int

      :rtype: bool



.. py:class:: Every

   Bases: :py:obj:`Trigger`


   Trigger recording ``length`` consecutive steps out of every ``n``.

   With ``tag``, it counts the values of the tag instead of steps: ``Every(25, tag='episode')``
   records one episode out of every 25. ``offset`` shifts the pattern: ``Every(1000, length=100,
   offset=50)`` records the steps [50, 150), [1050, 1150), and so on.

   :param n: Period, at least 1.
   :type n: int
   :param length: Steps recorded per period, at least 1.
   :type length: int, default 1
   :param offset: First step recorded.
   :type offset: int, default 0
   :param tag: Integer tag counted, such as ``'episode'``. Steps are counted without one.
   :type tag: str, optional

   :raises ValueError: When a field is not an integer, or ``n`` or ``length`` is below 1.

   .. seealso::

      :py:obj:`At`
          Records ``length`` steps from each of a set of points.

      :py:obj:`Between`
          Records the steps of a range.

      :py:obj:`When`
          Records steps after a condition holds for the records of other measurements.

      :py:obj:`Recorder.record`
          Records a set of measurements for the next steps.

   .. rubric:: Examples

   >>> Every(10000, length=500)            # 500 steps out of every 10000
   >>> Every(25, tag='episode')            # One episode out of every 25


   .. py:attribute:: n
      :type:  int


   .. py:attribute:: length
      :type:  int
      :value: 1



   .. py:attribute:: offset
      :type:  int
      :value: 0



   .. py:method:: __post_init__()


   .. py:method:: covers(start, stop)

      Tests whether any count in ``[start, stop)`` is recorded.

      :param start: First count.
      :type start: int
      :param stop: Count after the last.
      :type stop: int

      :rtype: bool



.. py:class:: At

   Bases: :py:obj:`Trigger`


   Trigger recording ``length`` consecutive steps from each of ``points``.

   With ``tag``, it counts the values of the tag instead of steps: ``At((0, 500), length=10,
   tag='episode')`` records the episodes [0, 10) and [500, 510).

   :param points: First steps recorded.
   :type points: sequence of int
   :param length: Steps recorded from each point, at least 1.
   :type length: int, default 1
   :param tag: Integer tag counted, such as ``'episode'``. Steps are counted without one.
   :type tag: str, optional

   :raises ValueError: When a point or ``length`` is not an integer, or ``length`` is below 1.

   .. seealso::

      :py:obj:`Every`
          Records ``length`` steps out of every ``n``.

      :py:obj:`Between`
          Records the steps of a range.

      :py:obj:`Recorder.record`
          Records a set of measurements for the next steps.


   .. py:attribute:: points
      :type:  tuple[int, ...]


   .. py:attribute:: length
      :type:  int
      :value: 1



   .. py:method:: __post_init__()


   .. py:method:: covers(start, stop)

      Tests whether any count in ``[start, stop)`` is recorded.

      :param start: First count.
      :type start: int
      :param stop: Count after the last.
      :type stop: int

      :rtype: bool



.. py:class:: Between

   Bases: :py:obj:`Trigger`


   Trigger recording the steps from ``start`` up to, not including, ``stop``.

   With ``tag``, it counts the values of the tag instead of steps: ``Between(100, 200,
   tag='episode')`` records the episodes 100 to 199.

   :param start: First step recorded.
   :type start: int, default 0
   :param stop: Step after the last recorded. No end when omitted.
   :type stop: int, optional
   :param tag: Integer tag counted, such as ``'episode'``. Steps are counted without one.
   :type tag: str, optional

   :raises ValueError: When ``start`` or ``stop`` is not an integer, or ``stop`` is not after ``start``.

   .. seealso::

      :py:obj:`Every`
          Records ``length`` steps out of every ``n``.

      :py:obj:`At`
          Records ``length`` steps from each of a set of points.

      :py:obj:`Always`
          Records every call.

      :py:obj:`Recorder.record`
          Records a set of measurements for the next steps.


   .. py:attribute:: start
      :type:  int
      :value: 0



   .. py:attribute:: stop
      :type:  int | None
      :value: None



   .. py:method:: __post_init__()


   .. py:method:: covers(start, stop)

      Tests whether any count in ``[start, stop)`` is recorded.

      :param start: First count.
      :type start: int
      :param stop: Count after the last.
      :type stop: int

      :rtype: bool



.. py:class:: Always

   Bases: :py:obj:`Trigger`


   Trigger recording every call.

   :param tag: Integer tag counted. With one, the calls for which it is not set to an integer are not
               recorded.
   :type tag: str, optional

   .. seealso::

      :py:obj:`Between`
          Records the steps of a range.

      :py:obj:`Manual`
          Records nothing on its own. The default trigger.

      :py:obj:`Recorder.record`
          Records a set of measurements for the next steps.


   .. py:method:: covers(start, stop)

      Tests whether any count in ``[start, stop)`` is recorded.

      :param start: First count.
      :type start: int
      :param stop: Count after the last.
      :type stop: int

      :rtype: bool



.. py:class:: When

   Bases: :py:obj:`Trigger`


   Trigger recording ``length`` steps after ``condition`` holds for a record of ``watch``.

   ``condition`` receives the records of the measurements ``watch`` as they are written to the
   run, as `Record`. For measurements with a group, it receives one record per group, with its
   summaries, deltas and snapshots. Otherwise, it receives one per call of the model, with its
   traces and rasters. When the condition holds, the recorder records the measurements from its
   current step at its next call.

   :param condition: ``condition(record) -> bool``, called for every record of ``watch``.
   :type condition: callable
   :param watch: Name of the measurements whose records are tested, other than those this trigger
                 records.
   :type watch: str
   :param length: Steps recorded each time the condition holds.
   :type length: int, default 1

   :raises ValueError: When ``condition`` is not callable, ``watch`` is empty, or ``length`` is below 1.

   .. rubric:: Notes

   The condition runs on the writer thread once the records reach the host. The measurements
   are recorded some steps after the record that met the condition, more when the writer lags.
   The ``record`` event of the run gives the step of that record. `Measurements.lookback` keeps
   steps from before it.

   A condition that raises is no longer tested. The error is written to the run as an ``error``
   event.

   `to_dict` writes the qualified name of ``condition``. `from_dict` rebuilds the trigger as
   `Manual`.

   .. seealso::

      :py:obj:`Every`
          Records ``length`` steps out of every ``n``.

      :py:obj:`Manual`
          Records nothing on its own. The default trigger.

      :py:obj:`Recorder.record`
          Records a set of measurements for the next steps.

   .. rubric:: Examples

   >>> collapse = lambda record: record.summaries['A_excitatory.soma:spikes'].active_fraction < 1e-3
   >>> trigger = When(collapse, watch='summary', length=5000)
   >>> Measurements('collapse', probes, trigger=trigger, lookback=5000)


   .. py:attribute:: condition
      :type:  Callable[[spark.recording.records.Record], bool]


   .. py:attribute:: watch
      :type:  str


   .. py:attribute:: length
      :type:  int
      :value: 1



   .. py:method:: __post_init__()


   .. py:method:: covers(start, stop)

      Tests whether any count in ``[start, stop)`` is recorded.

      :param start: First count.
      :type start: int
      :param stop: Count after the last.
      :type stop: int

      :rtype: bool



   .. py:method:: to_dict()

      Returns the fields of the trigger as JSON types, ``condition`` by its qualified name.



