spark.recording.measurements
============================

.. py:module:: spark.recording.measurements


Classes
-------

.. autoapisummary::

   spark.recording.measurements.Measurements


Functions
---------

.. autoapisummary::

   spark.recording.measurements.scalar_key
   spark.recording.measurements.canonical_scalar_key
   spark.recording.measurements.scalar_key_forms
   spark.recording.measurements.merge_probes


Module Contents
---------------

.. py:class:: Measurements

   Named set of probes recorded together.

   The steps recorded are those the trigger or `Recorder.record` ask for. With ``group``, the
   summaries, snapshots and deltas give one record per group of steps.

   :param name: Name of the measurements and of their files. Letters, digits, ``_``, ``.`` and ``-``,
                starting with a letter or a digit.
   :type name: str
   :param probes: One probe per address and mode.
   :type probes: sequence of Probe
   :param group: How the steps of the run are split into groups for the `SummaryProbe`, `SnapshotProbe`
                 and `DeltaProbe` probes, which give one record per group. Required with such probes.

                 * A number of steps ``n``: groups of ``n`` steps on the steps of the run, ``[0, n)``,
                   ``[n, 2n)``, and so on, whatever step the recording starts at.
                 * The name of a tag set with `Recorder.tag`, such as ``'episode'``: a group lasts while
                   the tag keeps its value, and a new group starts each time the tag takes a different
                   value. A tag going from 1 to 2 and back to 1 gives three groups. The steps before the
                   tag is first set are one group.
   :type group: int or str, optional
   :param trigger: Steps recorded without a call to `Recorder.record`. `Manual` records none.
   :type trigger: Trigger, default Manual()
   :param raw: Names of the raw streams given to `Recorder.raw` that are written with these
               measurements.
   :type raw: sequence of str, optional
   :param lookback: Steps kept on the device before a recorded step, and written ahead of it.
   :type lookback: int, default 0
   :param views: How the viewer draws a value, by probe address or raw stream name, as
                 ``{'env/frame': {'kind': 'image', 'shape': [84, 84, 3]}}``. Written to the run.
   :type views: dict, optional

   :raises TypeError: When a probe is not a `Probe`, or ``trigger`` is not a `Trigger`.
   :raises ValueError: When the name is invalid, probes share a key, ``group`` is invalid or missing for
       grouped probes, a probe has a group other than ``group``, or ``lookback`` is negative.

   .. rubric:: Notes

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

   .. seealso::

      :py:obj:`Probe`
          A value read from a model, and how it is recorded.

      :py:obj:`Recorder.record`
          Records a set of measurements for the next steps.

      :py:obj:`Trigger`
          Base class for triggers, which ask for steps on their own.

      :py:obj:`presets.default`
          Measurements for any model, recorded by triggers.

   .. rubric:: Examples

   >>> Measurements('episode', probes, group='episode', raw=('env/frame',))
   >>> summary = spark.recording.presets.summary(brain)
   >>> Measurements('summary', summary, group=1000, trigger=Always())
   >>> activity = spark.recording.presets.activity(brain)
   >>> Measurements('activity', activity, trigger=Every(100_000, length=1000))


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: probes
      :type:  tuple[spark.recording.probe.Probe, ...]


   .. py:attribute:: group
      :type:  int | str | None
      :value: None



   .. py:attribute:: trigger
      :type:  spark.recording.triggers.Trigger


   .. py:attribute:: raw
      :type:  tuple[str, ...]
      :value: ()



   .. py:attribute:: lookback
      :type:  int
      :value: 0



   .. py:attribute:: views
      :type:  dict[str, dict]


   .. py:method:: __post_init__()


   .. py:method:: to_dict()

      Returns the fields of the measurements as JSON types.

      :returns: Keyword arguments of `Measurements`, with probes and trigger as their own `to_dict`.
      :rtype: dict



   .. py:method:: from_dict(data)
      :classmethod:


      Rebuilds measurements from `to_dict`.

      A `When` trigger comes back as `Manual`.

      :param data: Fields of the measurements, as `to_dict` gives them.
      :type data: dict

      :rtype: Measurements



.. py:function:: scalar_key(measurements, probe_key, reduction)

   Returns the name of the scalar series of one reduction of a probe.

   The reductions of summaries and deltas share no name, so the address of the probe and the
   reduction name one series of the measurements.

   :param measurements: Name of the measurements writing the series.
   :type measurements: str
   :param probe_key: Key of the probe, as `Probe.key`, or its address.
   :type probe_key: str
   :param reduction: Name of the reduction.
   :type reduction: str

   :returns: ``<measurements>/<address>/<reduction>``.
   :rtype: str


.. py:function:: canonical_scalar_key(name)

   Returns the name of a scalar series in the form `scalar_key`.


.. py:function:: scalar_key_forms(name)

   Returns the names a scalar series may be stored under: ``name``.


.. py:function:: merge_probes(probes)

   Merges probes sharing a key.

   Probes sharing a key become one probe. A `SummaryProbe` or a `DeltaProbe` takes the union of
   their reductions, in the order they are first given.

   :param probes: Probes to merge.
   :type probes: iterable of Probe

   :returns: One probe per key, ordered by key.
   :rtype: tuple of Probe

   :raises ValueError: When probes sharing a key differ in their group, or both count a histogram and differ in
       ``bins`` or ``range``, or have no reductions and are not equal.


