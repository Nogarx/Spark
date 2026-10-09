spark.recording.presets
=======================

.. py:module:: spark.recording.presets


Functions
---------

.. autoapisummary::

   spark.recording.presets.sample_units
   spark.recording.presets.summary
   spark.recording.presets.activity
   spark.recording.presets.weights
   spark.recording.presets.default


Module Contents
---------------

.. py:function:: sample_units(address, size, count)

   Draws ``count`` units out of ``size`` for the probe at ``address``.

   The draw is seeded by ``address``, and is the same in every process.

   :param address: Probe address.
   :type address: str
   :param size: Number of units of the value.
   :type size: int
   :param count: Number of units to draw.
   :type count: int

   :returns: Sorted flat indices, or None (all units) when ``size`` is not larger than ``count``.
   :rtype: tuple of int or None


.. py:function:: summary(model)

   Returns probes of the population statistics of every soma and set of synapses.

   * Every spike output of a soma: ``ACTIVE_FRACTION`` and ``INACTIVE_UNIT_FRACTION``.
   * The membrane potential of a soma: ``MEAN``, ``STD``, ``MIN`` and ``MAX``.
   * The kernel of a set of synapses: the ``NORM`` and ``MEAN_ABS`` of its change over each
     group.

   :param model: A built model.
   :type model: Controller

   :returns: `SummaryProbe` and `DeltaProbe` probes. The measurements holding them need a group.
   :rtype: tuple of Probe

   .. seealso::

      :py:obj:`activity`
          Traces and rasters of a model on every step.

      :py:obj:`weights`
          Snapshots of the weights of a model.


.. py:function:: activity(model, trace_units = 64, raster_units = 4096)

   Returns probes tracing the inputs, interfaces and somas of a model on every step.

   * The inputs of the model: a trace of each, whole.
   * Input interfaces: a raster of each spike output.
   * Control and output interfaces: a trace of each output that does not carry spikes.
   * Somas: a raster of each spike output, and a trace of the membrane potential.

   :param model: A built model.
   :type model: Controller
   :param trace_units: Units traced per soma or interface output, drawn by `sample_units`.
   :type trace_units: int, default 64
   :param raster_units: Units kept per raster, drawn by `sample_units`.
   :type raster_units: int, default 4096

   :returns: `TraceProbe` and `RasterProbe` probes.
   :rtype: tuple of Probe

   .. seealso::

      :py:obj:`summary`
          Population statistics of every soma and set of synapses.

      :py:obj:`weights`
          Snapshots of the weights of a model.


.. py:function:: weights(model)

   Returns probes of the weights of every set of synapses.

   Each probe is a `SnapshotProbe` of a whole kernel, after the last step of each group.

   :param model: A built model.
   :type model: Controller

   :returns: `SnapshotProbe` probes. The measurements holding them need a group.
   :rtype: tuple of Probe

   .. seealso::

      :py:obj:`summary`
          Population statistics of every soma and set of synapses.

      :py:obj:`activity`
          Traces and rasters of a model on every step.


.. py:function:: default(model, *, summary_group = 1000, activity_every = 100000, activity_length = 1000, weights_every = 100000)

   Returns measurements for any model, recorded by triggers.

   * ``'summary'``: the probes of `summary`, recorded on every step, one record per
     ``summary_group`` steps.
   * ``'activity'``: the probes of `activity`, recorded for ``activity_length`` steps out of
     every ``activity_every``.
   * ``'weights'``: the probes of `weights`, one record per ``weights_every`` steps. Only the
     call holding the last step of each group is recorded.

   Measurements without probes are left out.

   :param model: A built model.
   :type model: Controller
   :param summary_group: Steps per record of ``'summary'``.
   :type summary_group: int, default 1000
   :param activity_every: Period of ``'activity'``, in steps.
   :type activity_every: int, default 100_000
   :param activity_length: Steps of ``'activity'`` recorded per period.
   :type activity_length: int, default 1000
   :param weights_every: Steps per record of ``'weights'``.
   :type weights_every: int, default 100_000

   :rtype: tuple of Measurements

   .. rubric:: Notes

   Each set of measurements recorded together compiles once. At most four sets occur.

   .. seealso::

      :py:obj:`Recorder`
          Uses these measurements when given none.

      :py:obj:`Measurements`
          A named set of probes recorded together.


