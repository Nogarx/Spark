spark.recording.reduce
======================

.. py:module:: spark.recording.reduce


Attributes
----------

.. autoapisummary::

   spark.recording.reduce.Layout
   spark.recording.reduce.HISTOGRAM_LOW_BITS


Classes
-------

.. autoapisummary::

   spark.recording.reduce.Start
   spark.recording.reduce.StepRecords
   spark.recording.reduce.Packed


Functions
---------

.. autoapisummary::

   spark.recording.reduce.moduli
   spark.recording.reduce.start_of
   spark.recording.reduce.may_split
   spark.recording.reduce.warmup_starts
   spark.recording.reduce.slots
   spark.recording.reduce.ends
   spark.recording.reduce.step_value
   spark.recording.reduce.pack_step
   spark.recording.reduce.init_spaced
   spark.recording.reduce.write_spaced
   spark.recording.reduce.first_kept
   spark.recording.reduce.read_boundary
   spark.recording.reduce.init_captures
   spark.recording.reduce.capture
   spark.recording.reduce.uses_rows
   spark.recording.reduce.accumulated_fields
   spark.recording.reduce.init_accumulators
   spark.recording.reduce.accumulate
   spark.recording.reduce.merge_summaries
   spark.recording.reduce.finalize
   spark.recording.reduce.held
   spark.recording.reduce.group_values
   spark.recording.reduce.complete
   spark.recording.reduce.pack
   spark.recording.reduce.merge_packed


Module Contents
---------------

.. py:data:: Layout

   Layout of a byte buffer. One ``(key, dtype, shape, offset, size)`` entry per value, with
   ``offset`` and ``size`` in bytes.

.. py:data:: HISTOGRAM_LOW_BITS
   :value: 30


   Bits of the low word of the histogram counts kept over a call. The counts are two int32 words,
   ``high * 2 ** 30 + low``, combined as int64 on the host.

.. py:function:: moduli(probes)

   Returns the group sizes and strides of ``probes``, in steps.

   `Start` holds the first step of a call modulo each of them. `recorded_scan` aligns groups and
   strides on the steps of the run with these phases.

   :param probes: Probes of a call.
   :type probes: tuple of Probe

   :returns: The distinct group sizes of the probes grouped by steps, and the strides above 1 of
             traces and rasters, sorted.
   :rtype: tuple of int


.. py:class:: Start(phases, split = True)

   Position of a recorded call on the steps of the run.

   A pytree whose leaf is ``phases`` and whose static part is ``split``. `start_of` builds it.

   :param phases: First step of the call modulo each of `moduli`, as int32.
   :type phases: array
   :param split: Whether the call may cross the end of a group of steps.
   :type split: bool, default True

   .. rubric:: Notes

   Calls that differ only in ``phases`` share one compiled program. Each value of ``split``
   compiles to its own program. With ``split`` False, a summary keeps one set of statistics,
   and snapshots and deltas keep no values at the ends of groups within the call.


   .. py:attribute:: phases


   .. py:attribute:: split
      :value: True



   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(split, children)
      :classmethod:



.. py:function:: start_of(probes, step, steps = None)

   Returns the position of a recorded call in the groups and strides of ``probes``.

   :param probes: Probes of the call.
   :type probes: tuple of Probe
   :param step: Step of the run at which the call starts.
   :type step: int
   :param steps: Steps of the call. Without it, the call may cross the end of a group.
   :type steps: int, optional

   :returns: Phases of ``step`` modulo each of `moduli`, and whether the call crosses the end of a
             group of steps.
   :rtype: Start

   .. rubric:: Notes

   The phases are computed on the host from ``step`` as a Python integer, exact for any step.


.. py:function:: may_split(probes, step, steps)

   Returns whether calls of ``steps`` steps may cross the end of a group of ``probes``.

   The calls considered start at the steps ``step + k * steps``, for every ``k >= 0``. Calls
   whose length divides every group never cross one when ``step`` is a multiple of that length.

   :param probes: Probes of the calls.
   :type probes: tuple of Probe
   :param step: Step of the run at which the first call starts.
   :type step: int
   :param steps: Steps of every call.
   :type steps: int

   :returns: Whether one of those calls crosses the end of a group of steps.
   :rtype: bool


.. py:function:: warmup_starts(probes, step, steps)

   Returns the starts for which calls of ``steps`` steps from ``step`` compile apart.

   A call that may cross the end of a group compiles apart from one that does not
   (`Start.split`). The second is given only when such calls may cross one (`may_split`).

   :param probes: Probes of the calls.
   :type probes: tuple of Probe
   :param step: Step of the run at which the first call starts.
   :type step: int
   :param steps: Steps of every call.
   :type steps: int

   :returns: The starts to compile the calls with. Their phases are those of step 0.
   :rtype: tuple of Start


.. py:function:: slots(probe, steps)

   Returns the number of groups a call can touch, for a summary grouped by steps.

   :param probe: Probe of the call.
   :type probe: Probe
   :param steps: Steps of the call.
   :type steps: int

   :returns: Largest number of groups a call of ``steps`` steps holds steps of, over all first steps.
             1 for other probes.
   :rtype: int


.. py:function:: ends(probe, steps)

   Returns the number of group ends within a call, for a snapshot or delta grouped by steps.

   :param probe: Probe of the call.
   :type probe: Probe
   :param steps: Steps of the call.
   :type steps: int

   :returns: Largest number of ends of groups at the steps of a call of ``steps`` steps, over all
             first steps. 0 for other probes.
   :rtype: int


.. py:function:: step_value(probe, value)

   Returns what one step contributes to the record of a per-step probe.

   :param probe: A probe read on every step.
   :type probe: Probe
   :param value: The value read on this step.
   :type value: SparkPayload or Variable or array

   :returns: One of the following, by mode.

             * ``summary``: the value, flat.
             * ``raster``: whether each entry is nonzero, flat.
             * ``trace``: the value, flat when ``units`` is given.

             Traces and rasters keep only their ``units`` when these are selected on every step
             (`SETTINGS.select_per_step`).
   :rtype: array

   :raises ValueError: When the probe is not read per step, a summary has no entries, a histogram has
       ``2 ** 31`` entries or more, or ``units`` asks for an index past the entries.

   .. rubric:: Notes

   A float16 or bfloat16 value of a raster is rounded to its dtype before the test
   (`_rounded`).


.. py:class:: StepRecords(row, layout)

   Values of one step that a call stacks, packed into one byte row.

   `recorded_scan` packs the values of the traces, the rasters, and the summaries whose histogram
   is counted after the call (`uses_rows`). A pytree whose only leaf is the row. The layout is
   static. Returned from the body of ``jax.lax.scan``, the rows of all the steps stack into one
   output of shape ``(steps, bytes)``.

   :param row: uint8 row, or rows stacked along leading axes.
   :type row: array
   :param layout: The values of the row, along its last axis.
   :type layout: Layout

   .. rubric:: Notes

   Bool values take one bit per entry.


   .. py:attribute:: row


   .. py:attribute:: layout


   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(layout, children)
      :classmethod:



   .. py:method:: unpack()

      Unpacks the row into its values.

      :returns: Values by `Probe.key`, with the leading axes of ``row`` kept.
      :rtype: dict of str to array



.. py:function:: pack_step(values)

   Packs the values of one step into a `StepRecords`.

   :param values: Values of the step, by `Probe.key`.
   :type values: dict of str to array

   :returns: The values in one row, ordered by dtype, wider types first. ``values`` unchanged when
             empty.
   :rtype: StepRecords or dict


.. py:function:: init_spaced(probes, values, steps)

   Creates the buffers of the traces and rasters that write only the steps they keep.

   These are the traces and rasters with a ``stride`` above 1 whose values over the call would take
   more than `SETTINGS.spaced_rows_limit` bytes. The buffers and a step counter are carried through
   the steps of the call. The other traces and rasters with a stride are stacked with the rows of
   every step and strided after the call.

   :param probes: Probes of the call.
   :type probes: tuple of Probe
   :param values: What `ProbeContext.values` gives on one step, or its shapes.
   :type values: dict of str to array or ShapeDtypeStruct
   :param steps: Steps of the call.
   :type steps: int

   :returns: ``(step, buffers)``. ``step`` is the int32 step counter. ``buffers`` holds, by
             `Probe.key`, one row per step the call can keep and a spare row. An empty tuple when no
             probe needs a buffer.
   :rtype: tuple


.. py:function:: write_spaced(probes, spaced, values, firsts = None)

   Writes the values of one step to the buffers of `init_spaced`.

   Step ``i`` of the call writes row ``ceil((i - first) / stride)``, where ``first`` is the
   first step the stride keeps. A step kept is the last to write its row. The steps after the
   last one kept write the next row. That row is the spare row, or the last row of the record
   when the call keeps fewer than ``ceil(steps / stride)`` steps.

   :param probes: Probes of the call.
   :type probes: tuple of Probe
   :param spaced: As given by `init_spaced`, or by the previous step.
   :type spaced: tuple
   :param values: What `ProbeContext.values` gave on this step.
   :type values: dict of str to array
   :param firsts: First step kept by each stride, as `first_kept` gives it. 0 for a stride not in it.
   :type firsts: dict of int to int or array, optional

   :returns: The step counter, advanced by one, and the updated buffers. ``spaced`` unchanged when
             empty.
   :rtype: tuple


.. py:function:: first_kept(stride, phase)

   Returns the index within a call of the first step a stride keeps.

   A stride keeps the steps ``t`` of the run with ``t % stride == 0``.

   :param stride: Stride of a trace or raster.
   :type stride: int
   :param phase: First step of the call modulo ``stride``.
   :type phase: int or array

   :returns: Index of that step within the call, from 0. It may lie past the end of a short call.
   :rtype: int or array


.. py:function:: read_boundary(model, probes)

   Reads the values of the snapshot and delta probes from the current state of the model.

   `recorded_scan` reads them before and after the steps of a call, and passes them to `finalize`
   as ``start`` and ``end``. It also reads those of `capture` after every step.

   :param model: The model.
   :type model: Controller
   :param probes: Probes of the call. Probes read on every step are skipped.
   :type probes: tuple of Probe

   :returns: One entry per ``snapshot`` or ``delta`` probe, by `Probe.key`.
   :rtype: dict of str to array

   :raises ValueError: When a module path is not found, or the value of a delta has no entries.
   :raises TypeError: When an attribute holds no array.


.. py:function:: init_captures(probes, values, steps)

   Creates the buffers of the values at the ends of groups within a call.

   One buffer per snapshot or delta grouped by steps. The buffers are carried through the steps
   of the call.

   :param probes: Probes of the call.
   :type probes: tuple of Probe
   :param values: What `read_boundary` gives, or its shapes.
   :type values: dict of str to array or ShapeDtypeStruct
   :param steps: Steps of the call.
   :type steps: int

   :returns: Buffers by `Probe.key`, of shape ``(ends + 1, ...)``. One row per end of a group the
             call can hold (`ends`), and a spare row the other steps write. A snapshot keeps only its
             ``units``.
   :rtype: dict of str to array


.. py:function:: capture(probes, buffers, values, index, start)

   Writes the values after step ``index`` of a call to the buffers of `init_captures`.

   The step ending the ``e``-th group of the call, from 0, writes row ``e``. The other steps
   write the spare row.

   :param probes: Probes of the call.
   :type probes: tuple of Probe
   :param buffers: As given by `init_captures`, or by the previous step.
   :type buffers: dict of str to array
   :param values: What `read_boundary` gives after the step.
   :type values: dict of str to array
   :param index: Step within the call, from 0.
   :type index: array
   :param start: Where the call starts, as `start_of` gives it.
   :type start: Start or array or None

   :returns: The updated buffers.
   :rtype: dict of str to array

   .. rubric:: Notes

   Every step writes its value, without a branch.


.. py:function:: uses_rows(probe, accumulators = None)

   Returns whether a call stacks the values of every step of ``probe`` in its rows.

   True for traces and rasters. Given ``accumulators``, also true for summaries whose histogram
   is counted after the call.

   :param probe: Probe of the call.
   :type probe: Probe
   :param accumulators: As given by `init_accumulators`.
   :type accumulators: dict, optional

   :rtype: bool


.. py:function:: accumulated_fields(probe)

   Returns the names of the running statistics a summary probe keeps.

   ``n`` counts the steps. ``mean``, ``m2``, ``min``, ``max`` and ``active`` are kept per unit.
   ``hist`` and ``hist_high`` are the two words of the histogram counts.

   :param probe: Probe of the call.
   :type probe: Probe

   :returns: Names of the statistics the reductions of ``probe`` need. Empty for other modes.
   :rtype: tuple of str


.. py:function:: init_accumulators(probes, values, steps = None, split = True)

   Creates the initial running statistics of the summary probes.

   :param probes: Probes of the call.
   :type probes: tuple of Probe
   :param values: What `ProbeContext.values` gives on one step, or its shapes as ``jax.eval_shape`` gives
                  them.
   :type values: dict of str to array or ShapeDtypeStruct
   :param steps: Steps of the call. Without it, every histogram is counted step by step, and a summary
                 keeps one set of statistics.
   :type steps: int, optional
   :param split: Whether the call may cross the end of a group. When False, a summary keeps one set of
                 statistics.
   :type split: bool, default True

   :returns: One entry per summary probe, by `Probe.key`, with the fields of `accumulated_fields`.
   :rtype: dict of str to dict of str to array

   .. rubric:: Notes

   A histogram whose values over the call take at most `SETTINGS.histogram_rows_limit` bytes is
   counted after the call from its rows (`uses_rows`), and keeps no ``hist`` statistics. A summary
   grouped by steps, in a call that may cross the end of a group, keeps one set of statistics for
   each group the call can touch (`slots`), along a leading axis.


.. py:function:: accumulate(probes, accumulators, values, index = None, start = None)

   Updates the running statistics of the summary probes with the values of one step.

   The updates are elementwise, except for histograms, which count the values of the step. Mean
   and spread follow Welford's algorithm. A summary grouped by steps updates the statistics of
   the group holding the step.

   :param probes: Probes of the call.
   :type probes: tuple of Probe
   :param accumulators: As given by `init_accumulators`, or by the previous step.
   :type accumulators: dict
   :param values: What `ProbeContext.values` gave on this step.
   :type values: dict of str to array
   :param index: Step within the call, from 0. Needed by summaries grouped by steps.
   :type index: array, optional
   :param start: Where the call starts, as `start_of` gives it.
   :type start: Start or array, optional

   :returns: The updated statistics.
   :rtype: dict

   .. rubric:: Notes

   Floating values are rounded to their dtype first (`_rounded`). With at most
   `SETTINGS.masked_slots` groups, the statistics of every group are updated, masked to the group
   of the step. With more, those of the group of the step are read and written at a dynamic index.


.. py:function:: merge_summaries(parts)

   Merges the partials of a summary over several calls, on the host.

   * Step counts and ``active`` counts are added.
   * Means are weighted by steps.
   * Spreads follow the parallel form of Welford's algorithm.
   * ``min`` and ``max`` take the extremes of the parts.
   * Histogram counts are combined as int64.

   :param parts: Partials of one group from each call, on the host.
   :type parts: list of dict of str to array

   :returns: Partials over all their steps, as `_summary_complete` takes them. A single part is
             returned as is.
   :rtype: dict of str to array


.. py:function:: finalize(probes, rows, accumulators, start = None, end = None, spaced = (), phases = None)

   Builds the records of a call on the device, to move to the host.

   :param probes: Probes of the call.
   :type probes: tuple of Probe
   :param rows: Rows of every step, as `pack_step` or `ProbeContext.collect` gives them, stacked along a
                leading step axis by ``jax.lax.scan``.
   :type rows: StepRecords or dict of str to array
   :param accumulators: Running statistics after the last step.
   :type accumulators: dict
   :param start: What `read_boundary` returned before and after the call. The ungrouped ``delta`` needs
                 both, and the ungrouped ``snapshot`` needs ``end``.
   :type start: dict of str to array, optional
   :param end: What `read_boundary` returned before and after the call. The ungrouped ``delta`` needs
               both, and the ungrouped ``snapshot`` needs ``end``.
   :type end: dict of str to array, optional
   :param spaced: Buffers of `write_spaced` after the last step. Traces and rasters with a stride that are
                  not in them are read from ``rows``.
   :type spaced: tuple, optional
   :param phases: Where the call starts, as `start_of` gives it. Groups and strides are aligned on the
                  steps of the run. Without it, the call starts a group and every stride.
   :type phases: Start or array, optional

   :returns: One entry per probe moved to the host, by `Probe.key`.

             * ``trace`` and ``raster``: the final records. With a stride, ``ceil(steps / stride)``
               rows. Rows past the steps kept hold the last step of the call.
             * ``summary``: the running statistics (`accumulated_fields`), with a leading axis of
               groups for a summary with several groups (`slots`).
             * ungrouped ``delta``: the partials per row (`_delta_partials`).
             * ungrouped ``snapshot``: the value, restricted to ``units``.

             `complete` finishes them on the host. Snapshots and deltas with a group are kept on the
             device by `held`.
   :rtype: dict


.. py:function:: held(probes, start, end, captured)

   Returns the values a call keeps on the device for its snapshots and deltas with a group.

   The recorder reads them once it knows where the groups end.

   :param probes: Probes of the call.
   :type probes: tuple of Probe
   :param start: What `read_boundary` returned before and after the call.
   :type start: dict of str to array
   :param end: What `read_boundary` returned before and after the call.
   :type end: dict of str to array
   :param captured: Buffers of `capture` after the last step.
   :type captured: dict of str to array

   :returns: By `Probe.key`, values with a leading axis of rows.

             * ``last``: the value after the last step, one row. It ends a group at the end of the
               call, a group by tag, or a group cut short.
             * ``ends``: the values at the ends of groups within the call, the last row spare.
               Present for a group of steps in a call that may cross the end of one.
             * ``first``: the value before the first step, one row. Deltas only.
   :rtype: dict of str to dict of str to array


.. py:function:: group_values(probes, ends, end_rows, starts, start_rows)

   Computes what the snapshots and deltas of a set of measurements record for one group.

   Runs on the device, from the values `held` kept, in one call.

   :param probes: Snapshots and deltas with a group. Static.
   :type probes: tuple of Probe
   :param ends: For each probe, kept values holding the value after the last step of the group.
   :type ends: tuple of array
   :param end_rows: For each probe, the row of ``ends`` holding that value.
   :type end_rows: tuple of int32
   :param starts: For each delta, kept values holding the value before the first step of the group. None
                  for a snapshot.
   :type starts: tuple of array or None
   :param start_rows: For each delta, the row of ``starts`` holding that value.
   :type start_rows: tuple of int32

   :returns: For each probe, the snapshot value, or the delta partials (`_delta_partials`) from the
             value before the group to the value after it.
   :rtype: tuple

   .. rubric:: Notes

   Jitted with ``probes`` static. The rows are traced. Compiles once per set of probes and
   shapes.


.. py:function:: complete(probes, records)

   Completes the records `finalize` returned, on the host.

   :param probes: Probes of the call.
   :type probes: tuple of Probe
   :param records: As returned by `finalize`, moved to the host.
   :type records: dict

   :returns: One entry per probe of ``records``, by `Probe.key`, as NumPy arrays.

             * ``summary``: one entry per reduction, over steps and units. ``mean``, ``std``,
               ``active_fraction`` and ``inactive_unit_fraction`` are float32 scalars. ``min`` and
               ``max`` are scalars in float32 for floating values, uint8 for bool, and the integer
               type of integer values. ``active_fraction_per_unit`` holds one float32 per unit, and
               ``hist`` the int64 counts. A summary with a leading axis of groups gives a list of
               such entries, one per group holding steps of the call.
             * ``trace``: ``(ceil(steps / stride), ...)``, in the dtype read.
             * ``raster``: ``(ceil(steps / stride), units)`` bool.
             * ``snapshot``: the value at the end, restricted to ``units``.
             * ``delta``: one entry per reduction of ``end - start``, in float32.
   :rtype: dict


.. py:class:: Packed(buffer, layout, probes, held = None)

   Records of a call, packed into one byte buffer, and the values kept on the device.

   A pytree whose leaves are the buffer and the held values. The layout and the probes are
   static. `recorded_scan` returns one, and `Recorder.push` takes it.

   :param buffer: uint8 buffer.
   :type buffer: array
   :param layout: The records in the buffer, keyed by their path in the nested dictionary of records.
   :type layout: Layout
   :param probes: Probes the records belong to.
   :type probes: tuple of Probe
   :param held: What `held` keeps on the device, by `Probe.key`. Not part of the buffer.
   :type held: dict, optional

   .. rubric:: Notes

   Returned from a jitted function, the buffer moves to the host in one transfer. Bool records,
   such as rasters, take one bit per entry.

   .. seealso::

      :py:obj:`spark.scan`
          ``jax.lax.scan``, recorded within a call of a `spark.jit` function.

      :py:obj:`Recorder.push`
          Hands over the records of a call.


   .. py:attribute:: buffer


   .. py:attribute:: layout


   .. py:attribute:: probes


   .. py:attribute:: held


   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux, children)
      :classmethod:



   .. py:property:: nbytes
      :type: int


      Size of the buffer in bytes.


   .. py:method:: unpack(completed = True)

      Unpacks the records as nested dictionaries of NumPy arrays.

      Blocks until the buffer is on the host.

      :param completed: Whether to complete the records with `complete`. When False, they are returned as
                        `finalize` gave them.
      :type completed: bool, default True

      :returns: Records by `Probe.key`. The held values are not included.
      :rtype: dict



.. py:function:: pack(probes, records, kept = None)

   Packs the records of a call into one byte buffer.

   :param probes: Probes passed to `finalize`, stored for `Packed.unpack`.
   :type probes: tuple of Probe
   :param records: As returned by `finalize`.
   :type records: dict
   :param kept: As returned by `held`. Kept apart from the buffer.
   :type kept: dict, optional

   :returns: The records in one buffer, with ``kept``. ``records`` unchanged when both are empty.
   :rtype: Packed or dict


.. py:function:: merge_packed(parts, probes, device)

   Joins the records of calls of distinct probes into the records of one call of ``probes``.

   The buffers are moved to ``device`` and joined, and the values held on the device are moved
   there too, so that the end of a group reads them together. The transfers do not wait for the
   calls.

   :param parts: The records of each call, as `recorded_scan` returns them. No probe is in two of them.
   :type parts: sequence of Packed or dict
   :param probes: The probes of the call they make up, as `Recorder.probes` returned them.
   :type probes: tuple of Probe
   :param device: Where the records are joined.
   :type device: jax.Device

   :returns: The joined records, which `Recorder.push` takes as the records of the call. An empty
             dictionary when no call recorded anything.
   :rtype: Packed or dict


