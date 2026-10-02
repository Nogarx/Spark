spark.recording.settings
========================

.. py:module:: spark.recording.settings

.. autoapi-nested-parse::

   Settings of `spark.recording`: the thresholds, periods and timeouts the recorder works with.

   `SETTINGS` holds the settings of the process. Each one is read when it is used, so a change
   applies from its next use.



Attributes
----------

.. autoapisummary::

   spark.recording.settings.SETTINGS


Classes
-------

.. autoapisummary::

   spark.recording.settings.RecordingSettings


Module Contents
---------------

.. py:class:: RecordingSettings

   The thresholds, periods and timeouts of `spark.recording`.

   `SETTINGS` holds the settings in use. Each one is read when it is used, so a change applies
   from its next use. Setting an attribute that is not a setting raises AttributeError.

   A setting read while a function is traced, such as the limits of the reductions, applies to
   the calls traced after the change. Calls compiled before keep the value they were traced
   with.

   .. rubric:: Examples

   >>> spark.recording.SETTINGS.open_timeout = 1200.0


   .. py:attribute:: variant_warning
      :type:  int
      :value: 16


      Number of distinct sets of recorded measurements past which `Recorder.probes` warns. Each set is
      compiled once.


   .. py:attribute:: tag_warning_calls
      :type:  int
      :value: 100


      Calls of the model after which `Recorder.probes` warns about triggers counting a tag never set
      as an integer, and about measurements grouped by a tag never set.


   .. py:attribute:: warmup_calls
      :type:  int
      :value: 100000


      Bound on the calls of the model `Recorder.warmup_sets` simulates.


   .. py:attribute:: warmup_subsets
      :type:  int
      :value: 64


      Number of probe sets past which `Recorder.warmup_sets` adds the measurements not recorded by a
      step trigger one at a time and all together, instead of in every combination.


   .. py:attribute:: requests_every
      :type:  float
      :value: 0.5


      Seconds between two reads of the requests of a run by the writer.


   .. py:attribute:: request_reads
      :type:  int
      :value: 10


      Attempts the writer makes at reading a request file before it rejects the request.


   .. py:attribute:: commit_every
      :type:  float
      :value: 1.0


      Seconds between two commits of the index while items keep arriving. Scalars, events, tags, spans
      and groups reach the index at most about this long after the writer handles them.


   .. py:attribute:: retry_after
      :type:  tuple[float, ...]
      :value: (1.0, 4.0)


      Seconds waited before each new attempt at a file write or an operation on the index that failed
      with a transient error, as network file systems give. A write that fails every attempt fails the
      writer.


   .. py:attribute:: open_timeout
      :type:  float
      :value: 600.0


      With several processes, seconds process 0 waits for every other process to create its recorder
      before it raises.


   .. py:attribute:: sync_timeout
      :type:  float
      :value: 86400.0


      With several processes, seconds a process waits for a decision of process 0, or for the other
      processes at a barrier, before it raises.


   .. py:attribute:: signals_every
      :type:  float
      :value: 0.5


      Seconds between two reads, by process 0, of the signals the other processes received.


   .. py:attribute:: decision_block
      :type:  int
      :value: 1000


      With several processes, calls of the model after which every process waits for the others at a
      barrier. No process runs more than a block ahead of another. The decisions of the block before
      the last are then removed from the key-value store of ``jax.distributed``.


   .. py:attribute:: masked_slots
      :type:  int
      :value: 4


      Largest number of groups of a call for which a summary updates the statistics of every group on
      every step, masked to the group of the step. With more groups, only the statistics of the group
      of the step are read and written, at a dynamic index.


   .. py:attribute:: select_per_step
      :type:  int
      :value: 4096


      Smallest size of a value from which scattered ``units`` of a trace or raster, at most a quarter
      of its entries, are selected on every step. Below it, whole values are stacked and their units
      selected after the scan.


   .. py:attribute:: histogram_compare_limit
      :type:  int
      :value: 2097152


      Largest ``bins * values`` a histogram counts by comparing every value with every edge. Beyond
      it, with more than `histogram_compare_bins` bins, the bin of each value is computed and the
      values are counted by bin.


   .. py:attribute:: histogram_compare_bins
      :type:  int
      :value: 8


      Largest number of bins for which a histogram compares every value with every edge, for any
      number of values.


   .. py:attribute:: histogram_scatter_limit
      :type:  int
      :value: 262144


      Most values a histogram counts by scattering them into their bins, on backends other than the
      CPU. Beyond it, their bin indices are sorted and counted by a search, except on the CPU, which
      always scatters.


   .. py:attribute:: histogram_rows_limit
      :type:  int
      :value: 16777216


      Largest size in bytes of the values of all the steps of a call that a histogram stacks on the
      device and counts after the call. Beyond it, the histogram counts the values of each step within
      the step, with device memory in proportion to its bins.


   .. py:attribute:: spaced_rows_limit
      :type:  int
      :value: 16777216


      Largest size in bytes of the values of all the steps of a call that a trace or raster with a
      stride stacks on the device, the steps it does not keep included. Beyond it, only the steps kept
      are written, to a buffer of their own, with one write per step.


   .. py:attribute:: busy_timeout
      :type:  float
      :value: 60.0


      Seconds a statement of the recorder waits for readers holding the index before SQLite reports
      the index busy. `store.commit` then warns and waits again.


   .. py:attribute:: read_timeout
      :type:  float
      :value: 10.0


      Seconds a reader of a run waits for the index while the recorder holds it.


   .. py:attribute:: page_rows
      :type:  int
      :value: 100000


      Number of rows, or of row ids, one statement reads. The recorder writing the run cannot commit
      while a statement runs.


   .. py:attribute:: crashed_after
      :type:  int
      :value: 3


      Heartbeat periods without a heartbeat after which a run marked as running reads as crashed.


   .. py:attribute:: abandoned_after
      :type:  float
      :value: 600.0


      Seconds without a heartbeat after which a run marked as running is taken as abandoned by its
      process. Used when that process cannot be looked up.


   .. py:attribute:: creation_expires_after
      :type:  float
      :value: 3600.0


      Seconds after which the directory of a run left half created, by a process that ended while
      creating it, is removed by the next run created beside it.


   .. py:attribute:: git_untracked
      :type:  int
      :value: 200


      Maximum number of untracked files listed in ``run.json``.


.. py:data:: SETTINGS

   The settings of `spark.recording` in use.

