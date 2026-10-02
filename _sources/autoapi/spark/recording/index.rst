spark.recording
===============

.. py:module:: spark.recording


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/recording/calls/index
   /autoapi/spark/recording/current/index
   /autoapi/spark/recording/measurements/index
   /autoapi/spark/recording/presets/index
   /autoapi/spark/recording/probe/index
   /autoapi/spark/recording/probe_context/index
   /autoapi/spark/recording/probe_targets/index
   /autoapi/spark/recording/recorder/index
   /autoapi/spark/recording/records/index
   /autoapi/spark/recording/reduce/index
   /autoapi/spark/recording/run/index
   /autoapi/spark/recording/runner/index
   /autoapi/spark/recording/scan/index
   /autoapi/spark/recording/settings/index
   /autoapi/spark/recording/store/index
   /autoapi/spark/recording/triggers/index
   /autoapi/spark/recording/utils/index


Attributes
----------

.. autoapisummary::

   spark.recording.SETTINGS


Exceptions
----------

.. autoapisummary::

   spark.recording.Preempted
   spark.recording.RecordingWarning


Classes
-------

.. autoapisummary::

   spark.recording.Run
   spark.recording.Record
   spark.recording.Window
   spark.recording.Probe
   spark.recording.SummaryProbe
   spark.recording.TraceProbe
   spark.recording.RasterProbe
   spark.recording.SnapshotProbe
   spark.recording.DeltaProbe
   spark.recording.ProbeMode
   spark.recording.SummaryReduction
   spark.recording.DeltaReduction
   spark.recording.Packed
   spark.recording.Runner
   spark.recording.ProbeTarget
   spark.recording.Trigger
   spark.recording.Every
   spark.recording.At
   spark.recording.Between
   spark.recording.Always
   spark.recording.Manual
   spark.recording.When
   spark.recording.Recorder
   spark.recording.Measurements
   spark.recording.RecordingSettings


Functions
---------

.. autoapisummary::

   spark.recording.load
   spark.recording.runs
   spark.recording.validate
   spark.recording.get_probe_targets
   spark.recording.log
   spark.recording.event
   spark.recording.tag
   spark.recording.raw
   spark.recording.record


Package Contents
----------------

.. py:class:: Run(path)

   A run written by a `Recorder`, opened for reading.

   A run can be opened while it is being written. Reads return what was written so far.

   :param path: Directory of the run.
   :type path: str or path-like

   .. attribute:: path

      Directory of the run.

      :type: pathlib.Path

   .. attribute:: info

      Contents of ``run.json``: identity, environment, status and progress.

      :type: dict

   .. attribute:: hparams

      Contents of ``hparams.json``.

      :type: dict

   .. attribute:: measurements

      The measurements of the run, by name.

      :type: dict of str to Measurements

   :raises FileNotFoundError: When ``path`` holds no ``run.json``.
   :raises ValueError: When the index was written by another version of the tables.

   .. rubric:: Notes

   `info` and the progress of the run are read when it is opened and by `refresh`.

   Each query opens a read-only connection to ``index.sqlite`` and closes it, unless it runs inside
   `reading`. Tables are read in pages of `SETTINGS.page_rows` rows or row ids, one statement per
   page.

   .. seealso::

      :py:obj:`load`
          Opens a run.

      :py:obj:`runs`
          Opens every run of a directory.

      :py:obj:`Recorder`
          Writes a run.

      :py:obj:`Window`
          One file of a set of measurements.

   .. rubric:: Examples

   >>> run = spark.recording.load('runs/20260923-101500_Brain_1a2b3c')
   >>> t, rate = run.scalar('summary/first_pool.soma:spikes/active_fraction')
   >>> t, potential = run.read('episode')[20].traces['first_pool.soma.potential']


   .. py:attribute:: path


   .. py:attribute:: info


   .. py:attribute:: hparams


   .. py:attribute:: measurements


   .. py:method:: reading()

      Reads the run through one connection until the block ends.

      Queries inside the block share one read-only connection. A block nested in another
      reuses its connection. The index is held only while a statement runs.

      :returns: Yields the run itself.
      :rtype: context manager

      .. rubric:: Examples

      >>> with run.reading():
      ...     series = {key: run.scalar(key) for key in run.scalar_keys()}



   .. py:method:: refresh()

      Reads ``run.json`` and the progress of the run again.

      :returns: The run itself.
      :rtype: Run



   .. py:method:: rows(table, after = 0)

      Returns the rows of a table of the index with a row id greater than ``after``.

      :param table: Name of the table, one of `TABLES`.
      :type table: str
      :param after: Row id after which rows are returned.
      :type after: int, default 0

      :returns: Rows ordered by row id, each led by its row id.
      :rtype: list of tuple

      :raises ValueError: When ``table`` is not one of `TABLES`.



   .. py:property:: status
      :type: str


      Status of the run, from ``run.json``.

      One of ``'running'``, ``'finished'``, ``'failed'`` and ``'preempted'``, or ``'unknown'``
      when it is missing. A run marked as running reads as ``'crashed'`` when its last
      heartbeat is older than `SETTINGS.crashed_after` heartbeat periods plus one second, and no
      recorder is known to hold its lock.


   .. py:property:: experiment
      :type: str | None


      Name of the experiment the run belongs to, as given to its `Recorder`, or None.


   .. py:property:: step
      :type: int


      Number of steps written so far.


   .. py:method:: config()

      Reads the configuration of the model recorded, from ``model.scfg``.

      :returns: Configuration of the model.
      :rtype: SparkConfig



   .. py:method:: scalar_keys()

      Returns the names of every scalar series, in sorted order.

      The series of summaries and deltas are named ``<measurements>/<address>/<reduction>``,
      also for runs that stored them with the mode of the probe.



   .. py:method:: tag_of(key)

      Returns the tag a scalar series was written per, or None for steps.

      A series logged with `Recorder.log` has the tag given to it. A summary has the tag its
      measurements are grouped by, also in runs written before the tags of the series were
      kept, where a logged series reads as per step.

      :param key: Name of the series.
      :type key: str

      :returns: The name of the tag, or None for steps and for an unknown series.
      :rtype: str or None



   .. py:method:: scalar(key, points = None, until = None)

      Returns the steps and values of one scalar series, ordered by step.

      An unknown ``key`` gives empty arrays.

      :param key: Name of the series.
      :type key: str
      :param points: Approximate maximum number of rows returned. A longer series is reduced to its
                     envelope. The envelope holds the rows with the lowest and the highest value in each
                     of ``points // 2`` equal bins of steps, and the first NaN of each bin holding one.
      :type points: int, optional
      :param until: Row id past which rows are left out, as `scalar_rows` returns it.
      :type until: int, optional

      :returns: * **steps** (*ndarray of int64*) -- Step of each row.
                * **values** (*ndarray of float64*) -- Value of each row. A value stored as NULL (NaN) reads as NaN.



   .. py:method:: scalar_at(key, t)

      Returns the last value of a scalar series at or before step ``t``.

      :param key: Name of the series.
      :type key: str
      :param t: Step.
      :type t: int

      :returns: The value last written at the latest step up to ``t``. NaN for a value stored as
                NULL. None when the series has no value up to ``t``, or does not exist.
      :rtype: float or None



   .. py:method:: scalars(prefix = '')

      Returns every scalar series whose name starts with ``prefix``.

      :param prefix: Start of the names of the series returned.
      :type prefix: str, default ''

      :returns: Steps and values of each series, as `scalar` returns them, by name.
      :rtype: dict of str to tuple of ndarray



   .. py:method:: scalar_rows(after = 0, keys = None)

      Returns the scalar rows with a row id greater than ``after``, by series.

      :param after: Row id after which rows are returned.
      :type after: int, default 0
      :param keys: Names of the series returned. All of them by default.
      :type keys: collection of str, optional

      :returns: * **last** (*int*) -- Row id of the last row of the table, or ``after`` when the table has no newer row.
                * **rows** (*dict of str to ndarray*) -- ``(rows, 2)`` float64 arrays of steps and values, ordered by step, by name. Series
                  with no new row are left out. A value stored as NULL (NaN) reads as NaN.

      .. rubric:: Notes

      Every row after ``after`` is read, with any ``keys``. The rows are filtered by series on
      the host.



   .. py:method:: events(kind = None)

      Returns the events of the run, ordered by step.

      :param kind: Kind of the events returned. All kinds by default.
      :type kind: str, optional

      :returns: One dictionary per event, its payload with ``t``, ``kind`` and ``wall`` (wall-clock
                time in seconds since the epoch).
      :rtype: list of dict



   .. py:method:: warnings()

      Returns the warnings the recorder gave while recording, ordered by step.

      Each is a `RecordingWarning` given to the loop, and dropped what it concerns, such as the
      frames of a raw stream no measurements declare.

      :returns: The ``warning`` events: ``message``, the step ``t`` and ``wall``, and what the
                warning concerns (``raw``, ``measurements``, ``scalar`` or ``event``).
      :rtype: list of dict



   .. py:method:: tags(key = None)

      Returns the tags set during the run, ordered by step.

      :param key: Name of the tags returned. All of them by default.
      :type key: str, optional

      :returns: One dictionary per value set, with ``t``, ``key``, ``value`` (decoded from JSON) and
                ``wall`` (wall-clock time in seconds since the epoch).
      :rtype: list of dict



   .. py:method:: windows(name = None)

      Returns the window files of the run, ordered by measurements and number.

      :param name: Name of the measurements. All of them by default.
      :type name: str, optional

      :returns: One entry per window file written.
      :rtype: list of Window



   .. py:method:: window(name, number)

      Reads the arrays of one window file.

      The keys are:

      * ``span_t0`` and ``span_steps``, the first step and the number of steps of each span.
      * ``<probe key>`` and ``<probe key>#t``, the rows of a trace or raster and their steps.
        Rasters are bool arrays of ``(steps, units)``.
      * ``group_t0`` and ``group_steps``, the first step recorded of each group and the steps
        it covers, when the window holds groups.
      * ``<probe key>#<reduction>`` for summaries and deltas, and ``<probe key>`` for
        snapshots, one row per group.
      * ``raw:<name>`` and ``raw:<name>#t``, the frames of a raw stream and their steps.

      A group whose steps were not all recorded covers fewer steps than the group of the
      measurements.

      :param name: Name of the measurements.
      :type name: str
      :param number: Number of the window within the measurements.
      :type number: int

      :returns: Arrays by key.
      :rtype: dict of str to ndarray

      :raises ValueError: When the measurements ``name`` have no window ``number``.



   .. py:method:: timeline(name)

      Reads every window of a set of measurements, joined along the first axis.

      The rows of every record follow each other, on the steps of the run. `read` gives them
      one record at a time.

      :param name: Name of the measurements.
      :type name: str

      :returns: Arrays by key, with the keys of `window`. Empty when the measurements have no
                window.
      :rtype: dict of str to ndarray

      :raises ValueError: When the windows hold different probes, as after a resume with other measurements.



   .. py:method:: read(name)

      Reads what a set of measurements recorded, one record per group.

      Measurements grouped by a tag give one record per value of the tag, such as one per
      episode, keyed by the value. Measurements grouped by steps give one record per group,
      keyed by its number. Measurements without a group give one record per stretch of steps
      recorded one after the other, keyed by its first step. Only what was recorded has a
      record.

      :param name: Name of the measurements.
      :type name: str

      :returns: Records by key, ordered by step. Empty when the measurements have no window.
      :rtype: dict of object to Record

      :raises ValueError: When the windows hold different probes, as after a resume with other measurements.

      .. rubric:: Notes

      A tag that takes a value again after another one starts another record. Its key is
      ``(value, n)``, for the ``n``-th time the value comes back. The steps before the first
      value of the tag are in the record keyed None.

      .. seealso::

         :py:obj:`Record`
             What a record holds.

         :py:obj:`timeline`
             The rows of every record, on the steps of the run.

      .. rubric:: Examples

      >>> episodes = run.read('episode')
      >>> t, potential = episodes[20].traces['first_pool.soma.potential']
      >>> run.read('summary')[20].summaries['first_pool.soma:spikes'].active_fraction



   .. py:method:: record(name, steps = 1)

      Asks the recorder writing this run to record a set of measurements, as
      `Recorder.record`.

      The request is written as a file of ``requests/``. It can be written from any process,
      on any host, with write access to the run directory. The recorder reads requests every
      half second and applies them before its next call.

      :param name: Name of the measurements.
      :type name: str
      :param steps: Steps to record them for.
      :type steps: int, default 1

      :returns: Id of the request, as listed by `requests`.
      :rtype: str

      :raises ValueError: When the run is not being written, has no measurements ``name``, or ``steps`` is not
          a positive integer.



   .. py:method:: requests()

      Returns the requests made with `record`, oldest first.

      :returns: One dictionary per request, its payload with ``id``, ``kind``, ``status``, ``wall``
                and ``handled``. ``status`` is one of:

                * ``'pending'`` until the recorder reads the request.
                * ``'received'`` or ``'rejected'`` once the recorder reads it.
                * ``'applied'`` once the measurements are recorded.
                * ``'expired'`` for a request the recorder can no longer apply, left when it closes
                  or when the run is resumed.
      :rtype: list of dict



   .. py:method:: checkpoints()

      Returns the steps of the checkpoints written completely, in increasing order.

      A checkpoint is written aside and moved in place once complete, so a file
      ``checkpoints/<step>.spark`` is a complete checkpoint.



   .. py:method:: restore(step = None)

      Returns the model saved by a checkpoint of the run, with `Controller.from_checkpoint`.

      :param step: Step of the checkpoint. The last one by default.
      :type step: int, optional

      :returns: The model, built from the configuration saved with the checkpoint and holding its
                state.
      :rtype: Controller

      :raises FileNotFoundError: When the run has no checkpoint, or none at ``step``.



   .. py:method:: __repr__()


.. py:class:: Record(key, t0, t, *, traces = None, rasters = None, raw = None, summaries = None, deltas = None, snapshots = None)

   Object holding a collection of measurements recorded over one group of steps (`Measurements.group`).

   The data is held by type, addressed by the type of the probe:

   * `traces`, `rasters` and `raw` (by stream name): `Rows`, the rows and their steps.
   * `summaries` and `deltas`: `Reductions`, the reductions of the group by name.
   * `snapshots`: the value at the end of the group, an array.

   .. attribute:: key

      For a group by tag, the value of the tag: ``(value, n)`` the ``n``-th time the value
      comes back, None for the steps before the tag is first set. For a group of ``n`` steps,
      its number ``g``, of the steps ``[g * n, (g + 1) * n)``. Without a group, the first step
      of the stretch.

      :type: object

   .. attribute:: t0

      Step of the run at which the group starts, or the stretch.

      :type: int

   .. attribute:: t

      Steps recorded, counted from `t0`. A group recorded from its middle starts past 0.

      :type: ndarray

   .. attribute:: traces, rasters, raw



      :type: dict of str to Rows

   .. attribute:: summaries, deltas



      :type: dict of str to Reductions

   .. attribute:: snapshots



      :type: dict of str to ndarray

   .. rubric:: Notes

   Printed, or shown by a notebook, a record lists what it holds.

   .. seealso::

      :py:obj:`Run.read`
          The records of a set of measurements.

   .. rubric:: Examples

   >>> episode = run.read('episode')[20]
   >>> t, potential = episode.traces['first_pool.soma.potential']
   >>> episode.summaries['first_pool.soma:spikes'].active_fraction


   .. py:attribute:: key


   .. py:attribute:: t0


   .. py:attribute:: t


   .. py:attribute:: traces


   .. py:attribute:: rasters


   .. py:attribute:: raw


   .. py:attribute:: summaries


   .. py:attribute:: deltas


   .. py:attribute:: snapshots


   .. py:method:: from_arrays(key, t0, t, arrays)
      :classmethod:


      Builds a record from arrays named as in a window file.

      ``<address>@trace`` and ``<address>@raster`` with their steps in ``<...>#t``,
      ``raw:<name>`` with ``raw:<name>#t``, ``<address>@summary#<reduction>``,
      ``<address>@delta#<reduction>`` and ``<address>@snapshot``. The steps are counted from
      ``t0``.



   .. py:property:: steps
      :type: int


      Number of steps recorded.


   .. py:method:: __repr__()


   .. py:method:: __str__()


.. py:class:: Window

   One file of a set of measurements.

   A window holds what the measurements recorded over some time: the rows of their traces and
   rasters, one per step kept; their summaries, snapshots and deltas, one row per group of
   steps; and the raw frames kept meanwhile.

   .. attribute:: measurements

      Name of the measurements.

      :type: str

   .. attribute:: number

      Number of the file within the measurements, from 0.

      :type: int

   .. attribute:: t0, t1

      First step covered, and the step after the last one.

      :type: int

   .. attribute:: spans

      Number of spans in the file, one per call that recorded the measurements.

      :type: int

   .. attribute:: groups

      Number of groups in the file, one row each.

      :type: int

   .. attribute:: file

      Path of the ``.npz`` file.

      :type: pathlib.Path

   .. attribute:: raw

      Raw streams with frames in the file.

      :type: tuple of str

   .. seealso::

      :py:obj:`Run.windows`
          The windows of a run.

      :py:obj:`Run.window`
          The arrays of one window, by name and number.


   .. py:attribute:: measurements
      :type:  str


   .. py:attribute:: number
      :type:  int


   .. py:attribute:: t0
      :type:  int


   .. py:attribute:: t1
      :type:  int


   .. py:attribute:: spans
      :type:  int


   .. py:attribute:: groups
      :type:  int


   .. py:attribute:: file
      :type:  pathlib.Path


   .. py:attribute:: raw
      :type:  tuple[str, ...]
      :value: ()



   .. py:method:: read(keys = None)

      Reads the arrays of the file.

      :param keys: Keys to read. All of them by default. Missing keys are left out.
      :type keys: iterable of str, optional

      :returns: Arrays by key, as listed by `Run.window`. Rasters are bool arrays of
                ``(steps, units)``.
      :rtype: dict of str to ndarray

      .. rubric:: Notes

      Rasters are stored as bits along the unit axis, with their number of units as
      ``<key>#units``. Those entries are not returned.



.. py:function:: load(path)

   Opens a run written by a `Recorder`.

   :param path: Directory of the run.
   :type path: str or path-like

   :returns: The run, opened for reading.
   :rtype: Run

   :raises FileNotFoundError: When ``path`` holds no ``run.json``.
   :raises ValueError: When the index was written by another version of the tables.

   :Warns: **RecordingWarning** -- When the recorder gave warnings while recording, with the first of them. `Run.warnings`
           lists them all.

   .. seealso::

      :py:obj:`runs`
          Opens every run of a directory.

      :py:obj:`Run`
          A run written by a `Recorder`, opened for reading.


.. py:function:: runs(root = 'runs', experiment = None)

   Opens every run of a directory, or those of one experiment.

   Directories without ``run.json``, and hidden ones such as the runs being created, are
   skipped. Directories that cannot be read as a run are skipped with a warning.

   :param root: Directory of the runs.
   :type root: str or path-like, default 'runs'
   :param experiment: Name of an experiment, as given to `Recorder`. Only its runs are opened.
   :type experiment: str, optional

   :returns: The runs, oldest first by the time they were created, then by name. Empty when ``root``
             does not exist.
   :rtype: list of Run

   .. seealso::

      :py:obj:`load`
          Opens one run.

      :py:obj:`Run`
          A run written by a `Recorder`, opened for reading.


.. py:class:: Probe

   Bases: :py:obj:`abc.ABC`


   Specification for a measurement, denoted by a port address and the measurement operator:
   `SummaryProbe`, `TraceProbe`, `RasterProbe`, `SnapshotProbe` or `DeltaProbe`.

   :param address: Address of the value. ``path`` is the dotted chain of module names from the root
                   controller.

                   * ``path:port``: an output port of a module.
                   * ``path.__call__:port``: an input port of a controller, ``__call__:port`` for the root.
                   * ``path.name``: an attribute of a module.
   :type address: str

   .. attribute:: mode

      Mode of the class of the probe.

      :type: ProbeMode

   .. attribute:: kind

      ``'port'`` or ``'attribute'``.

      :type: str

   .. attribute:: path

      Path of the module producing the port or holding the attribute.

      :type: tuple of str

   .. attribute:: name

      Port or attribute name.

      :type: str

   .. attribute:: key

      Name of the entry of the probe in the records, ``address@mode``.

      :type: str

   :raises ValueError: When the address is malformed, or a field is invalid.

   .. rubric:: Notes

   Probes are frozen and hashable. Probes of the same class with equal fields compare and hash
   equal. A tuple of probes can be a static argument of a jitted function.

   Ports are read as the module produces them. Attributes are read at the end of each step,
   after every module of the step ran.

   .. seealso::

      :py:obj:`Measurements`
          A named set of probes recorded together.

      :py:obj:`get_probe_targets`
          Lists the values of a controller that a probe can address.

      :py:obj:`validate`
          Checks probes against a controller.

      :py:obj:`Recorder`
          Records the measurements asked for and writes them to a run.


   .. py:attribute:: address
      :type:  str


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: kind
      :type:  str


   .. py:attribute:: path
      :type:  tuple[str, ...]


   .. py:attribute:: name
      :type:  str


   .. py:method:: __post_init__()


   .. py:method:: __eq__(other)


   .. py:method:: __hash__()


   .. py:method:: __reduce__()


   .. py:property:: key
      :type: str



   .. py:property:: per_step
      :type: bool


      Whether the probe is read on every step (`STEP_PROBES`).


   .. py:method:: to_dict()

      Returns the mode and the fields of the probe as JSON types.

      :returns: ``mode``, and the keyword arguments of the class, with lists in place of tuples.
      :rtype: dict



   .. py:method:: from_dict(data)
      :classmethod:


      Rebuilds a probe from `to_dict`.

      :param data: Mode and fields of a probe, as `to_dict` gives them.
      :type data: dict

      :returns: A probe of the class of the mode.
      :rtype: Probe

      :raises ValueError: When the mode is unknown, or is not the mode of the class called.



.. py:class:: SummaryProbe

   Bases: :py:obj:`Probe`


   Reductions of a value over the units and over each group of steps.

   :param address: Address of the value, as in `Probe`.
   :type address: str
   :param reduce: Reductions, as `SummaryReduction` members or their names.

                  * ``MEAN``, ``STD``, ``MIN``, ``MAX``: over the units and the steps of the group.
                  * ``ACTIVE_FRACTION``: fraction of the units active, averaged over the steps.
                  * ``ACTIVE_FRACTION_PER_UNIT``: fraction of the steps on which each unit is active.
                  * ``INACTIVE_UNIT_FRACTION``: fraction of the units active on none of the steps.
                  * ``HIST``: counts in ``bins`` equal bins over ``range``.
   :type reduce: SummaryReduction or str or sequence of them, default ('mean', 'std', 'min', 'max')
   :param bins: Number of bins of ``HIST``. Ignored without it.
   :type bins: int, default 32
   :param range: Lower and upper edges of ``HIST``, finite as float32. Required by ``HIST`` and ignored
                 without it. Values outside the range, and NaN, are not counted.
   :type range: tuple of float, optional
   :param group: How the steps are split into groups, one record per group. Set by the `Measurements`
                 holding the probe; see `Measurements` for how a group is recorded.

                 * A number of steps ``n``: groups of ``n`` steps on the steps of the run, ``[0, n)``,
                   ``[n, 2n)``, and so on.
                 * The name of a tag, such as ``'episode'``: a new group each time the tag takes a
                   different value (`Recorder.tag`).
   :type group: int or str, optional

   .. rubric:: Notes

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

   .. rubric:: Examples

   >>> SummaryProbe('first_pool:out_spikes', reduce=(
   ...     SummaryReduction.ACTIVE_FRACTION, SummaryReduction.INACTIVE_UNIT_FRACTION,
   ... ))
   >>> SummaryProbe('first_pool.soma.potential', reduce=SummaryReduction.HIST,
   ...              range=(-80.0, 40.0))


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: reduce
      :type:  tuple[SummaryReduction | str, ...]
      :value: ('mean', 'std', 'min', 'max')



   .. py:attribute:: bins
      :type:  int
      :value: 32



   .. py:attribute:: range
      :type:  tuple[float, float] | None
      :value: None



   .. py:attribute:: group
      :type:  int | str | None
      :value: None



.. py:class:: TraceProbe

   Bases: :py:obj:`Probe`


   The value on every step.

   :param address: Address of the value, as in `Probe`.
   :type address: str
   :param units: Flat indices of the units kept. All units when omitted.
   :type units: sequence of int, optional
   :param stride: Keeps the steps ``t`` of the run with ``t % stride == 0``.
   :type stride: int, default 1

   .. rubric:: Notes

   A unit is an entry of the value. A trace keeps the dtype of the value. A trace of an
   attribute holds the value the next step starts from.

   A trace keeps every step of the call it records. With a stride, it keeps only the steps of
   the stride when every step would take more than `SETTINGS.spaced_rows_limit`
   bytes.

   .. rubric:: Examples

   >>> TraceProbe('first_pool.soma.potential', units=range(64))
   >>> TraceProbe('__call__:signal', stride=10)


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: units
      :type:  tuple[int, ...] | None
      :value: None



   .. py:attribute:: stride
      :type:  int
      :value: 1



.. py:class:: RasterProbe

   Bases: :py:obj:`Probe`


   Whether each unit is active, on every step.

   :param address: Address of the value, as in `Probe`.
   :type address: str
   :param units: Flat indices of the units kept. All units when omitted.
   :type units: sequence of int, optional
   :param stride: Keeps the steps ``t`` of the run with ``t % stride == 0``.
   :type stride: int, default 1

   .. rubric:: Notes

   A unit is an entry of the value. A unit is active on a step when its value is nonzero. A
   NaN counts as active. A float16 or bfloat16 value is rounded to its dtype before the test.

   A raster is moved and stored as bits, and read as bool. It keeps every step of the call it
   records. With a stride, it keeps only the steps of the stride when every step would take
   more than `SETTINGS.spaced_rows_limit` bytes.

   XLA on the CPU reads float32 subnormals (below about 1.2e-38 in magnitude) as zero. GPUs do
   not read them as zero.

   .. rubric:: Examples

   >>> RasterProbe('first_pool:out_spikes')
   >>> RasterProbe('first_pool:out_spikes', units=range(256))


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: units
      :type:  tuple[int, ...] | None
      :value: None



   .. py:attribute:: stride
      :type:  int
      :value: 1



.. py:class:: SnapshotProbe

   Bases: :py:obj:`Probe`


   The value of an attribute at the end of each group of steps.

   :param address: Address of an attribute, as in `Probe`.
   :type address: str
   :param units: Flat indices of the units kept. All units when omitted.
   :type units: sequence of int, optional
   :param group: How the steps are split into groups, one record per group, as for `SummaryProbe`: a
                 number of steps or the name of a tag. Set by the `Measurements` holding the probe.
   :type group: int or str, optional

   .. rubric:: Notes

   The value is read after the last step of each group. Until the group ends, a snapshot with a
   group keeps the value after each call on the device, and, for a call crossing the end of a
   group, the value at that end.

   .. rubric:: Examples

   >>> SnapshotProbe('first_pool.synapses.kernel')


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: units
      :type:  tuple[int, ...] | None
      :value: None



   .. py:attribute:: group
      :type:  int | str | None
      :value: None



.. py:class:: DeltaProbe

   Bases: :py:obj:`Probe`


   Reductions of the change of an attribute over each group of steps.

   :param address: Address of an attribute, as in `Probe`.
   :type address: str
   :param reduce: Reductions, as `DeltaReduction` members or their names.

                  * ``FULL``: the change of every unit.
                  * ``NORM``: the Euclidean norm of the change.
                  * ``MEAN_ABS``: the mean absolute change over the units.
   :type reduce: DeltaReduction or str or sequence of them, default ('norm',)
   :param group: How the steps are split into groups, one record per group, as for `SummaryProbe`: a
                 number of steps or the name of a tag. Set by the `Measurements` holding the probe.
   :type group: int or str, optional

   .. rubric:: Notes

   The change is the value after the last step of the group minus the value before its first
   step recorded, computed in float32. Until the group ends, a delta with a group keeps the
   value before and after each call on the device, and, for a call crossing the end of a group,
   the value at that end.

   .. rubric:: Examples

   >>> DeltaProbe('first_pool.synapses.kernel',
   ...            reduce=(DeltaReduction.NORM, DeltaReduction.MEAN_ABS))


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: reduce
      :type:  tuple[DeltaReduction | str, ...]
      :value: ('norm',)



   .. py:attribute:: group
      :type:  int | str | None
      :value: None



.. py:class:: ProbeMode

   Bases: :py:obj:`enum.StrEnum`


   How a probe records its value.

   Each probe is associated with a specific `Probe.mode`.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: SUMMARY
      :value: 'summary'


      reductions of the value over the units and over each group of steps.

      :type: `SummaryProbe`


   .. py:attribute:: TRACE
      :value: 'trace'


      the value on every step.

      :type: `TraceProbe`


   .. py:attribute:: RASTER
      :value: 'raster'


      whether each unit is active, on every step.

      :type: `RasterProbe`


   .. py:attribute:: SNAPSHOT
      :value: 'snapshot'


      the value of an attribute at the end of each group of steps.

      :type: `SnapshotProbe`


   .. py:attribute:: DELTA
      :value: 'delta'


      reductions of the change of an attribute over each group of steps.

      :type: `DeltaProbe`


.. py:class:: SummaryReduction

   Bases: :py:obj:`enum.StrEnum`


   `SummaryProbe` requested operations (reductions) on the Probes.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: MEAN
      :value: 'mean'


      Mean over the units and the steps.


   .. py:attribute:: STD
      :value: 'std'


      Standard deviation over the units and the steps.


   .. py:attribute:: MIN
      :value: 'min'


      Minimum over the units and the steps.


   .. py:attribute:: MAX
      :value: 'max'


      Maximum over the units and the steps.


   .. py:attribute:: ACTIVE_FRACTION
      :value: 'active_fraction'


      Fraction of the units active, averaged over the steps. For spikes, the firing rate per step.


   .. py:attribute:: ACTIVE_FRACTION_PER_UNIT
      :value: 'active_fraction_per_unit'


      Fraction of the steps on which each unit is active. One value per unit.


   .. py:attribute:: INACTIVE_UNIT_FRACTION
      :value: 'inactive_unit_fraction'


      Fraction of the units inactive in the group.


   .. py:attribute:: HIST
      :value: 'hist'


      Counts of the values of the units and the steps in ``bins`` equal bins over ``range``.


   .. py:property:: scalar
      :type: bool


      Whether the reduction gives one number per group of steps. ``ACTIVE_FRACTION_PER_UNIT``
      and ``HIST`` give an array. A `Recorder` also writes the scalar reductions to the
      scalars of the run.


.. py:class:: DeltaReduction

   Bases: :py:obj:`enum.StrEnum`


   `DeltaProbe` requested operations (reductions) on the Probes. Unlike `SummaryProbe`, `DeltaProbe`
   compute differences of an attribute over each group of steps.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: FULL
      :value: 'full'


      The change of every unit.


   .. py:attribute:: NORM
      :value: 'norm'


      Euclidean norm of the change.


   .. py:attribute:: MEAN_ABS
      :value: 'mean_abs'


      Mean absolute change over the units.


   .. py:property:: scalar
      :type: bool


      Whether the reduction gives one number per group of steps. ``FULL`` gives an array. A
      `Recorder` also writes the scalar reductions to the scalars of the run.


.. py:function:: validate(controller, probes)

   Checks that every probe is a valid probe (targets an existing variable within the controller).

   The address of each probe must name a module, port or attribute of the controller. ``units`` must
   be within the size of the value, and a `SummaryProbe` or a `DeltaProbe` must have entries to reduce.
   An attribute must hold an array.

   :param controller: A built controller, called at least once.
   :type controller: Controller
   :param probes: Probes to check, one per key.
   :type probes: iterable of Probe

   :raises TypeError: When ``controller`` is not a controller, or a probe is not a `Probe`.
   :raises ValueError: When the controller is not built, two probes share a key, or a probe does not fit the controller.
       The message lists the names available where the address failed.

   .. rubric:: Notes

   The sizes of controller inputs and of the ports of nested controllers are not known before
   the controller is traced, and are not checked. `Runner` traces every probe of its recorder before
   its first call.

   .. seealso::

      :py:obj:`get_probe_targets`
          Lists the values of a controller that a probe can address.


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



.. py:class:: Runner(model, recorder = None, *, outputs = 'last', unroll = 1, donate = True)

   Steps a model and records what its recorder asks for.

   The runner holds its own copy of the model state. Each `run` advances it by one compiled
   ``jax.lax.scan``. The recorder gives the probes before the call and receives the records
   after it, without waiting for them.

   :param model: A built model, called once with example inputs.
   :type model: Controller
   :param recorder: Where the records go. A path creates a `Recorder` there with the default measurements of
                    the model (`presets.default`). Without one, nothing is recorded.
   :type recorder: Recorder or str or path-like, optional
   :param outputs: What `run` returns. ``'last'`` gives the outputs of the last step, ``'all'`` those of
                   every step stacked, and ``'none'`` nothing.
   :type outputs: str, default 'last'
   :param unroll: Passed to ``jax.lax.scan``.
   :type unroll: int, default 1
   :param donate: Whether each call reuses the memory of the state it receives. The runner then copies the
                  state of ``model`` first. The model given is not modified.
   :type donate: bool, default True

   .. attribute:: state

      The current state, as given by ``spark.split``.

      :type: State

   .. attribute:: graph

      The graph of the model, as given by ``spark.split``.

      :type: GraphDef

   .. attribute:: recorder

      The recorder given or created, or None.

      :type: Recorder or None

   :raises ValueError: When ``model`` is not built, or ``outputs`` is unknown.

   .. rubric:: Notes

   The probes of a recorder given are validated against the model. A recorder given to a second
   runner gives a warning. The second runner starts from the state of the model given, while
   the run goes on from its current step.

   Before its first `run` or `warmup`, the runner traces the model with the probes of every set
   of measurements of the recorder, without compiling it. A probe that cannot record the model
   raises there, before its measurements are first recorded. Inputs are checked against the
   shapes the model was built with.

   With the state sharded across devices, the values of a step are not packed into one row.
   With several processes, the records of a call are replicated on every device, where
   process 0 reads them.

   .. seealso::

      :py:obj:`Recorder`
          Decides what each call records and writes it to a run.

      :py:obj:`spark.jit`
          Compiles a function whose calls the open recorder records.

      :py:obj:`Run`
          A run written by a `Recorder`, opened for reading.

   .. rubric:: Examples

   >>> runner = spark.recording.Runner(brain, 'runs')
   >>> for episode in range(100):
   ...     runner.recorder.tag(episode=episode)
   ...     outputs = runner.run(50, {'signal': observation})
   >>> runner.close()


   .. py:attribute:: recorder
      :value: None



   .. py:attribute:: outputs
      :value: 'last'



   .. py:attribute:: unroll
      :value: 1



   .. py:attribute:: state


   .. py:method:: run(steps, inputs = None, per_step = None)

      Advances the model ``steps`` steps, in one call of the compiled scan.

      :param steps: Steps of the call. Each distinct value compiles once per probe set.
      :type steps: int
      :param inputs: Inputs held over the call, by input name. Arrays are converted to the payload type
                     and dtype the model expects. For spikes, nonzero entries spike and negative ones are
                     inhibitory.
      :type inputs: dict, optional
      :param per_step: Inputs given step by step, with a leading axis of length ``steps``.
      :type per_step: dict, optional

      :returns: The outputs, as chosen by ``outputs``. They stay on the device until read.
      :rtype: dict of str to SparkPayload or None

      :raises ValueError: When ``steps`` is not a positive integer, or when an input is unknown, missing,
          given both held and per step, or of another shape than the model was built with.

      .. rubric:: Notes

      The call is traced and compiled first, where SIGINT stops it. A SIGINT during the call
      and the hand-over of its records is delivered once both are done. A second SIGINT while
      the recorder waits for room in its queue raises at once.



   .. py:attribute:: __call__


   .. py:method:: warmup(steps, inputs = None, per_step = None)

      Compiles the calls of the run ahead of it.

      Compiles one call of ``steps`` steps for every probe set of `Recorder.warmup_sets`. The
      state is not advanced.

      :param steps: Steps of every call.
      :type steps: int
      :param inputs: Example inputs held over the call, as for `run`.
      :type inputs: dict, optional
      :param per_step: Example inputs given step by step, as for `run`.
      :type per_step: dict, optional

      :returns: Number of sets compiled.
      :rtype: int

      .. rubric:: Notes

      Warns when a set needs more memory than the device has.



   .. py:method:: checkpoint(step = None)

      Saves the current state to the run of the recorder, as `Recorder.checkpoint`.

      :param step: Step the checkpoint is filed under. The current step of the recorder by default.
      :type step: int, optional

      :returns: File of the checkpoint.
      :rtype: pathlib.Path

      :raises ValueError: When the runner has no recorder.



   .. py:property:: model
      :type: spark.nn.controllers.base.Controller


      A copy of the model in its current state.


   .. py:method:: close()

      Closes the recorder, writing what is left. Does nothing without a recorder.



   .. py:method:: __enter__()


   .. py:method:: __exit__(kind, error, traceback)


.. py:class:: ProbeTarget

   Description of variable that a Probe can measure.

   Similar in spirit to Specs but for Probes.

   .. attribute:: address

      Probe address.

      :type: str

   .. attribute:: kind

      ``'port'`` or ``'attribute'``.

      :type: str

   .. attribute:: shape

      Shape of the value.

      :type: tuple of int

   .. attribute:: dtype

      Dtype of the recorded array. Bool for spike payloads.

      :type: numpy.dtype

   .. attribute:: payload

      Class name of the value of a port or property. ``'Variable'`` for a variable.

      :type: str

   .. attribute:: module

      Class name of the module producing the port or holding the attribute. For the inputs of
      a controller, the controller.

      :type: str

   .. seealso::

      :py:obj:`get_probe_targets`
          Lists every value of a controller that a probe can address.

      :py:obj:`Probe`
          A value read from a controller, and how it is recorded.


   .. py:attribute:: address
      :type:  str


   .. py:attribute:: kind
      :type:  str


   .. py:attribute:: shape
      :type:  tuple[int, ...]


   .. py:attribute:: dtype
      :type:  numpy.dtype


   .. py:attribute:: payload
      :type:  str


   .. py:attribute:: module
      :type:  str


   .. py:property:: size
      :type: int


      Number of entries of the value.


   .. py:property:: spikes
      :type: bool


      Whether the value is a spike payload.


   .. py:method:: __str__()


.. py:function:: get_probe_targets(controller, inputs)

   Returns a list of every variable (input, output or property), within the controller,
   that can be targeted using a Probe, as well as its shape and dtype.

   :param controller: The controller to be described.
   :type controller: Controller
   :param inputs: Input sample.
   :type inputs: dict[str, SparkPayload]

   :returns: Tuple of ProbeTargets pointing towards the controller's input/output/attributes,
             sorted by module path and name, that a Probe can target.
   :rtype: tuple of ProbeTarget

   :raises TypeError: When an input is not a payload.

   .. seealso::

      :py:obj:`ProbeTarget`
          A value of a controller that a probe can target.

      :py:obj:`Probe`
          A value read from a controller, and how it is recorded.

      :py:obj:`validate`
          Checks that every probe addresses something the controller produces.

   .. rubric:: Examples

   >>> for target in spark.recording.get_probe_targets(brain, inputs):
   ...     print(target)


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



.. py:class:: Recorder(root = 'runs', config = None, measurements = None, *, name = None, experiment = None, run_id = None, hparams = None, source = None, **options)

   Records the measurements asked for and writes them to a run.

   `record` asks for measurements over the next steps. A trigger, when the measurements have
   one, also asks for steps on its own. With the default trigger, `Manual`, measurements are
   recorded only when asked. Host values are written with `log`, `event`, `tag` and `raw`.

   The recorder works one call of the model at a time. While it is open, every call of a
   `spark.jit` function running a `spark.scan` is a call of the recorder: before the call, the
   recorder decides what it records, `spark.scan` records it, and the records are handed over
   after the call. `spark.jit` and `Runner` follow the same steps: before a call, `probes`
   returns the probes of the measurements recorded for it, and `start` where the call starts.
   After the call, `push` takes its records, with as many steps as `probes` was given. Their
   transfers start at once, and a writer thread completes the records and writes them.

   Records are placed on the steps of the run. Groups and strides do not depend on how the
   steps are split into calls. The recorder decides at the start of each call what it records,
   and records the call whole.

   The calls of a thread go to the last recorder it opened among those still open. A recorder
   opened within a run, such as one for an evaluation, takes the calls until it closes. Runs
   trained in threads of one process are recorded apart, each by the recorder its thread
   opened. A thread that opened no recorder uses the recorder open in the process, when only
   one is open, and raises at its first call when several are.

   :param root: Directory holding the runs. The run gets its own directory inside, see `path`.
   :type root: str or path-like, default 'runs'
   :param config: The configuration of the model recorded, written to the run. A built model can be given
                  in its place: its configuration is written, and the probes are checked against it.
   :type config: SparkConfig or Controller, optional
   :param measurements: What can be recorded. Defaults to `presets.default` for a model given as ``config``,
                        recorded by triggers.
   :type measurements: sequence of Measurements, optional
   :param name: Name of the run, part of its directory name. Defaults to the class name of the model,
                or ``'run'`` without ``config``.
   :type name: str, optional
   :param experiment: Name of the experiment the run belongs to, such as one configuration trained with
                      several seeds. Written to ``run.json``, and read as `Run.experiment`. The run viewer
                      draws the runs of one experiment as a whole.
   :type experiment: str, optional
   :param run_id: Name of the directory of the run, in place of ``<date>-<time>_<name>_<id>``. A job
                  restarted by its scheduler finds its run again with it, see `resume`.
   :type run_id: str, optional
   :param hparams: Parameters of the experiment, written to ``hparams.json``.
   :type hparams: dict, optional
   :param source: File the model configuration was read from. Its metadata, such as the node positions of
                  the editor, is written with the configuration of the run.
   :type source: str or path-like, optional
   :param queue_size: Items waiting to be written, such as the records of a call or a host value, past which
                      calls block.
   :type queue_size: int, default 64
   :param queue_bytes: Bytes of records and raw frames waiting to be written, past which calls block.
   :type queue_bytes: int, default 1 GiB
   :param flush_steps: Recorded steps after which a file of a set of measurements is written. By default, files
                       are written by time and size alone.
   :type flush_steps: int, optional
   :param flush_seconds: Seconds after which an open file is written.
   :type flush_seconds: float, default 60.0
   :param flush_bytes: Host memory, in bytes, of the records and raw frames of an open file, past which it is
                       written.
   :type flush_bytes: int, default 256 MiB
   :param heartbeat: Seconds between updates of ``run.json`` while the run is open.
   :type heartbeat: float, default 5.0
   :param on_error: What follows a failure of the writer, as on a full disk. With ``'raise'``, the next call
                    to the recorder raises RuntimeError, and the run stops. With ``'continue'``, it warns
                    once and training goes on without recording. Mistakes in what the loop hands over are
                    not failures of the writer: they give a `RecordingWarning` either way.
   :type on_error: str, default 'raise'
   :param signals: Signals after which `probes` raises `Preempted` before the next call, such as the
                   SIGTERM a cluster scheduler sends before it stops a job. The run is then closed as
                   ``'preempted'``.
   :type signals: sequence of int, optional

   .. attribute:: path

      Directory of the run.

      :type: pathlib.Path

   .. attribute:: measurements

      The measurements that can be recorded.

      :type: tuple of Measurements

   .. attribute:: step

      Current step of the run, advanced by `push`.

      :type: int

   .. attribute:: tags

      The last value of every tag.

      :type: dict

   :raises ValueError: When ``measurements`` is not given and ``config`` is not a model, when an option is
       invalid, or when two measurements share a name or disagree on a probe they share.
   :raises FileExistsError: When the directory of ``run_id`` exists.
   :raises RuntimeError: With several processes, when ``jax.distributed`` is not initialized, or when a process
       fails to open the run or to create its recorder within `SETTINGS.open_timeout`.

   .. rubric:: Notes

   A run directory holds:

   * ``run.json``, the identity, environment, status and progress of the run.
   * ``model.scfg`` and ``hparams.json``, the configuration of the model and the parameters.
   * ``recorder.json``, the measurements.
   * ``index.sqlite``, the scalars, events, tags, spans and groups recorded, windows, requests
     and progress.
   * ``run.lock``, held by the recorder writing the run.
   * ``requests/``, the requests not read yet.
   * ``windows/<measurements>/<number>.npz``, the window files.
   * ``checkpoints/<step>.spark``, the checkpoints written by `checkpoint`.

   A window file holds what one set of measurements recorded over some time: the rows of its
   traces and rasters, one per step kept; its summaries, snapshots and deltas, one row per
   group of steps; and the raw frames kept meanwhile. An open
   file is written ``flush_seconds`` after it is opened, or once it holds ``flush_bytes``
   bytes. With ``flush_steps``, a file holding that many steps is written when the next call of
   its measurements arrives.

   Files are written to a temporary name, flushed to disk and renamed. After a crash, each file is
   complete or absent. The index is committed about once a second (`SETTINGS.commit_every`) and
   when a window is written.

   A group of steps (`Measurements.group`) is written once it ends. A group of a number of
   steps ends with its last step, and a group by tag when the tag takes a different value. The
   group in progress when the recorder closes is written cut short. A group whose steps were
   not all recorded, as when the measurements were recorded from the middle of it, is written
   with the steps it covers.

   While the queue of the writer is full, in items or in bytes, calls to the recorder block
   until the writer catches up. The waits are counted in ``run.json`` as ``stalls``. Records
   waiting to be written hold device memory. A single item larger than ``queue_bytes`` is let
   through when nothing else waits. After a failure of the writer, the open windows are written
   if they can be, and the run is marked ``'failed'``.

   What stops a run, and what does not. A failure of the writer, such as a full disk or a
   quota, stops the run with ``on_error='raise'``, the default: the next call to the recorder
   raises RuntimeError. A mistake in what the loop hands over never stops it: a frame of a raw
   stream no measurements declare, or of another shape than the first, measurements named
   without existing, a scalar that is not a number, or a declared raw stream recorded without
   a frame. Each gives a `RecordingWarning` once, is written to the run as a ``warning`` event
   (`Run.warnings`), and what it concerns is dropped. A short run with
   ``warnings.simplefilter('error', RecordingWarning)`` raises each where it happens, before a
   long run is started.

   A recorder left open is closed when the interpreter exits, before the thread pools of the
   interpreter stop, and waits for the checkpoints written in the background. The run is marked
   ``'preempted'`` after one of ``signals``, ``'failed'`` after an uncaught exception, and
   ``'finished'`` otherwise, `sys.exit` with an error code included. Closed by a ``with``
   block, the run is marked ``'failed'`` after `sys.exit` with an error code too. In a
   notebook, an uncaught exception does not mark the run ``'failed'``.

   An interrupt (Ctrl-C) received during a call of a `spark.jit` function is held until the call
   returns its result, and raised by the next call of the recorder from the loop (`probes`,
   `record`, `log`, `event`, `tag` or `raw`), before it does anything. The steps of the run then
   match the state the loop holds. A second interrupt during the call raises at once. A long
   compilation is not held. An interrupt (Ctrl-C) received while the recorder closes is raised
   once what is left is written. A second interrupt stops the wait, and what is not written yet may be lost. A
   process killed by a signal it does not handle, such as the SIGKILL that follows SIGTERM,
   loses the windows being filled and the groups in progress. `resume` marks the spans and
   groups of those windows as lost.

   Measurements can also be recorded from outside the training process, by the run viewer of
   the editor or by `Run.record` from any process or host with access to the run directory. The
   writer thread reads those requests, and the recorder applies them before its next call.

   The handlers of ``signals`` are installed from the main thread and shared by the open
   recorders. A handler set before is still called, except the one of SIGINT raising
   KeyboardInterrupt. A signal received after the last call, which `probes` never raised, takes
   its usual effect once the run is closed. Without a handler set before, the process then
   ends. In a notebook, the kernel sets the handler of SIGINT again for every cell, and SIGINT
   is handled only in the cell creating the recorder.

   With several JAX processes (``jax.distributed``), the recorder records one model sharded
   across them, and every process runs the same calls. Every process creates the recorder and
   calls it in the same order. Process 0 creates and writes the run, and decides what every
   call records. The other processes receive its decisions through the key-value store of
   ``jax.distributed`` and write nothing. A recorder created on process 0 alone raises after
   `SETTINGS.open_timeout`.

   Tags, `record`, `When` conditions and requests take effect from process 0. The processes
   stop together. A failure of the writer with ``on_error='raise'``, or a signal of ``signals``
   received by any process, raises from `probes` on every process at the same call. When the
   preemption service of ``jax.distributed`` runs, it notes SIGTERM and keeps its handler.
   Other signals are published by the process receiving them. Process 0 reads both from the
   key-value store. Records are replicated on every device, where process 0 reads them. A
   snapshot of a value too large for one device cannot be recorded.

   .. seealso::

      :py:obj:`Measurements`
          A named set of probes recorded together.

      :py:obj:`spark.jit`
          Compiles a function whose calls the open recorder records.

      :py:obj:`Runner`
          Steps a model and records what its recorder asks for.

      :py:obj:`Run`
          A run read back from its directory.

   .. rubric:: Examples

   >>> @partial(spark.jit, static_argnames=['steps'])
   ... def run(graph, state, steps, **inputs):
   ...     def step(state, _):
   ...         model = spark.merge(graph, state)
   ...         outputs = model(**inputs)
   ...         return spark.split(model)[1], outputs
   ...     return spark.scan(step, state, length=steps)
   >>> episode = spark.recording.Measurements('episode', probes, group='episode')
   >>> graph, state = spark.split(brain)
   >>> hparams = {'lr': 0.1}
   >>> with spark.recording.Recorder('runs', brain.config, [episode], hparams=hparams) as recorder:
   ...     for number in range(100):
   ...         recorder.tag(episode=number)
   ...         if number % 20 == 0:
   ...             recorder.record('episode')
   ...         state, outputs = run(graph, state, steps=50, **inputs)
   ...         recorder.log(reward=reward)


   .. py:method:: resume(path, model = None, measurements = None, *, step = None, **options)
      :classmethod:


      Reopens a run and appends to it.

      The step count continues from the last step written to the run, or from the step of a
      later checkpoint. Window numbers continue after the last window of each set of
      measurements.

      :param path: Directory of the run. With several processes, only process 0 reads it.
      :type path: str or path-like
      :param model: The model recorded. When it is built, the probes are checked against it.
      :type model: Controller, optional
      :param measurements: What can be recorded. Defaults to the measurements the run was last opened with.
      :type measurements: sequence of Measurements, optional
      :param step: Step to continue from, such as the step of the checkpoint the model was restored
                   from. The last step written by default.
      :type step: int, optional
      :param \*\*options: The options of `Recorder`: ``queue_size``, ``queue_bytes``, ``flush_steps``,
                          ``flush_seconds``, ``flush_bytes``, ``heartbeat``, ``on_error`` and ``signals``.

      :returns: The recorder writing the run.
      :rtype: Recorder

      :raises ValueError: When ``step`` is not a non-negative integer, or an option is invalid.
      :raises PermissionError: When the run directory cannot be written.
      :raises RuntimeError: When another recorder is writing the run.

      .. rubric:: Notes

      The spans and groups listed for a window that was never written are marked lost, with
      window -1. A window file written but not listed is listed. Requests left pending expire,
      and files left half written are removed.

      The condition of a `When` trigger is code and is not written to the run. Without
      ``measurements``, such measurements come back with a `Manual` trigger, with a warning.

      With ``step``, the records written after it are kept, and the steps after it are
      recorded twice. The tags take the values they had at ``step``. A group holding ``step``
      starts again there, and is written twice too.

      A lock left by the file system of a node that failed is released by removing
      ``run.lock``, once no process writes the run.

      .. rubric:: Examples

      A job that its scheduler may restart:

      >>> run = pathlib.Path('runs') / os.environ['SLURM_JOB_ID']
      >>> step = None                     # the step of the last checkpoint, 0 without one
      >>> if (run / 'run.json').exists():
      ...     step = max(spark.recording.load(run).checkpoints(), default=0)
      >>> if jax.process_count() > 1:     # every process goes on as process 0 does
      ...     found = np.asarray(-1 if step is None else step)
      ...     step = int(multihost_utils.broadcast_one_to_all(found))
      ...     step = None if step < 0 else step
      >>> if step is None:
      ...     brain = build(config)
      ...     recorder = Recorder('runs', brain, run_id=os.environ['SLURM_JOB_ID'])
      ... else:
      ...     saved = spark.recording.load(run)
      ...     if step in saved.checkpoints():
      ...         brain = saved.restore(step)
      ...     else:
      ...         brain = build(saved.config())
      ...     recorder = Recorder.resume(run, brain, step=step)

      ``build`` builds a model from a configuration and calls it once with example inputs. The
      model is restored from the last checkpoint, if any, and built from the configuration of
      the run otherwise. The recording goes on from the step of that checkpoint, or from the
      start without one.

      With several processes, one process may see the files of the run a moment before
      another. Every process takes the decision of process 0. ``multihost_utils`` is
      ``jax.experimental.multihost_utils``.



   .. py:method:: probes(steps = None)

      Returns the probes of the measurements recorded for the next call of the model.

      The same tuple object is returned for the same set of recorded measurements. Pass it as
      a static keyword argument, ``()`` included, with `start`. Measurements with a
      ``lookback`` are captured on every call.

      :param steps: Steps of the next call. A trigger counting steps records the call when any of its
                    steps is recorded. Without ``steps``, it records the call when its first step is.
                    When given, `push` must receive the same steps.
      :type steps: int, optional

      :returns: Probes to run in the call. Empty when nothing is recorded or captured.
      :rtype: tuple of Probe

      :raises Preempted: When one of the signals of the recorder was received. The call is not run.
      :raises RuntimeError: When the recorder is closed, or its writer failed with ``on_error='raise'``.

      .. rubric:: Notes

      With one process, a second call before `push` warns, and the call the first one recorded
      is not recorded. With several processes, every process calls it for every call. Process
      0 decides what the call records, and the other processes receive the decision.



   .. py:method:: start(probes)

      Returns where the next call starts on the steps of the run, for the recorded scan.

      Called after `probes`. The result holds the current step modulo the group sizes and strides
      of ``probes``. The recorded scan aligns its groups and strides on the steps of the run with
      it, however the steps are split into calls. The result also tells whether the call crosses
      the end of a group, from the steps given to `probes`. Without them, the call may cross one.

      :param probes: The probes `probes` returned for the call.
      :type probes: tuple of Probe

      :returns: Phases of the current step, and whether the call may cross the end of a group.
      :rtype: Start

      .. rubric:: Examples

      >>> probes = recorder.probes(50)
      >>> start = recorder.start(probes)
      >>> outputs, state, records = call(graph, state, inputs, start, steps=50, probes=probes)
      >>> recorder.push(records, 50)



   .. py:method:: record(name, steps = 1)

      Records the measurements ``name`` for the next ``steps`` steps.

      The steps count from the current step. Measurements with any trigger can be recorded
      this way. Every call of the model holding one of those steps is recorded whole.
      Measurements with a group are recorded to the end of every group holding one of those
      steps, with one record per group. Called again before those steps end, the recording
      lasts until the later end.

      :param name: Name of the measurements.
      :type name: str
      :param steps: Steps to record them for.
      :type steps: int, default 1

      :Warns: **RecordingWarning** -- When there are no measurements ``name``, or ``steps`` is not a positive integer.
              Nothing is recorded.

      .. rubric:: Notes

      With several processes, only the calls on process 0 take effect.

      .. rubric:: Examples

      >>> recorder.record('activity', 500)  # the next 500 steps
      >>> recorder.record('episode')        # grouped by a tag: to the end of the episode



   .. py:method:: push(records, steps)

      Hands over the records of the call just dispatched and advances the step count.

      Call it after every call of the model, recorded or not. The transfers of ``records``
      start at once, and it returns without waiting for them.

      :param records: The records the call returned. `Packed`, or an empty dictionary when nothing was
                      recorded.
      :type records: Packed or dict
      :param steps: Steps of the call. The steps given to `probes`, if any.
      :type steps: int

      :raises TypeError: When ``records`` is neither `Packed` nor empty.
      :raises ValueError: When ``steps`` is not a positive integer or differs from the steps given to
          `probes`, or when ``records`` does not match the probes `probes` returned for the
          call.
      :raises RuntimeError: When the recorder is closed, or its writer failed with ``on_error='raise'``.

      .. rubric:: Notes

      Blocks while the queue of the writer is full. The step count advances even when handing
      the records over is interrupted. With several processes, only process 0 hands records to
      the writer.



   .. py:method:: warmup_sets(steps = None)

      Returns the probe sets `spark.Jit.warmup` and `Runner.warmup` compile.

      The sets are those the triggers counting steps record over the next `SETTINGS.warmup_calls`
      calls of ``steps`` steps. Measurements grouped by steps stay recorded until their group
      ends. The measurements recorded otherwise are added to each set. These are the measurements
      recorded by `record`, requests, conditions or triggers counting a tag, and those grouped by
      a tag. The set of all measurements, which needs the most memory, is included. Measurements
      with a ``lookback`` are captured in every set.

      :param steps: Steps of every call. Without it, all measurements count as recorded otherwise.
      :type steps: int, optional

      :returns: The distinct probe sets, each the tuple `probes` returns for it.
      :rtype: list of tuple of Probe

      .. rubric:: Notes

      The triggers counting steps are simulated over one period of their combined pattern. The
      simulation starts from the current step, from the offsets of `Every`, and from the steps
      where `At` and `Between` start and stop. When the measurements recorded otherwise make more
      than `SETTINGS.warmup_subsets` sets, each is added alone and all together, instead of in
      every combination.

      Measurements with a schedule of their own that `record` records as well, as from the
      viewer, can record a set not compiled ahead. That set compiles when it first occurs.



   .. py:method:: log(values = None, *, step = None, tag = None, **scalars)

      Writes scalars held on the host, at the current step or at ``step``.

      A series is logged per step, or per the integer tag ``tag``, such as once per episode.
      Its tag is written with its name once, read by `Run.tag_of`; the run viewer draws the
      series against the values of the tag. Scalars logged per a tag are written at the step
      the tag took its value, where the summaries grouped by the tag are, unless ``step`` is
      given.

      :param values: Scalars by name.
      :type values: dict of str to float, optional
      :param step: Step of the scalars. The current step by default, or the step the tag ``tag`` took
                   its value.
      :type step: int, optional
      :param tag: Name of the integer tag, set with `tag`, the scalars are logged per. Per step
                  without one.
      :type tag: str, optional
      :param \*\*scalars: Scalars by name, added to ``values``. Scalars named ``step`` or ``tag`` are given in
                          ``values``.
      :type \*\*scalars: float

      :raises RuntimeError: When the recorder is closed, or its writer failed with ``on_error='raise'``.

      :Warns: **RecordingWarning** -- When a name is not a non-empty string, or a value is not a number. That value is
              dropped. When ``tag`` is not a name, or is ``'step'``: the scalars are logged per
              step. When ``tag`` is not set to an integer yet: the rows logged before it is have
              no value of it. When a series is logged per another tag than before: it keeps the
              first.

      .. rubric:: Examples

      >>> recorder.log(reward=1.0, episode_steps=212)
      >>> recorder.log({'episode/steps': 212}, tag='episode')



   .. py:method:: event(kind, *, step = None, **payload)

      Writes an event, such as the end of an episode, with a JSON payload.

      :param kind: Kind of the event.
      :type kind: str
      :param step: Step of the event. The current step by default.
      :type step: int, optional
      :param \*\*payload: Payload of the event, written as JSON. It takes no key of `RESERVED_EVENT_KEYS`.

      :raises RuntimeError: When the recorder is closed, or its writer failed with ``on_error='raise'``.

      :Warns: **RecordingWarning** -- When the payload holds a key of `RESERVED_EVENT_KEYS`. The key is dropped.



   .. py:method:: tag(**tags)

      Sets tags on the timeline, such as ``episode=3``.

      The tags are written at the current step. Integer tags can be counted by triggers.
      Measurements grouped by a tag, as ``group='episode'``, start a new group of steps each
      time the tag takes a different value. An integer held in a NumPy or JAX array without
      dimensions is read as an int.

      :param \*\*tags: Values by tag name.

      :raises RuntimeError: When the recorder is closed, or its writer failed with ``on_error='raise'``.



   .. py:method:: raw(name, frame, *, step = None)

      Keeps a frame of a raw stream when measurements declaring the stream are recorded.

      The frame is copied and kept when measurements declaring ``name`` are recorded for the
      call starting at the current step. That call is the one being run between `probes` and
      `push`, or else the next one. When none are recorded, nothing is kept. It can be called
      on every step of an environment.

      :param name: Name of the raw stream, as declared by `Measurements`.
      :type name: str
      :param frame: Array of numbers or bools, such as an observation.
      :type frame: array-like
      :param step: Step of the frame. The current step by default.
      :type step: int, optional

      :raises RuntimeError: When a frame is kept after the recorder closed, or after its writer failed with
          ``on_error='raise'``.

      :Warns: **RecordingWarning** -- When no measurements declare ``name``, when ``frame`` holds no numbers or bools, or
              when it differs in shape or dtype from the first frame of ``name``. The frame is
              dropped.

      .. rubric:: Notes

      With several processes, only process 0 keeps frames.

      Measurements declaring ``name`` that are recorded over a group, or over calls one after
      the other, without a frame of it also give a `RecordingWarning`, from the next call of
      the model.



   .. py:method:: is_recording(name)

      Returns whether the measurements ``name`` are recorded for the call at the current step.

      That call is the one being run between `probes` and `push`, or else the next one, as for
      `raw`.

      :param name: Name of the measurements.
      :type name: str

      :rtype: bool

      .. rubric:: Notes

      With several processes, processes other than process 0 know it between `probes` and
      `push` only.

      :Warns: **RecordingWarning** -- When there are no measurements ``name``. Returns False.



   .. py:method:: checkpoint(model, *, step = None)

      Saves a model to ``checkpoints/<step>.spark`` in the run, with `Controller.checkpoint`.

      The state of the model is copied to the host before it returns, and the file is written
      in the background. A ``checkpoint`` event is written when the save starts.
      `Run.checkpoints` lists the checkpoints written completely, and `Run.restore` reads them
      back.

      :param model: The model, such as ``spark.merge(graph, state)`` in a loop over its state.
      :type model: Controller
      :param step: Step the checkpoint is filed under. The current step by default.
      :type step: int, optional

      :returns: File of the checkpoint.
      :rtype: pathlib.Path

      :raises RuntimeError: When the recorder is closed, its writer failed with ``on_error='raise'``, or the
          previous checkpoint could not be written.

      .. rubric:: Notes

      A save first waits for the previous one to be written, and `close` waits for the last
      one. A checkpoint at the same step is replaced. With several processes, every process
      calls it, the state is gathered on each of them, and process 0 writes it.

      .. rubric:: Examples

      >>> recorder.checkpoint(spark.merge(graph, state))



   .. py:method:: flush()

      Writes everything handed over so far and waits until it is written.

      The open windows are written as they stand. Each call ends the current file of every set
      of measurements.

      :raises RuntimeError: When the recorder is closed, or its writer failed with ``on_error='raise'``.

      .. rubric:: Notes

      With several processes, it returns at once on processes other than process 0.



   .. py:method:: close(status = 'finished', error = None)

      Writes what is left, marks the run with ``status`` and releases it.

      The groups in progress are written cut short, and the last checkpoint is waited for. A
      recorder that received one of its signals marks the run ``'preempted'`` in place of
      ``'finished'``. Does nothing when closed already.

      :param status: Status written to ``run.json``.
      :type status: str, default 'finished'
      :param error: Error written to ``run.json``.
      :type error: str, optional

      :raises RuntimeError: When the writer failed, with ``on_error='raise'``.

      .. rubric:: Notes

      An interrupt (Ctrl-C) received while the writer writes what is left is raised once it is
      done. A second interrupt stops the wait. An interrupt `spark.jit` held during the last call
      is dropped: the loop it would have stopped is over.



   .. py:method:: __enter__()


   .. py:method:: __exit__(kind, error, traceback)


   .. py:method:: __repr__()


.. py:exception:: Preempted(signum)

   Bases: :py:obj:`SystemExit`


   Exception raised by `Recorder.probes` after one of the signals the recorder handles.

   The call is not run. A recorder closed on it marks the run ``'preempted'``. Uncaught, it
   ends the process with exit code ``128 + signal``. With several processes, it is raised on
   every process at the same call.

   :param signum: The signal received.
   :type signum: int

   .. attribute:: signal

      The signal received.

      :type: int

   .. seealso::

      :py:obj:`Recorder`
          Records the measurements asked for and writes them to a run.

      :py:obj:`Initialize`


   .. py:attribute:: signal


   .. py:method:: __str__()

      Return str(self).



.. py:exception:: RecordingWarning

   Bases: :py:obj:`UserWarning`


   Warns of a mistake in what the loop hands over to a recorder. What it concerns is dropped,
   and the run goes on.

   Given by `Recorder.raw`, `Recorder.record`, `Recorder.log`, `Recorder.event` and
   `Recorder.is_recording`, and when measurements declaring a raw stream are recorded over a
   group, or a stretch of calls, without a frame of it. Each cause warns once per recorder, and
   is written to the run as a ``warning`` event, which `Run.warnings` lists.

   .. rubric:: Notes

   A failure of the writer, such as a full disk, is not a warning: it follows the ``on_error``
   option of the recorder.

   .. rubric:: Examples

   In a short run before a long one, every warning raises where it happens:

   >>> import warnings
   >>> warnings.simplefilter('error', spark.recording.RecordingWarning)

   From the command line: ``python -W error::spark.recording.RecordingWarning train.py``.

   Initialize self.  See help(type(self)) for accurate signature.


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



.. py:data:: SETTINGS

   The settings of `spark.recording` in use.

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


