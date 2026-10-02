spark.recording.run
===================

.. py:module:: spark.recording.run


Attributes
----------

.. autoapisummary::

   spark.recording.run.TABLES


Classes
-------

.. autoapisummary::

   spark.recording.run.Window
   spark.recording.run.Run


Functions
---------

.. autoapisummary::

   spark.recording.run.load
   spark.recording.run.runs


Module Contents
---------------

.. py:data:: TABLES
   :value: ('scalars', 'keys', 'events', 'tags', 'windows', 'spans', 'groups', 'requests')


   Tables read by `Run.rows`.

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


