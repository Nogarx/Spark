spark.graph_editor.runs.data
============================

.. py:module:: spark.graph_editor.runs.data


Attributes
----------

.. autoapisummary::

   spark.graph_editor.runs.data.POINTS
   spark.graph_editor.runs.data.RAW_FAILURES
   spark.graph_editor.runs.data.RETRY_READS
   spark.graph_editor.runs.data.READ_ERRORS


Classes
-------

.. autoapisummary::

   spark.graph_editor.runs.data.RunData


Module Contents
---------------

.. py:data:: POINTS
   :value: 100000


   Approximate number of rows of a scalar series kept in memory. A longer series is kept as its
   envelope, reduced again once it holds more than twice this number of rows.

.. py:data:: RAW_FAILURES
   :value: 8


   Number of windows that fail to be read after which a search for a raw frame stops.

.. py:data:: RETRY_READS
   :value: 5.0


   Seconds for which a search for a raw frame skips a window that could not be read.

.. py:data:: READ_ERRORS

   Errors from reading a run that the viewer reports and continues after. They cover a run moved,
   deleted or being replaced, and file system errors.

.. py:class:: RunData(path, cached_bytes = 256 * 2**20)

   Object used to serve data to a RunViewer.

   RunData wraps a `spark.recording.Run` to stay up to date while the run is being written.

   Events, spans, groups, and windows are read once and each `refresh` reads the new rows added since.
   Values are sorted up by node of the graph and by step.

   :param path: Directory of the run.
   :type path: str or path-like
   :param cached_bytes: Memory kept for arrays read from window files.
   :type cached_bytes: int, default 256 MiB

   .. attribute:: run

      The run read.

      :type: Run

   .. attribute:: path

      Directory of the run.

      :type: pathlib.Path

   .. attribute:: config

      Configuration of the model recorded, or None when it cannot be loaded.

      :type: SparkConfig or None

   .. attribute:: config_error

      Error of loading the configuration, or None.

      :type: str or None

   .. attribute:: dt

      Duration of a step, in milliseconds, from the configuration. 1.0 without one.

      :type: float

   .. attribute:: events

      Events read so far, each as ``{**payload, 't', 'kind', 'wall'}``.

      :type: list of dict

   .. attribute:: windows

      Windows read so far.

      :type: list of Window

   .. attribute:: error

      Error of the last read that failed, as ``'Type: message'``, until a `refresh` succeeds.

      :type: str or None

   .. rubric:: Notes

   The arrays read from window files are kept up to ``cached_bytes``, and the least recently
   used are dropped first. When the run is resumed, or a commit read before is rolled back,
   everything read is dropped and read again. When more than ``4 * POINTS`` scalar rows were
   written since the last read, the series read so far are dropped and read again as envelopes.

   .. seealso::

      :py:obj:`RunViewerWindow`
          The window showing a run over the graph of its model.

      :py:obj:`spark.recording.Run`
          A run written by a `Recorder`, read back.


   .. py:attribute:: run
      :type:  spark.recording.run.Run


   .. py:attribute:: path


   .. py:attribute:: config
      :value: None



   .. py:attribute:: config_error
      :type:  str | None
      :value: None



   .. py:attribute:: dt


   .. py:attribute:: error
      :type:  str | None
      :value: None



   .. py:attribute:: scalars
      :type:  dict[str, _Table]


   .. py:property:: measurements
      :type: dict[str, spark.recording.measurements.Measurements]


      The measurements of the run, by name.


   .. py:property:: status
      :type: str


      Status of the run, as given by `Run.status`.


   .. py:property:: step
      :type: int


      Number of steps written so far.

      The larger of `Run.step` and the end of the last window read.


   .. py:method:: refresh()

      Reads what was written since the last call.

      A read that fails with one of `READ_ERRORS` is kept in `error`. What was read before the
      failure is kept.

      :returns: What changed, among ``'info'``, ``'scalars'``, ``'events'`` and ``'windows'``. Holds
                ``'error'`` when the read failed, or succeeded after a failure.
      :rtype: set of str



   .. py:method:: release()

      Drops the arrays and rows read so far.

      Called when the viewer window closes.



   .. py:method:: probes_of(node)

      Returns the probes addressing a node of the graph.

      :param node: Name of a module, or of an input of the model.
      :type node: str

      :returns: Name of the measurements and probe, for every probe whose path starts at the module
                ``node`` or that reads the input ``node`` of the model.
      :rtype: list of (str, Probe)



   .. py:method:: scalar_keys()

      Returns the names of every scalar series, as of the last `refresh`.

      :rtype: list of str



   .. py:method:: scalar(key)

      Returns the steps and values of a scalar series, ordered by step.

      The series is read from the run on the first call and extended by each `refresh`. A
      series of more than `POINTS` rows is kept as its envelope.

      :param key: Name of the series.
      :type key: str

      :returns: * **steps** (*ndarray*) -- Step of every row.
                * **values** (*ndarray*) -- Value of every row.

      .. rubric:: Notes

      An unknown key gives empty arrays. So does a read that fails, which is kept in `error`.
      A series held in memory is reduced to its envelope again once it has more than twice
      `POINTS` rows.



   .. py:method:: value_at(key, t)

      Returns the last value of a scalar series at or before a step.

      A series held whole in memory gives the value from memory. Otherwise the value is read
      from the run, without reading the series.

      :param key: Name of the series.
      :type key: str
      :param t: Step.
      :type t: int

      :returns: The value, or None when the series has none at or before ``t`` or the read fails.
      :rtype: float or None



   .. py:method:: logged_keys()

      Returns the names of the scalar series logged with `Recorder.log`.

      These are the series of `scalar_keys` that no probe of the measurements wrote.

      :rtype: list of str



   .. py:method:: activity(node, t)

      Returns the firing rate of a node at a step, in Hz.

      The rate is the mean of the ``active_fraction`` reductions of the summaries of the output
      ports of the node whose name holds ``spike``. Each is taken at its last value at or before ``t``.

      :param node: Name of the node.
      :type node: str
      :param t: Step.
      :type t: int

      :returns: The rate, or None when no such summary has a value at or before ``t``.
      :rtype: float or None

      .. rubric:: Notes

      The ``active_fraction`` reduction gives spikes per unit and step. It is converted to Hz with `dt`.



   .. py:method:: recorded(measurements)

      Returns the spans listed for a set of measurements, ordered by first step.

      One span is listed per call of the model. Spans lost with the process writing their
      window are left out.

      :param measurements: Name of the measurements.
      :type measurements: str

      :returns: * **t0** (*ndarray*) -- First step of every span.
                * **steps** (*ndarray*) -- Steps of every span.
                * **number** (*ndarray*) -- Window number of every span.



   .. py:method:: groups(measurements)

      Returns the groups listed for a set of measurements, ordered by first step.

      Groups lost with the process writing their window are left out.

      :param measurements: Name of the measurements.
      :type measurements: str

      :returns: * **t0** (*ndarray*) -- First step recorded in every group.
                * **steps** (*ndarray*) -- Steps of every group.
                * **number** (*ndarray*) -- Window number of every group.



   .. py:method:: written(measurements, number)

      Returns whether the file of a window is written.

      A live run lists its spans and groups before it writes their window.

      :param measurements: Name of the measurements.
      :type measurements: str
      :param number: Window number.
      :type number: int

      :rtype: bool



   .. py:method:: spans(measurements)

      Returns the steps recorded by a set of measurements, as joined spans.

      Overlapping and consecutive spans are joined. While the run is written, spans listed
      ahead of their window count as recorded, and the joined spans are extended with the
      spans added. Once it is not, only spans whose window is written count.

      :param measurements: Name of the measurements.
      :type measurements: str

      :returns: ``(spans, 2)`` array of the ``[start, end)`` steps of every joined span.
      :rtype: ndarray

      .. rubric:: Notes

      The result is cached until the spans listed, the windows read or the status of the run
      change.



   .. py:method:: segments(measurements)

      Returns `spans` as a list.

      :param measurements: Name of the measurements.
      :type measurements: str

      :returns: ``(start, end)`` of every joined span.
      :rtype: list of (int, int)



   .. py:method:: segment_at(measurements, t)

      Returns the joined span holding a step, or else the last one before it.

      :param measurements: Name of the measurements.
      :type measurements: str
      :param t: Step.
      :type t: int

      :returns: ``(start, end)`` of the span, or None when no joined span starts at or before ``t``.
      :rtype: tuple of (int, int) or None



   .. py:method:: span_at(measurements, t, written = False)

      Returns the last span listed that starts at or before a step.

      :param measurements: Name of the measurements.
      :type measurements: str
      :param t: Step.
      :type t: int
      :param written: Skip the spans whose window is not written yet.
      :type written: bool, default False

      :returns: ``(t0, steps, window number)`` of the span, or None without one.
      :rtype: tuple of (int, int, int) or None



   .. py:method:: group_at(measurements, t)

      Returns the last written group that starts at or before a step.

      :param measurements: Name of the measurements.
      :type measurements: str
      :param t: Step.
      :type t: int

      :returns: ``(t0, steps, window number)`` of the group, or None without one.
      :rtype: tuple of (int, int, int) or None



   .. py:method:: arrays(measurements, number, keys)

      Returns arrays of a window.

      Only the arrays not kept from an earlier call are read from the file. A file that cannot
      be read gives the arrays kept before, and is read again on the next call.

      :param measurements: Name of the measurements.
      :type measurements: str
      :param number: Window number.
      :type number: int
      :param keys: Names of the arrays.
      :type keys: sequence of str

      :returns: Arrays of ``keys`` that the window holds, or None while the window is not written.
      :rtype: dict of str to ndarray or None

      .. rubric:: Notes

      The arrays read are kept up to ``cached_bytes``, and the least recently used are dropped
      first. The arrays of the current call are never dropped.



   .. py:method:: rows_at(measurements, t)

      Returns the window and the joined span that `rows` reads at a step.

      These are the window of the last written span starting at or before ``t``, and the
      joined span holding that span. Equal results give equal rows.

      :param measurements: Name of the measurements.
      :type measurements: str
      :param t: Step.
      :type t: int

      :returns: Window number and ``(start, end)`` of the joined span, or None without them.
      :rtype: tuple of (int, (int, int)) or None



   .. py:method:: rows(measurements, key, t)

      Returns the steps and rows of a trace or raster around a step, ordered by step.

      The rows are those of the joined span of `rows_at`, read from the window of `rows_at`.
      The joined span holds consecutive recorded steps, such as an episode for measurements
      recorded by episode.

      :param measurements: Name of the measurements.
      :type measurements: str
      :param key: Key of the trace or raster.
      :type key: str
      :param t: Step.
      :type t: int

      :returns: Steps and rows, or None without a written span or when the window lacks ``key``.
      :rtype: tuple of (ndarray, ndarray) or None

      .. rubric:: Notes

      The rows of a window written in order are views of the arrays kept, without a copy.



   .. py:method:: group_value(measurements, key, t)

      Returns a value recorded once per group, for the group at a step.

      The group is the last written group starting at or before ``t``.

      :param measurements: Name of the measurements.
      :type measurements: str
      :param key: Name of the value, ``<probe key>#<reduction>`` or the key of a snapshot.
      :type key: str
      :param t: Step.
      :type t: int

      :returns: First step of the group and the value, or None without such a group or when its
                window lacks the value.
      :rtype: tuple of (int, ndarray) or None



   .. py:method:: raw_at(measurements, name, t)

      Returns the last frame of a raw stream at or before a step.

      Searches the windows holding frames of the stream, the latest first. A window that
      cannot be read is skipped for `RETRY_READS` seconds. The search stops after
      `RAW_FAILURES` windows that cannot be read.

      :param measurements: Name of the measurements.
      :type measurements: str
      :param name: Name of the raw stream.
      :type name: str
      :param t: Step.
      :type t: int

      :returns: Step and frame, or None when no frame is found.
      :rtype: tuple of (int, ndarray) or None



