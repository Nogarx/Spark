spark.recording.recorder
========================

.. py:module:: spark.recording.recorder


Attributes
----------

.. autoapisummary::

   spark.recording.recorder.PREEMPTION_NOTICE
   spark.recording.recorder.RESERVED_EVENT_KEYS
   spark.recording.recorder.ON_ERROR
   spark.recording.recorder.TRANSIENT_ERRORS


Exceptions
----------

.. autoapisummary::

   spark.recording.recorder.RecordingWarning
   spark.recording.recorder.Preempted


Classes
-------

.. autoapisummary::

   spark.recording.recorder.Recorder


Module Contents
---------------

.. py:data:: PREEMPTION_NOTICE
   :value: 'RECEIVED_PREEMPTION_NOTICE'


   Key the preemption service of ``jax.distributed`` sets when a process receives SIGTERM.

.. py:data:: RESERVED_EVENT_KEYS
   :value: ('t', 'kind', 'wall')


   Keys of an event row, not accepted in its payload.

.. py:data:: ON_ERROR
   :value: ('raise', 'continue')


   Accepted values of the ``on_error`` argument of `Recorder`.

.. py:data:: TRANSIENT_ERRORS

   Error numbers of file writes that are tried again.

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


