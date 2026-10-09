spark.recording.runner
======================

.. py:module:: spark.recording.runner


Attributes
----------

.. autoapisummary::

   spark.recording.runner.OUTPUTS


Classes
-------

.. autoapisummary::

   spark.recording.runner.Runner


Module Contents
---------------

.. py:data:: OUTPUTS
   :value: ('last', 'all', 'none')


   Accepted values of the ``outputs`` argument of `Runner`.

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


