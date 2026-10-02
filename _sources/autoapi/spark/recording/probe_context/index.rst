spark.recording.probe_context
=============================

.. py:module:: spark.recording.probe_context


Classes
-------

.. autoapisummary::

   spark.recording.probe_context.ProbeContext
   spark.recording.probe_context.ModelCalls


Module Contents
---------------

.. py:class:: ProbeContext(probes, capture_all = False)

   Bases: :py:obj:`_Scoped`


   Probe context that captures the values used by a collection of probes, during JIT tracing.

   On model call, controllers expose their values to the open probe context: inputs and the outputs
   of each of its modules, as well as the top most controller. Note that captured values are
   JAX tracers, not numbers.

   It is used as a context manager, and `recorded_scan` opens one around each step it traces.
   `get_probe_targets` opens one with ``capture_all`` to list the ports of a model.

   :param probes: Probes to capture, with distinct keys.
   :type probes: tuple of Probe
   :param capture_all: Whether to keep every port offered, requested or not.
   :type capture_all: bool, default False

   .. attribute:: model

      The model called within the context, once called.

      :type: Controller or None

   .. attribute:: calls

      Calls of the model within the context.

      :type: int

   :raises ValueError: When two probes share a key.
   :raises RuntimeError: When entered while another probe context is open. Probe contexts do not nest.

   .. rubric:: Notes

   The open probe context is held in a context variable, `spark.core.recording_hooks.PROBE_CONTEXT`.


   .. py:attribute:: probes


   .. py:attribute:: capture_all
      :value: False



   .. py:attribute:: model
      :type:  spark.nn.controllers.base.Controller | None
      :value: None



   .. py:property:: captured
      :type: dict[tuple[tuple[str, ...], str], Any]


      Values kept, by ``(path, port)``, in the order they were offered.


   .. py:method:: offer(values, suffix = None)

      Keeps the requested entries of ``values``, produced at the current scope.

      With ``capture_all``, keeps every entry.

      :param values: Ports produced at the current scope.
      :type values: dict of str to SparkPayload
      :param suffix: Appended to the scope. Controllers offer their inputs under ``__call__``.
      :type suffix: str, optional



   .. py:method:: values(model = None)

      Returns the values of the per-step probes for the step just taken.

      :param model: The model called within the probe context. Attributes are read from it. Defaults to
                    `model`.
      :type model: Controller, optional

      :returns: Values by `Probe.key`, as `step_value` gives them.
      :rtype: dict of str to array

      :raises RuntimeError: When a requested port was not produced during the call.
      :raises ValueError: When an attribute is not found, or a value does not suit its probe (`step_value`).
      :raises TypeError: When an attribute holds no array.



.. py:class:: ModelCalls(read = None)

   Bases: :py:obj:`_Scoped`


   Probe context that counts the calls of a model.

   Both uses happen while JAX traces a function:

   * Counting. The first time `spark.jit` traces a function while a recorder is open, it
     traces it within a `ModelCalls` to count the calls of the model outside `spark.scan`.
     The calls within the scan are not counted: no probe context is open there. A call
     outside the scan is an error, as the steps of a recorded call are the steps of its scan.
   * Reading. With ``read``, `run` calls a step function once more, on the carry of a scan.
     At the first call of the model, before any module runs, ``read`` takes values from the
     model and `_Reached` stops the tracing of the step function. A recorded scan reads the
     model before its first step and after its last one this way.

   Only the calls of the outermost model count. Controllers nested in it, such as the neurons
   of a brain, are not counted.

   :param read: Called with the model at its first call. What it returns is returned by `run`.
   :type read: callable, optional

   .. attribute:: calls

      Calls of the model within the context.

      :type: int

   :raises RuntimeError: When entered while another probe context is open.


   .. py:method:: run(f, *args)

      Calls ``f`` up to its first call of the model, and returns what ``read`` read there.

      :raises RuntimeError: When ``f`` returns without calling the model.



