spark.recording.scan
====================

.. py:module:: spark.recording.scan


Functions
---------

.. autoapisummary::

   spark.recording.scan.recorded_scan


Module Contents
---------------

.. py:function:: recorded_scan(f, init, xs = None, *, steps, probes, start = None, unroll = 1, pack_steps = True, split_transpose = False)

   Scans ``f`` over ``steps`` steps of a model and records ``probes``.

   ``f`` takes and returns what ``jax.lax.scan`` passes to it, and calls the model once per
   step. The model is found by the probe context, where it reports its call. `spark.scan` and
   `Runner` record with it.

   :param f: ``f(carry, x) -> (carry, y)``, one step of the model.
   :type f: callable
   :param init: As for ``jax.lax.scan``.
   :param xs: As for ``jax.lax.scan``.
   :param steps: Number of steps, at least 1. With ``xs``, the length of its leading axis.
   :type steps: int
   :param probes: What to record. Without probes, ``jax.lax.scan``.
   :type probes: tuple of Probe
   :param start: Where the call starts on the steps of the run, as `Recorder.start` gives it.
   :type start: Start or array, optional
   :param unroll: Passed to ``jax.lax.scan``.
   :type unroll: int or bool, default 1
   :param pack_steps: Whether to pack the values each step adds to the records into one byte row.
   :type pack_steps: bool, default True
   :param split_transpose: Passed to ``jax.lax.scan``.
   :type split_transpose: bool, default False

   :returns: * *carry, ys* -- As ``jax.lax.scan`` returns them.
             * **records** (*Packed or dict*) -- The records of the call in one buffer, and the values kept on the device for the
               snapshots and deltas with a group. An empty dictionary without probes.

   :raises RuntimeError: When ``f`` does not call the model once per step, or when a port a probe asks for is
       not produced during a step.

   .. rubric:: Notes

   The values of snapshots and deltas before the first step and after the last one are read
   from the model as ``f`` builds it from the carry, where it calls the model. ``f`` runs up to
   that call outside the scan, and what it computes there is not used otherwise.


