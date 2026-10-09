spark.core.recording_hooks
==========================

.. py:module:: spark.core.recording_hooks

.. autoapi-nested-parse::

   What `spark.recording` plugs into.

   The core and the controllers check these to know whether something records, and `spark.recording`
   sets them when it is imported and while a recorder is open. They live here so that nothing below
   `spark.recording` imports it.



Attributes
----------

.. autoapisummary::

   spark.core.recording_hooks.OPEN_RECORDERS
   spark.core.recording_hooks.TRACED_CALL
   spark.core.recording_hooks.PROBE_CONTEXT


Classes
-------

.. autoapisummary::

   spark.core.recording_hooks.RecordingHooks


Functions
---------

.. autoapisummary::

   spark.core.recording_hooks.active_probe_context
   spark.core.recording_hooks.set_recording_hooks
   spark.core.recording_hooks.recording_hooks


Module Contents
---------------

.. py:data:: OPEN_RECORDERS
   :type:  dict

   The open recorders of `spark.recording`, in the order they were opened, with the thread that opened each.

.. py:data:: TRACED_CALL
   :type:  contextvars.ContextVar[Any]

   The call of a `jit` function traced for a recorder, or None.

.. py:data:: PROBE_CONTEXT
   :type:  contextvars.ContextVar[spark.recording.probe_context.ProbeContext | None]

   The open `ProbeContext`, or None.

.. py:function:: active_probe_context()

   Returns the open probe context.

   :returns: The open probe context, or None when none is open.
   :rtype: ProbeContext or None


.. py:class:: RecordingHooks

   Bases: :py:obj:`Protocol`


   What `spark.recording` provides to `jit` and `scan` while a recorder is open.


   .. py:method:: call(function, args, kwargs)


   .. py:method:: scan(f, init, xs, length, reverse, unroll, split_transpose)


   .. py:method:: warmup(function, args, kwargs)


   .. py:method:: parts(function, parts, owner, device)


.. py:function:: set_recording_hooks(hooks)

   Installs the hooks of `spark.recording`. Called once, when it is imported.


.. py:function:: recording_hooks()

   Returns the hooks of `spark.recording`, or None before it is imported.


