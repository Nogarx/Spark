spark.recording.calls
=====================

.. py:module:: spark.recording.calls


Classes
-------

.. autoapisummary::

   spark.recording.calls.Layout
   spark.recording.calls.Learned


Functions
---------

.. autoapisummary::

   spark.recording.calls.replicated
   spark.recording.calls.check_memory
   spark.recording.calls.layout_of


Module Contents
---------------

.. py:function:: replicated(sharding)

   Returns a sharding that replicates a value on the devices of ``sharding``.

   Use to gracefully expose fragmented arrays (multi-host) to the recorded.


.. py:function:: check_memory(compiled, steps, probes)

   Memory check that warns when a compiled call needs more memory than available.


.. py:class:: Layout

   Bases: :py:obj:`NamedTuple`


   How the records of the calls of a function are laid out, from where its arguments are.

   .. attribute:: pack_steps

      Whether the values of a step are packed into one row. Not when the state is sharded
      across devices.

      :type: bool

   .. attribute:: sharding

      With several processes, where the records are replicated.

      :type: NamedSharding or None


   .. py:attribute:: pack_steps
      :type:  bool
      :value: True



   .. py:attribute:: sharding
      :type:  jax.sharding.NamedSharding | None
      :value: None



.. py:class:: Learned

   Bases: :py:obj:`NamedTuple`


   What the first trace of a function found for one value of its static arguments.

   .. attribute:: steps

      Steps of a call. 0 when the call runs no `spark.scan`.

      :type: int

   .. attribute:: layout

      How the records are laid out.

      :type: Layout


   .. py:attribute:: steps
      :type:  int


   .. py:attribute:: layout
      :type:  Layout


.. py:function:: layout_of(dynamic)

   Returns the layout of the records of a call with the dynamic arguments ``dynamic``, such as
   the state of the model.


