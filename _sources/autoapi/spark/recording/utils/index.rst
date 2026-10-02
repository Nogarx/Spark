spark.recording.utils
=====================

.. py:module:: spark.recording.utils


Classes
-------

.. autoapisummary::

   spark.recording.utils.HeldInterrupt


Functions
---------

.. autoapisummary::

   spark.recording.utils.whole
   spark.recording.utils.integer
   spark.recording.utils.is_name
   spark.recording.utils.deliver_signal
   spark.recording.utils.hold_interrupt


Module Contents
---------------

.. py:function:: whole(value)

   Returns ``value`` as an int when it is a whole number, or None.

   Integers, NumPy integers included, and floats without a fractional part, such as ``1e5``,
   are whole numbers. Bools are not.


.. py:function:: integer(value, name, lowest = None, highest = None, owner = None)

   Returns ``value`` as an int, or raises a ValueError naming ``name``.

   ``value`` must be a whole number (`whole`), at least ``lowest`` and at most ``highest`` when
   they are given.

   :param value: Value to check.
   :type value: object
   :param name: Name of the argument, for the error message.
   :type name: str
   :param lowest: Bounds of the value, both included.
   :type lowest: int, optional
   :param highest: Bounds of the value, both included.
   :type highest: int, optional
   :param owner: What the argument belongs to, such as ``'Every'``, leading the error message.
   :type owner: str, optional

   :rtype: int

   :raises ValueError: When ``value`` is not a whole number, or is out of bounds.


.. py:function:: is_name(value)

   Returns whether ``value`` is a string of letters, digits, ``_``, ``.`` and ``-``, starting
   with a letter or a digit.


.. py:function:: deliver_signal(signum, handler, frame = None)

   Passes a received signal to ``handler`` as the process would without the recorder.


.. py:class:: HeldInterrupt(handler, armed)

   A SIGINT received while `hold_interrupt` held it.

   .. attribute:: frame

      The frame the signal interrupted, or None when none was received.

      :type: frame or None

   .. attribute:: handler

      The handler of SIGINT before the hold.

      :type: object

   .. attribute:: armed

      Whether a second SIGINT raises KeyboardInterrupt at once.

      :type: bool


   .. py:attribute:: frame
      :type:  Any
      :value: None



   .. py:attribute:: handler


   .. py:attribute:: armed


   .. py:method:: arm()

      Lets a second SIGINT raise KeyboardInterrupt at once, from now on.



   .. py:method:: deliver()

      Passes the SIGINT received, if any, to the handler before the hold (`deliver_signal`).



.. py:function:: hold_interrupt(armed = True)

   Holds SIGINT while the block runs. A second SIGINT raises KeyboardInterrupt at once.

   Without ``armed``, a second SIGINT is held as well until `HeldInterrupt.arm`. The SIGINT
   held is not delivered when the block ends: `HeldInterrupt.deliver` delivers it.

   :param armed: Whether a second SIGINT raises from the start of the block.
   :type armed: bool, default True

   :returns: Yields the `HeldInterrupt`, or None away from the main thread and while SIGINT is
             ignored, where nothing is held.
   :rtype: context manager


