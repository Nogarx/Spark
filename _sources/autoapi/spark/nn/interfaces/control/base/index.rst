spark.nn.interfaces.control.base
================================

.. py:module:: spark.nn.interfaces.control.base


Attributes
----------

.. autoapisummary::

   spark.nn.interfaces.control.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.control.base.ControlInterfaceOutput
   spark.nn.interfaces.control.base.ControlInterfaceConfig
   spark.nn.interfaces.control.base.ControlInterface


Module Contents
---------------

.. py:class:: ControlInterfaceOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a control interface.

   .. attribute:: output

      The result of the operation. Its type follows the inputs.

      :type: SparkPayload

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: output
      :type:  spark.core.payloads.SparkPayload


.. py:class:: ControlInterfaceConfig

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Base configuration for control interfaces.


.. py:data:: ConfigT

.. py:class:: ControlInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for control interfaces.

   A control interface moves and transforms payloads around a graph rather than modelling anything.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ControlInterfaceConfig, optional

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Named by the graph that wires them, not by the signature.

   :Output Ports: **output** (*SparkPayload*) -- Result of the operation. Its type follows the inputs.

   .. seealso::

      :py:obj:`Concat`
          Joins several inputs of one type along an axis.

      :py:obj:`Sampler`
          Draws a subset of one input.

      :py:obj:`SignalTrace`
          Exponentially decaying trace of an input.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Runs the control operation.

      :param \*args: Inputs, named by the graph rather than by the signature.
      :type \*args: SparkPayload
      :param \*\*kwargs: Inputs, named by the graph rather than by the signature.

      :returns: Dictionary with one entry, ``output``.
      :rtype: ControlInterfaceOutput



