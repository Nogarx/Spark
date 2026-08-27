spark.nn.interfaces.input.base
==============================

.. py:module:: spark.nn.interfaces.input.base


Attributes
----------

.. autoapisummary::

   spark.nn.interfaces.input.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.input.base.InputInterfaceOutput
   spark.nn.interfaces.input.base.InputInterfaceConfig
   spark.nn.interfaces.input.base.InputInterface


Module Contents
---------------

.. py:class:: InputInterfaceOutput

   Bases: :py:obj:`TypedDict`


   Output ports of an input interface.

   .. attribute:: spikes

      The encoded signal.

      :type: SpikeArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: spikes
      :type:  spark.core.payloads.SpikeArray


.. py:class:: InputInterfaceConfig

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Base configuration for input interfaces.


.. py:data:: ConfigT

.. py:class:: InputInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for input interfaces.

   An input interface encodes a continuous signal as spikes, which is what lets a network
   read data that did not come from a network.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: InputInterfaceConfig

   :Input Ports: **signal** (*FloatArray*) -- Value to encode.

   :Output Ports: **spikes** (*SpikeArray*) -- The encoded signal.

   .. seealso::

      :py:obj:`PoissonSpiker`
          Stochastic rate encoding.

      :py:obj:`LinearSpiker`
          Deterministic rate encoding.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Encodes the signal as spikes.

      :param \*args: Inputs, as declared by the concrete interface.
      :type \*args: SparkPayload
      :param \*\*kwargs: Inputs, as declared by the concrete interface.

      :returns: Dictionary with one entry, ``spikes``.
      :rtype: InputInterfaceOutput



