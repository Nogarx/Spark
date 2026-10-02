spark.nn.interfaces.output.base
===============================

.. py:module:: spark.nn.interfaces.output.base


Attributes
----------

.. autoapisummary::

   spark.nn.interfaces.output.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.output.base.OutputInterfaceOutput
   spark.nn.interfaces.output.base.OutputInterfaceConfig
   spark.nn.interfaces.output.base.OutputInterface


Module Contents
---------------

.. py:class:: OutputInterfaceOutput

   Bases: :py:obj:`TypedDict`


   Output ports of an output interface.

   .. attribute:: signal

      The decoded signal.

      :type: FloatArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: signal
      :type:  spark.core.payloads.FloatArray


.. py:class:: OutputInterfaceConfig

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Base configuration for output interfaces.


.. py:data:: ConfigT

.. py:class:: OutputInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for output interfaces.

   An output interface decodes spikes into a continuous signal, which is what lets something
   that is not a network read what a network produced.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: OutputInterfaceConfig, optional

   :Input Ports: **spikes** (*SpikeArray*) -- Spikes to decode.

   :Output Ports: **signal** (*FloatArray*) -- The decoded signal.

   .. seealso::

      :py:obj:`ExponentialIntegrator`
          Exponential filter of the incoming spikes.


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Decodes the incoming spikes into a signal.

      :param \*args: Inputs, as declared by the concrete interface.
      :type \*args: SpikeArray
      :param \*\*kwargs: Inputs, as declared by the concrete interface.

      :returns: Dictionary with one entry, ``signal``.
      :rtype: dict of str to SparkPayload



