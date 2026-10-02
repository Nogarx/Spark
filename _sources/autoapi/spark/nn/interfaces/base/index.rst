spark.nn.interfaces.base
========================

.. py:module:: spark.nn.interfaces.base


Attributes
----------

.. autoapisummary::

   spark.nn.interfaces.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.base.InterfaceOutput
   spark.nn.interfaces.base.InterfaceConfig
   spark.nn.interfaces.base.Interface


Module Contents
---------------

.. py:class:: InterfaceOutput

   Bases: :py:obj:`TypedDict`


   Output ports of an interface.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: InterfaceConfig

   Bases: :py:obj:`spark.core.config.DefaultSparkConfig`


   Base configuration for interfaces.


.. py:data:: ConfigT

.. py:class:: Interface(config = None, **kwargs)

   Bases: :py:obj:`spark.core.module.SparkModule`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for interfaces.

   An interface sits between a network and something that is not a network. Unlike a
   `Component`, it holds no neuronal state: it converts, routes or summarizes payloads.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: InterfaceConfig

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Declared by the concrete interface through the signature of its ``__call__``.

   :Output Ports: **\*\*outputs** (*SparkPayload*) -- Declared by the concrete interface through the TypedDict its ``__call__`` returns.

   .. seealso::

      :py:obj:`InputInterface`
          Turns an external signal into spikes.

      :py:obj:`OutputInterface`
          Turns spikes into a continuous signal.

      :py:obj:`ControlInterface`
          Routes and combines payloads inside a network.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Runs the interface operation.

      :param \*args: Inputs, as declared by the concrete interface.
      :type \*args: SparkPayload
      :param \*\*kwargs: Inputs, as declared by the concrete interface.

      :returns: Dictionary of output ports, as declared by the concrete interface.
      :rtype: InterfaceOutput



