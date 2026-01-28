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


   InputInterface model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: spikes
      :type:  spark.core.payloads.SpikeArray


.. py:class:: InputInterfaceConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Abstract InputInterface model configuration class.


.. py:data:: ConfigT

.. py:class:: InputInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract input interface model.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Transform the input signal into an Spike signal.



