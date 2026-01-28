spark.nn.components.delays.base
===============================

.. py:module:: spark.nn.components.delays.base


Attributes
----------

.. autoapisummary::

   spark.nn.components.delays.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.components.delays.base.DelaysOutput
   spark.nn.components.delays.base.DelaysConfig
   spark.nn.components.delays.base.Delays


Module Contents
---------------

.. py:class:: DelaysOutput

   Bases: :py:obj:`TypedDict`


   Generic delay model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: out_spikes
      :type:  spark.core.payloads.SpikeArray


.. py:class:: DelaysConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Base synaptic delay configuration class.


.. py:data:: ConfigT

.. py:class:: Delays(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract synaptic delay model.


   .. py:method:: reset()
      :abstractmethod:


      Resets component state.



   .. py:method:: __call__(in_spikes)
      :abstractmethod:


      Execution method.



