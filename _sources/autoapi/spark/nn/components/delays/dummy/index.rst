spark.nn.components.delays.dummy
================================

.. py:module:: spark.nn.components.delays.dummy


Classes
-------

.. autoapisummary::

   spark.nn.components.delays.dummy.DummyDelaysConfig
   spark.nn.components.delays.dummy.DummyDelays


Module Contents
---------------

.. py:class:: DummyDelaysConfig(**kwargs)

   Bases: :py:obj:`spark.nn.components.delays.base.DelaysConfig`


   DummyDelays configuration class.


.. py:class:: DummyDelays(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.delays.base.Delays`


   A dummy delay that is equivalent to no delay at all, it simply forwards its input.
   This is a convinience module, that should be avoided whenever is possible and its
   only purpose is to simplify some scenarios.

   Init:

   Input:
       in_spikes: SpikeArray

   Output:
       out_spikes: SpikeArray


   .. py:attribute:: config
      :type:  DummyDelaysConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(in_spikes)

      Execution method.



