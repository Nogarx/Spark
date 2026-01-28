spark.nn.components.delays.n_delays
===================================

.. py:module:: spark.nn.components.delays.n_delays


Classes
-------

.. autoapisummary::

   spark.nn.components.delays.n_delays.NDelaysConfig
   spark.nn.components.delays.n_delays.NDelays


Module Contents
---------------

.. py:class:: NDelaysConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.delays.base.DelaysConfig`


   NDelays configuration class.


   .. py:attribute:: max_delay
      :type:  float


   .. py:attribute:: delays
      :type:  jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: NDelays(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.delays.base.Delays`


   Data structure for spike storage and retrival for efficient neuron spike delay implementation.
   This synaptic delay model implements a generic conduction delay of the outputs spikes of neruons.
   Example: Neuron A fires, every neuron that listens to A recieves its spikes K timesteps later,
           neuron B fires, every neuron that listens to B recieves its spikes L timesteps later.

   Init:
       max_delay: float
       delay_initializer: DelayInitializerConfig

   Input:
       in_spikes: SpikeArray

   Output:
       out_spikes: SpikeArray


   .. py:attribute:: config
      :type:  NDelaysConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: get_dense()

      Convert bitmask to dense vector (aligned with MSB-first packing).



   .. py:method:: __call__(in_spikes)

      Execution method.



