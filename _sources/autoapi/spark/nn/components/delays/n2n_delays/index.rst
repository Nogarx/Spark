spark.nn.components.delays.n2n_delays
=====================================

.. py:module:: spark.nn.components.delays.n2n_delays


Classes
-------

.. autoapisummary::

   spark.nn.components.delays.n2n_delays.N2NDelaysConfig
   spark.nn.components.delays.n2n_delays.N2NDelays


Module Contents
---------------

.. py:class:: N2NDelaysConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.delays.n_delays.NDelaysConfig`


   N2NDelays configuration class.


   .. py:attribute:: units
      :type:  tuple[int, Ellipsis]


.. py:class:: N2NDelays(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.delays.base.Delays`


   Data structure for spike storage and retrival for efficient neuron to neuron spike delay implementation.
   This synaptic delay model implements specific conduction delays between specific neruons.
   Example: Neuron A fires and neuron B, C, and D listens to A; neuron B recieves A's spikes I timesteps later,
            neuron C recieves A's spikes J timesteps later and neuron D recieves A's spikes K timesteps later.

   Init:
       units: tuple[int, ...]
       max_delay: float
       delays: jnp.ndarray | Initializer

   Input:
       in_spikes: SpikeArray

   Output:
       out_spikes: SpikeArray


   .. py:attribute:: config
      :type:  N2NDelaysConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: get_dense()

      Convert bitmask to dense vector (aligned with MSB-first packing).



   .. py:method:: __call__(in_spikes)

      Execution method.



