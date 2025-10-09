spark.nn.components.synapses.traced
===================================

.. py:module:: spark.nn.components.synapses.traced


Classes
-------

.. autoapisummary::

   spark.nn.components.synapses.traced.TracedSynapsesConfig
   spark.nn.components.synapses.traced.TracedSynapses
   spark.nn.components.synapses.traced.DoubleTracedSynapsesConfig
   spark.nn.components.synapses.traced.DoubleTracedSynapses


Module Contents
---------------

.. py:class:: TracedSynapsesConfig(**kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapsesConfig`


   TracedSynapses model configuration class.


   .. py:attribute:: tau
      :type:  float | jax.Array


   .. py:attribute:: scale
      :type:  float | jax.Array


   .. py:attribute:: base
      :type:  float | jax.Array


.. py:class:: TracedSynapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapses`


   Traced synaptic model.
   Output currents are computed as the trace of the dot product of the kernel with the input spikes.

   Init:
       units: Shape
       async_spikes: bool
       kernel_initializer: KernelInitializerConfig
       tau: float | jax.Array
       scale: float | jax.Array
       base: float | jax.Array

   Input:
       spikes: SpikeArray

   Output:
       currents: CurrentArray


   .. py:attribute:: config
      :type:  TracedSynapsesConfig


   .. py:attribute:: current_tracer
      :type:  spark.core.tracers.Tracer


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



.. py:class:: DoubleTracedSynapsesConfig(**kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapsesConfig`


   DoubleTracedSynapses model configuration class.


   .. py:attribute:: tau_1
      :type:  float | jax.Array


   .. py:attribute:: scale_1
      :type:  float | jax.Array


   .. py:attribute:: base_1
      :type:  float | jax.Array


   .. py:attribute:: tau_2
      :type:  float | jax.Array


   .. py:attribute:: scale_2
      :type:  float | jax.Array


   .. py:attribute:: base_2
      :type:  float | jax.Array


.. py:class:: DoubleTracedSynapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapses`


   Traced synaptic model.
   Output currents are computed as the trace of the dot product of the kernel with the input spikes.

   Init:
       units: Shape
       async_spikes: bool
       kernel_initializer: KernelInitializerConfig
       tau_1: float | jax.Array
       scale_1: float | jax.Array
       base_1: float | jax.Array
       tau_2: float | jax.Array
       scale_2: float | jax.Array
       base_2: float | jax.Array

   Input:
       spikes: SpikeArray

   Output:
       currents: CurrentArray


   .. py:attribute:: config
      :type:  DoubleTracedSynapsesConfig


   .. py:attribute:: current_tracer
      :type:  spark.core.tracers.DoubleTracer


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



