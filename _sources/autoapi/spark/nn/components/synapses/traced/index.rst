spark.nn.components.synapses.traced
===================================

.. py:module:: spark.nn.components.synapses.traced


Classes
-------

.. autoapisummary::

   spark.nn.components.synapses.traced.TracedSynapsesConfig
   spark.nn.components.synapses.traced.TracedSynapses
   spark.nn.components.synapses.traced.RDTracedSynapsesConfig
   spark.nn.components.synapses.traced.RDTracedSynapses
   spark.nn.components.synapses.traced.RFSTracedSynapsesConfig
   spark.nn.components.synapses.traced.RFSTracedSynapses


Module Contents
---------------

.. py:class:: TracedSynapsesConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapsesConfig`


   TracedSynapses model configuration class.


   .. py:attribute:: tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale
      :type:  float | jax.Array


   .. py:attribute:: base
      :type:  float | jax.Array


.. py:class:: TracedSynapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapses`


   Traced synaptic model.
   Output currents are computed as the trace of the dot product of the kernel with the input spikes.

   Init:
       units: tuple[int, ...]
       kernel: KernelInitializerConfig
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



.. py:class:: RDTracedSynapsesConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapsesConfig`


   RDTracedSynapses model configuration class.


   .. py:attribute:: tau_rise
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale_rise
      :type:  float | jax.Array


   .. py:attribute:: base_rise
      :type:  float | jax.Array


   .. py:attribute:: tau_decay
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale_decay
      :type:  float | jax.Array


   .. py:attribute:: base_decay
      :type:  float | jax.Array


.. py:class:: RDTracedSynapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapses`


   Rise-Decay traced synaptic model.
   Output currents are computed as the RDTrace of the dot product of the kernel with the input spikes.

   Init:
       units: tuple[int, ...]
       kernel: KernelInitializerConfig
       tau_rise: float | jax.Array
       scale_rise: float | jax.Array
       base_rise: float | jax.Array
       tau_decay: float | jax.Array
       scale_decay: float | jax.Array
       base_decay: float | jax.Array

   Input:
       spikes: SpikeArray

   Output:
       currents: CurrentArray


   .. py:attribute:: config
      :type:  RDTracedSynapsesConfig


   .. py:attribute:: current_tracer
      :type:  spark.core.tracers.RDTracer


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



.. py:class:: RFSTracedSynapsesConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapsesConfig`


   RFSTracedSynapses model configuration class.


   .. py:attribute:: alpha
      :type:  float | jax.Array


   .. py:attribute:: tau_rise
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale_rise
      :type:  float | jax.Array


   .. py:attribute:: base_rise
      :type:  float | jax.Array


   .. py:attribute:: tau_fast_decay
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale_fast_decay
      :type:  float | jax.Array


   .. py:attribute:: base_fast_decay
      :type:  float | jax.Array


   .. py:attribute:: tau_slow_decay
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale_slow_decay
      :type:  float | jax.Array


   .. py:attribute:: base_slow_decay
      :type:  float | jax.Array


.. py:class:: RFSTracedSynapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapses`


   Traced synaptic model.
   Output currents are computed as the trace of the dot product of the kernel with the input spikes.

   Init:
       units: tuple[int, ...]
       kernel: KernelInitializerConfig
       alpha: float | jax.Array
       tau_rise: float | jax.Array
       scale_rise: float | jax.Array
       base_rise: float | jax.Array
       tau_fast_decay: float | jax.Array
       scale_fast_decay: float | jax.Array
       base_fast_decay: float | jax.Array
       tau_slow_decay: float | jax.Array
       scale_slow_decay: float | jax.Array
       base_slow_decay: float | jax.Array

   Input:
       spikes: SpikeArray

   Output:
       currents: CurrentArray


   .. py:attribute:: config
      :type:  RFSTracedSynapsesConfig


   .. py:attribute:: current_tracer
      :type:  spark.core.tracers.RDTracer


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



