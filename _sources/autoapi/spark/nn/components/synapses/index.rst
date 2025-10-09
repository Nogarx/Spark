spark.nn.components.synapses
============================

.. py:module:: spark.nn.components.synapses


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/synapses/base/index
   /autoapi/spark/nn/components/synapses/linear/index
   /autoapi/spark/nn/components/synapses/traced/index


Classes
-------

.. autoapisummary::

   spark.nn.components.synapses.Synanpses
   spark.nn.components.synapses.SynanpsesOutput
   spark.nn.components.synapses.LinearSynapses
   spark.nn.components.synapses.LinearSynapsesConfig
   spark.nn.components.synapses.TracedSynapses
   spark.nn.components.synapses.TracedSynapsesConfig
   spark.nn.components.synapses.DoubleTracedSynapses
   spark.nn.components.synapses.DoubleTracedSynapsesConfig


Package Contents
----------------

.. py:class:: Synanpses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract synapse model.

   Init:

   Input:
       spikes: SpikeArray

   Output:
       currents: CurrentArray


   .. py:method:: get_kernel()
      :abstractmethod:



   .. py:method:: set_kernel(new_kernel)
      :abstractmethod:



   .. py:method:: __call__(spikes)

      Compute synanpse's currents.



.. py:class:: SynanpsesOutput

   Bases: :py:obj:`TypedDict`


   Generic synapses model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: currents
      :type:  spark.core.payloads.CurrentArray


.. py:class:: LinearSynapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.base.Synanpses`


   Linea synaptic model.
   Output currents are computed as the dot product of the kernel with the input spikes.

   Init:
       units: Shape
       async_spikes: bool
       kernel_initializer: KernelInitializerConfig

   Input:
       spikes: SpikeArray

   Output:
       currents: CurrentArray


   Reference:
       Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
       Gerstner W, Kistler WM, Naud R, Paninski L.
       Chapter 1.3 Integrate-And-Fire Models
       https://neuronaldynamics.epfl.ch/online/Ch1.S3.html


   .. py:attribute:: config
      :type:  LinearSynapsesConfig


   .. py:attribute:: async_spikes


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: get_kernel()


   .. py:method:: get_flat_kernel()


   .. py:method:: set_kernel(new_kernel)


.. py:class:: LinearSynapsesConfig(**kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.base.SynanpsesConfig`


   LinearSynapses model configuration class.


   .. py:attribute:: units
      :type:  spark.core.shape.Shape


   .. py:attribute:: async_spikes
      :type:  bool


   .. py:attribute:: kernel_initializer
      :type:  spark.nn.initializers.kernel.KernelInitializerConfig


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



.. py:class:: TracedSynapsesConfig(**kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapsesConfig`


   TracedSynapses model configuration class.


   .. py:attribute:: tau
      :type:  float | jax.Array


   .. py:attribute:: scale
      :type:  float | jax.Array


   .. py:attribute:: base
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


