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
   spark.nn.components.synapses.RDTracedSynapses
   spark.nn.components.synapses.RDTracedSynapsesConfig
   spark.nn.components.synapses.RFSTracedSynapses
   spark.nn.components.synapses.RFSTracedSynapsesConfig


Package Contents
----------------

.. py:class:: Synanpses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract synapse model.

   Note that we require the kernel entries to be in pA for numerical stability, since most of the time we want to run in half-precision.
   However somas expect the current in nA so we need to rescale the output.

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
       units: tuple[int, ...]
       kernel: jax.Array | Initializer

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


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: get_kernel()


   .. py:method:: get_flat_kernel()


   .. py:method:: set_kernel(new_kernel)


.. py:class:: LinearSynapsesConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.base.SynanpsesConfig`


   LinearSynapses model configuration class.


   .. py:attribute:: units
      :type:  tuple[int, Ellipsis]


   .. py:attribute:: kernel
      :type:  jax.Array | spark.nn.initializers.base.Initializer


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



.. py:class:: TracedSynapsesConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapsesConfig`


   TracedSynapses model configuration class.


   .. py:attribute:: tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale
      :type:  float | jax.Array


   .. py:attribute:: base
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


