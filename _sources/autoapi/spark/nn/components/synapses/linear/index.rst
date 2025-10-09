spark.nn.components.synapses.linear
===================================

.. py:module:: spark.nn.components.synapses.linear


Classes
-------

.. autoapisummary::

   spark.nn.components.synapses.linear.LinearSynapsesConfig
   spark.nn.components.synapses.linear.LinearSynapses


Module Contents
---------------

.. py:class:: LinearSynapsesConfig(**kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.base.SynanpsesConfig`


   LinearSynapses model configuration class.


   .. py:attribute:: units
      :type:  spark.core.shape.Shape


   .. py:attribute:: async_spikes
      :type:  bool


   .. py:attribute:: kernel_initializer
      :type:  spark.nn.initializers.kernel.KernelInitializerConfig


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


