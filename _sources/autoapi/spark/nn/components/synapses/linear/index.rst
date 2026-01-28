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

.. py:class:: LinearSynapsesConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.base.SynanpsesConfig`


   LinearSynapses model configuration class.


   .. py:attribute:: units
      :type:  tuple[int, Ellipsis]


   .. py:attribute:: kernel
      :type:  jax.Array | spark.nn.initializers.base.Initializer


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


