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

.. py:class:: LinearSynapsesConfig

   Bases: :py:obj:`spark.nn.components.synapses.base.SynanpsesConfig`


   Configuration for `LinearSynapses`.

   :param units: Shape of the postsynaptic pool.
   :type units: tuple of int
   :param kernel: Synaptic weights, in pA. The kernel is built with shape ``units + input_shape``.
   :type kernel: jax.Array or Initializer, default SparseUniformInitializerConfig()


   .. py:attribute:: units
      :type:  tuple[int, ...]


   .. py:attribute:: kernel
      :type:  jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: LinearSynapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.base.Synapses`


   Linear synapse.

   The postsynaptic current is the weighted sum of the incoming spikes, with no extra dynamics,
   equivalent to delta increments in current.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: LinearSynapsesConfig, optional

   :Input Ports: **spikes** (*SpikeArray*) -- Presynaptic spikes.

   :Output Ports: **currents** (*CurrentArray*) -- Current delivered to each postsynaptic unit, of shape ``units``.

   :Properties: **kernel** (*FloatArray*) -- Synaptic weights, in pA, of shape ``units + input_shape``. Writable, which is how a
                plasticity rule updates them.

   .. rubric:: Notes

   .. math::
       I_i = \sum_j W_{ij} s_j

   The kernel is built with shape ``units + input_shape`` and the sum runs over the
   presynaptic axes. Spikes marked asynchronous, as produced by `N2NDelays`, already carry
   one entry per (postsynaptic, presynaptic) pair; the postsynaptic axes are then matched
   elementwise and only the presynaptic axes are summed.

   .. rubric:: References

   .. [1] W. Gerstner, W. M. Kistler, R. Naud and L. Paninski, "Neuronal Dynamics: From
          Single Neurons to Networks and Models of Cognition", Chapter 1.3, Integrate-And-Fire
          Models. https://neuronaldynamics.epfl.ch/online/Ch1.S3.html

   .. seealso::

      :py:obj:`TracedSynapses`
          This model with an exponential postsynaptic current.


   .. py:attribute:: config
      :type:  LinearSynapsesConfig


   .. py:method:: build(spikes)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: get_kernel()


   .. py:method:: get_flat_kernel()


   .. py:method:: set_kernel(new_kernel)


