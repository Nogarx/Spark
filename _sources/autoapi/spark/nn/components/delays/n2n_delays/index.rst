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

.. py:class:: N2NDelaysConfig

   Bases: :py:obj:`spark.nn.components.delays.n_delays.NDelaysConfig`


   Configuration for `N2NDelays`.

   :param units: Shape of the postsynaptic pool.
   :type units: tuple of int
   :param max_delay: Longest delay the buffer can hold, in ms. The buffer holds ``ceil(max_delay / dt)``
                     steps, which bounds every drawn delay.
   :type max_delay: float, default 8.0
   :param delays: Delay of every (postsynaptic, presynaptic) pair, in steps. Drawn over
                  ``[1, buffer_size]`` when an initializer is given.
   :type delays: jax.Array or Initializer, default UniformInitializerConfig()


   .. py:attribute:: units
      :type:  tuple[int, ...]


.. py:class:: N2NDelays(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.delays.base.Delays`


   Conduction delay attached to the pair of units.

   Every connection carries its own delay, so one presynaptic spike reaches each of its
   targets at a different time. If unit A fires and B, C and D listen to A, they see the
   spike ``k_BA``, ``k_CA`` and ``k_DA`` steps later.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: N2NDelaysConfig, optional

   :Input Ports: **in_spikes** (*SpikeArray*) -- Spikes emitted on this step.

   :Output Ports: **out_spikes** (*SpikeArray*) -- Spikes due on this step, of shape ``units + in_spikes.shape`` and marked asynchronous. A
                  synapse consuming it sums over the presynaptic axes only.

   :Properties: **kernel** (*IntegerArray*) -- Delay of every (postsynaptic, presynaptic) pair, in steps rather than in ms. Writable.

   .. rubric:: Notes

   Spikes are held in the same bit-packed ring buffer as `NDelays`. Only the kernel grows: it
   holds one delay per pair rather than one per presynaptic unit.

   .. seealso::

      :py:obj:`NDelays`
          One delay per presynaptic unit.


   .. py:attribute:: config
      :type:  N2NDelaysConfig


   .. py:method:: build(in_spikes)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



   .. py:method:: get_dense()

      Convert bitmask to dense vector (aligned with MSB-first packing).



   .. py:method:: __call__(in_spikes)

      Stores the incoming spikes and returns the ones due on this step.

      :param in_spikes: Spikes emitted on this step.
      :type in_spikes: SpikeArray

      :returns: Dictionary with one entry, ``out_spikes``, the spikes whose delay elapsed on this
                step.
      :rtype: DelaysOutput



