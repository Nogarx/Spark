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

.. py:class:: NDelaysConfig

   Bases: :py:obj:`spark.nn.components.delays.base.DelaysConfig`


   Configuration for `NDelays`.

   :param max_delay: Longest delay the buffer can hold, in ms. The buffer holds ``ceil(max_delay / dt)``
                     steps, which bounds every drawn delay.
   :type max_delay: float, default 8.0
   :param delays: Delay of every presynaptic unit, in steps. Drawn over ``[1, buffer_size]`` when an
                  initializer is given.
   :type delays: jax.Array or Initializer, default UniformInitializerConfig()


   .. py:attribute:: max_delay
      :type:  float


   .. py:attribute:: delays
      :type:  jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: NDelays(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.delays.base.Delays`


   Conduction delay attached to the presynaptic unit.

   Every spike a unit emits reaches all of its targets after the same number of steps. If
   unit A fires, everything listening to A sees the spike ``k_A`` steps later, with ``k_A``
   the delay of A alone.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: NDelaysConfig, optional

   :Input Ports: **in_spikes** (*SpikeArray*) -- Spikes emitted on this step.

   :Output Ports: **out_spikes** (*SpikeArray*) -- Spikes due on this step, of the same shape as the input.

   :Properties: **kernel** (*IntegerArray*) -- Delay of every presynaptic unit, in steps rather than in ms. Writable.

   .. rubric:: Notes

   Spikes are held in a ring buffer of ``ceil(max_delay / dt)`` steps, bit-packed eight units
   to a byte, so the buffer costs one bit per unit per step. Reading is a gather at the
   per-unit offset, which makes the cost independent of the delay values.

   .. seealso::

      :py:obj:`N2NDelays`
          One delay per (postsynaptic, presynaptic) pair.


   .. py:attribute:: config
      :type:  NDelaysConfig


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



