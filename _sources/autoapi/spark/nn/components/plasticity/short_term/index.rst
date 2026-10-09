spark.nn.components.plasticity.short_term
=========================================

.. py:module:: spark.nn.components.plasticity.short_term


Classes
-------

.. autoapisummary::

   spark.nn.components.plasticity.short_term.RUShortTermPlasticityConfig
   spark.nn.components.plasticity.short_term.RUShortTermPlasticity


Module Contents
---------------

.. py:class:: RUShortTermPlasticityConfig

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   Configuration for `RUShortTermPlasticity`.

   :param r_tau: Recovery constant of the resource trace, in ms.
   :type r_tau: float or jax.Array or Initializer, default 750.0
   :param u_tau: Decay constant of the usage trace, in ms.
   :type u_tau: float or jax.Array or Initializer, default 50.0
   :param u_scale: Fraction of the available resources a spike consumes.
   :type u_scale: float or jax.Array or Initializer, default 0.45


   .. py:attribute:: r_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: u_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: u_scale
      :type:  float


.. py:class:: RUShortTermPlasticity(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Resource-usage short term plasticity.

   Scales the synaptic current by the product of a usage variable and a resource variable.
   A spike raises usage and drains resources, and both relax back between spikes, so a burst
   is transmitted at a falling amplitude and recovers once the input pauses. Note that weights
   are not touched, this is a gain applied to the current.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: RUShortTermPlasticityConfig

   :Input Ports: * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay.
                 * **currents** (*CurrentArray*) -- Synaptic current before the short term gain is applied.

   :Output Ports: **kernel** (*CurrentArray*) -- The scaled current. The port is named ``kernel`` because every plasticity rule writes
                  through that port, but this rule returns a current rather than weights.

   .. rubric:: Notes

   With :math:`u` the usage trace, which rises by ``u_scale`` per spike and decays to zero,
   and :math:`r` the resource trace, which rests at one and is drained by :math:`u r` per
   spike, the transmitted current is

   .. math::
       I_{\mathrm{out}} = u \, r \, I_{\mathrm{in}}

   Facilitation and depression are set by the ratio of ``u_tau`` to ``r_tau``: a usage trace
   that outlives the resource trace facilitates, the reverse depresses.

   .. rubric:: References

   .. [1] M. Tsodyks, K. Pawelzik and H. Markram, "Neural Networks with Dynamic Synapses",
          Neural Computation 10(4), 821-835, 1998.
          https://doi.org/10.1162/089976698300017502
   .. [2] http://www.scholarpedia.org/article/Short-term_synaptic_plasticity


   .. py:attribute:: config
      :type:  RUShortTermPlasticityConfig


   .. py:method:: build(pre_spikes, currents)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(pre_spikes, currents)

      Applies the short term gain to the synaptic current.

      :param pre_spikes: Presynaptic spikes, after any conduction delay.
      :type pre_spikes: SpikeArray
      :param currents: Synaptic current before the gain is applied.
      :type currents: CurrentArray

      :returns: Dictionary with one entry, ``kernel``, carrying the scaled `CurrentArray`.
      :rtype: PlasticityOutput



