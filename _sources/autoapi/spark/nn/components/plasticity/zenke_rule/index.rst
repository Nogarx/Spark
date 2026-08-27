spark.nn.components.plasticity.zenke_rule
=========================================

.. py:module:: spark.nn.components.plasticity.zenke_rule


Classes
-------

.. autoapisummary::

   spark.nn.components.plasticity.zenke_rule.ZenkeRuleConfig
   spark.nn.components.plasticity.zenke_rule.ZenkeRule
   spark.nn.components.plasticity.zenke_rule.ModulatedZenkeRuleConfig
   spark.nn.components.plasticity.zenke_rule.ModulatedZenkeRule


Module Contents
---------------

.. py:class:: ZenkeRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   Configuration for `ZenkeRule`.

   :param pre_tau: Decay constant of the presynaptic trace, in ms.
   :type pre_tau: float or jax.Array or Initializer, default 20.0
   :param post_tau: Decay constant of the fast postsynaptic trace, in ms.
   :type post_tau: float or jax.Array or Initializer, default 20.0
   :param post_slow_tau: Decay constant of the slow postsynaptic trace, in ms.
   :type post_slow_tau: float or jax.Array or Initializer, default 100.0
   :param target_tau: Decay constant of the weight target, in ms. Much slower than the other traces, so the
                      target moves on the time scale of consolidation rather than of activity.
   :type target_tau: float or jax.Array or Initializer, default 1200000.0
   :param a: Weight of the triplet potentiation term.
   :type a: float, default 2.0
   :param b: Weight of the doublet depression term. Negative.
   :type b: float, default -0.02
   :param c: Weight of the heterosynaptic term. Negative.
   :type c: float, default -400.0
   :param d: Weight of the transmitter-induced term.
   :type d: float, default 2e-05
   :param p: Depth of the double-well potential that holds the weight target.
   :type p: float, default 20.0
   :param eta: Learning rate.
   :type eta: float, default 0.1


   .. py:attribute:: pre_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: post_slow_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: target_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: a
      :type:  float | jax.Array


   .. py:attribute:: b
      :type:  float | jax.Array


   .. py:attribute:: c
      :type:  float | jax.Array


   .. py:attribute:: d
      :type:  float | jax.Array


   .. py:attribute:: p
      :type:  float | jax.Array


   .. py:attribute:: eta
      :type:  float | jax.Array


.. py:class:: ZenkeRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Zenke triplet rule with homeostatic consolidation.

   A triplet rule extended with two terms that keep the weights bounded on their own: a
   heterosynaptic term that pulls each weight towards a slowly moving target, and a
   transmitter-induced term that potentiates on presynaptic activity alone. The combination
   is stable without an external constraint on the weights.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ZenkeRuleConfig, optional

   :Input Ports: * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay.
                 * **post_spikes** (*SpikeArray*) -- Postsynaptic spikes emitted on this step.
                 * **kernel** (*FloatArray*) -- Current synaptic weights, read from the synapse.

   :Output Ports: **kernel** (*FloatArray*) -- The updated weights, written back onto the synapse as an effect.

   .. rubric:: Notes

   With :math:`x` the presynaptic trace, :math:`y` and :math:`\bar{y}` the fast and slow
   postsynaptic traces and :math:`\tilde{W}` the weight target,

   .. math::
       \Delta W = \eta \left(
           a \, x \bar{y} \, s_{\mathrm{post}}
         + b \, y \, s_{\mathrm{pre}}
         + c \, (W - \tilde{W}) \, y^3 s_{\mathrm{post}}
         + d \, s_{\mathrm{pre}} \right)

   applied as :math:`W \leftarrow \max(W + \Delta t \, \Delta W, 0)`. The target follows

   .. math::
       \tilde{W} \leftarrow \tilde{W} + \lambda_{\tilde{W}} \left(
           W - p \, \tilde{W} \left(\tfrac{1}{4} - \tilde{W}\right)
                               \left(\tfrac{1}{2} - \tilde{W}\right) - \tilde{W} \right)

   a double-well potential with minima that separate weak from strong synapses, so a weight
   settles into one of the two rather than drifting between them.

   .. rubric:: References

   .. [1] F. Zenke, E. J. Agnes and W. Gerstner, "Diverse Synaptic Plasticity Mechanisms
          Orchestrated to Form and Retrieve Memories in Spiking Neural Networks", Nature
          Communications 6, 6922, 2015. https://doi.org/10.1038/ncomms7922

   .. seealso::

      :py:obj:`HebbianRule`
          Pair-based potentiation without the homeostatic terms.


   .. py:attribute:: config
      :type:  ZenkeRuleConfig


   .. py:method:: build(pre_spikes, post_spikes, kernel)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



.. py:class:: ModulatedZenkeRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticityConfig`, :py:obj:`ZenkeRuleConfig`


   Configuration for `ModulatedZenkeRule`.

   Union of `ZenkeRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of its
   own, so the defaults are those of `ZenkeRuleConfig`.


.. py:class:: ModulatedZenkeRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticity`, :py:obj:`ZenkeRule`


   Zenke triplet rule scaled by a third factor.

   `ZenkeRule` composed with `ModulatedPlasticity`. The signal on the ``modulation`` port
   scales the whole update, the homeostatic terms included.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ModulatedZenkeRuleConfig, optional

   :Input Ports: * **modulation** (*FloatArray*) -- Third factor scaling the whole update.
                 * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay.
                 * **post_spikes** (*SpikeArray*) -- Postsynaptic spikes emitted on this step.
                 * **kernel** (*FloatArray*) -- Current synaptic weights, read from the synapse.

   :Output Ports: **kernel** (*FloatArray*) -- The updated weights, written back onto the synapse as an effect.

   .. rubric:: Notes

   .. math::
       W \leftarrow \max\left(
           W + \Delta t \, (\eta M) \, \Delta W, 0 \right)

   with :math:`\Delta W` the four terms of `ZenkeRule`.

   The heterosynaptic and transmitter-induced terms are what bound the weights, and they are
   scaled along with the rest. A modulation of zero therefore suspends the homeostasis as
   well as the learning, and a sustained negative modulation removes the bound rather than
   reversing it. The weight target keeps advancing on its own in either case, since it
   follows the weights rather than the update.

   .. rubric:: References

   .. [1] F. Zenke, E. J. Agnes and W. Gerstner, "Diverse Synaptic Plasticity Mechanisms
          Orchestrated to Form and Retrieve Memories in Spiking Neural Networks", Nature
          Communications 6, 6922, 2015. https://doi.org/10.1038/ncomms7922

   .. seealso::

      :py:obj:`ZenkeRule`
          The same rule without the third factor.

      :py:obj:`ModulatedPlasticity`
          The mixin supplying the third factor.


   .. py:attribute:: config
      :type:  ModulatedZenkeRuleConfig


