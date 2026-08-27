spark.nn.components.plasticity.quadruplet_rule
==============================================

.. py:module:: spark.nn.components.plasticity.quadruplet_rule


Classes
-------

.. autoapisummary::

   spark.nn.components.plasticity.quadruplet_rule.QuadrupletRuleConfig
   spark.nn.components.plasticity.quadruplet_rule.QuadrupletRule
   spark.nn.components.plasticity.quadruplet_rule.ModulatedQuadrupletRuleConfig
   spark.nn.components.plasticity.quadruplet_rule.ModulatedQuadrupletRule


Module Contents
---------------

.. py:class:: QuadrupletRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   Configuration for `QuadrupletRule`.

   :param pre_tau: Decay constant of the presynaptic trace, in ms. May be a 4-tuple, one value per
                   connection type.
   :type pre_tau: float or jax.Array or Initializer, default 1
   :param post_tau: Decay constant of the postsynaptic trace, in ms. May be a 4-tuple, one value per
                    connection type.
   :type post_tau: float or jax.Array or Initializer, default (1.0, 2.0, 3.0, 4.0)
   :param q_alpha: Weight of the presynaptic spike term.
   :type q_alpha: float or jax.Array or Initializer, default 1.0
   :param q_beta: Weight of the postsynaptic trace times presynaptic spike term.
   :type q_beta: float or jax.Array or Initializer, default 1.0
   :param q_gamma: Weight of the postsynaptic spike term.
   :type q_gamma: float or jax.Array or Initializer, default (1.0, 2.0, 3.0, 4.0)
   :param q_delta: Weight of the presynaptic trace times postsynaptic spike term.
   :type q_delta: float or jax.Array or Initializer, default 1.0
   :param eta: Learning rate.
   :type eta: float, default 0.1
   :param max_clip: Upper bound on the weights, per connection type in the order ``(EE, EI, IE, II)``.
   :type max_clip: float or tuple of float, default (20.0, 20.0, 20.0, 20.0)


   .. py:attribute:: pre_tau
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike


   .. py:attribute:: post_tau
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike


   .. py:attribute:: q_alpha
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike


   .. py:attribute:: q_beta
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike


   .. py:attribute:: q_gamma
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike


   .. py:attribute:: q_delta
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike


   .. py:attribute:: eta
      :type:  float


   .. py:attribute:: max_clip
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike


.. py:class:: QuadrupletRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Four-term modulated plasticity rule.

   Standard pre-post pair plasticity rule model, with elegibility traces:
   Update is scaled by the signal on the ``modulation`` port.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: QuadrupletRuleConfig, optional

   :Input Ports: * **modulation** (*FloatArray*) -- Third factor scaling the whole update.
                 * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay.
                 * **post_spikes** (*SpikeArray*) -- Postsynaptic spikes emitted on this step.
                 * **kernel** (*FloatArray*) -- Current synaptic weights, read from the synapse.

   :Output Ports: **kernel** (*FloatArray*) -- The updated weights, written back onto the synapse as an effect.

   .. rubric:: Notes

   .. math::
       \Delta W = \eta \, M \left(
           \alpha \, s_{\mathrm{pre}}
         + \beta \, y \, s_{\mathrm{pre}}
         + \gamma \, s_{\mathrm{post}}
         + \delta \, x \, s_{\mathrm{post}} \right)

   applied as :math:`W \leftarrow \mathrm{clip}(W + \Delta t \, \Delta W, 0, W_{\max})`. The
   :math:`\alpha` and :math:`\gamma` terms fire on a spike alone and act as a baseline drift.

   .. rubric:: References

   .. [1] E. Oja, "Memory by a thousand rules: Automated discovery of functional multi-type plasticity rules
          reveals variety & degeneracy at the heart of learning", Basile Confavreux, Zoe P. M. Harrington, Maciej Kania,
          Poornima Ramesh, Anastasia N. Krouglova, Panos A. Bozelos,  Jakob H. Macke, Andrew M. Saxe, Pedro J. Gonçalves,
          Tim P. Vogels, bioRxiv 2025.05.28.656584; doi: https://doi.org/10.1101/2025.05.28.656584

   .. seealso::

      :py:obj:`ModulatedHebbianRule`
          Modulated Hebbian rule, with the two trace terms only.


   .. py:attribute:: config
      :type:  QuadrupletRuleConfig


   .. py:method:: build(pre_spikes, post_spikes, kernel)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



.. py:class:: ModulatedQuadrupletRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticityConfig`, :py:obj:`QuadrupletRuleConfig`


   Configuration for `ModulatedQuadrupletRule`.

   Union of `QuadrupletRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of
   its own.


.. py:class:: ModulatedQuadrupletRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticity`, :py:obj:`QuadrupletRule`


   Four-term rule scaled by a third factor.

   `QuadrupletRule` composed with `ModulatedPlasticity`. The four terms are unchanged and the
   whole update is scaled by the signal on the ``modulation`` port.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ModulatedQuadrupletRuleConfig, optional

   :Input Ports: * **modulation** (*FloatArray*) -- Third factor scaling the whole update.
                 * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay.
                 * **post_spikes** (*SpikeArray*) -- Postsynaptic spikes emitted on this step.
                 * **kernel** (*FloatArray*) -- Current synaptic weights, read from the synapse.

   :Output Ports: **kernel** (*FloatArray*) -- The updated weights, written back onto the synapse as an effect.

   .. rubric:: Notes

   .. math::
       W \leftarrow \mathrm{clip}\left(
           W + \Delta t \, M \, \Delta W, 0, W_{\max} \right)

   with :math:`\Delta W` the four terms of `QuadrupletRule`. The upper bound of the rule is
   still applied.

   .. seealso::

      :py:obj:`QuadrupletRule`
          The four terms, without the third factor.

      :py:obj:`ModulatedHebbianRule`
          Modulated Hebbian rule, with the two trace terms only.


   .. py:attribute:: config
      :type:  ModulatedQuadrupletRuleConfig


