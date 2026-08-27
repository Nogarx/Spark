spark.nn.components.plasticity.hebbian_rule
===========================================

.. py:module:: spark.nn.components.plasticity.hebbian_rule


Classes
-------

.. autoapisummary::

   spark.nn.components.plasticity.hebbian_rule.HebbianRuleConfig
   spark.nn.components.plasticity.hebbian_rule.HebbianRule
   spark.nn.components.plasticity.hebbian_rule.OjaRuleConfig
   spark.nn.components.plasticity.hebbian_rule.OjaRule
   spark.nn.components.plasticity.hebbian_rule.ModulatedHebbianRuleConfig
   spark.nn.components.plasticity.hebbian_rule.ModulatedHebbianRule
   spark.nn.components.plasticity.hebbian_rule.ModulatedOjaRuleConfig
   spark.nn.components.plasticity.hebbian_rule.ModulatedOjaRule


Module Contents
---------------

.. py:class:: HebbianRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   Configuration for `HebbianRule`.

   :param pre_tau: Decay constant of the presynaptic trace, in ms. May be a 4-tuple, one value per
                   connection type.
   :type pre_tau: float or jax.Array or Initializer, default 20.0
   :param post_tau: Decay constant of the postsynaptic trace, in ms. May be a 4-tuple, one value per
                    connection type.
   :type post_tau: float or jax.Array or Initializer, default 20.0
   :param eta: Learning rate.
   :type eta: float, default 0.01


   .. py:attribute:: pre_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: eta
      :type:  float


.. py:class:: HebbianRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Pair-based Hebbian rule.

   Every presynaptic spike is potentiated in proportion to the postsynaptic trace, and every
   postsynaptic spike in proportion to the presynaptic trace. The rule only potentiates, so
   the weights grow without bound unless something else keeps them in check.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: HebbianRuleConfig

   :Input Ports: * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay.
                 * **post_spikes** (*SpikeArray*) -- Postsynaptic spikes emitted on this step.
                 * **kernel** (*FloatArray*) -- Current synaptic weights, read from the synapse.

   :Output Ports: **kernel** (*FloatArray*) -- The updated weights, written back onto the synapse as an effect.

   .. rubric:: Notes

   With :math:`x` the presynaptic trace and :math:`y` the postsynaptic trace,

   .. math::
       \Delta W = \eta \left( y \, s_{\mathrm{pre}} + x \, s_{\mathrm{post}} \right)

   applied as :math:`W \leftarrow \max(W + \Delta t \, \Delta W, 0)`. Each trace is scaled by
   the reciprocal of its own time constant, so its magnitude does not change with ``tau``.

   .. rubric:: References

   .. [1] W. Gerstner, W. M. Kistler, R. Naud and L. Paninski, "Neuronal Dynamics: From
          Single Neurons to Networks and Models of Cognition", Chapter 19.2, Hebbian Rate
          Models. https://neuronaldynamics.epfl.ch/online/Ch19.S2.html

   .. seealso::

      :py:obj:`OjaRule`
          The same potentiation with a normalizing term.


   .. py:attribute:: config
      :type:  HebbianRuleConfig


   .. py:method:: build(pre_spikes, post_spikes, kernel)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



.. py:class:: OjaRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   Configuration for `OjaRule`.

   :param post_tau: Decay constant of the postsynaptic trace, in ms. May be a 4-tuple, one value per
                    connection type.
   :type post_tau: float or jax.Array or Initializer, default 20.0
   :param eta: Learning rate.
   :type eta: float, default 0.1


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: eta
      :type:  float


.. py:class:: OjaRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Oja's rule.

   Hebbian potentiation with a decay term proportional to the weight and to the square of
   the postsynaptic trace. The decay bounds the weight vector, which plain Hebbian
   potentiation does not.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: OjaRuleConfig

   :Input Ports: * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay.
                 * **post_spikes** (*SpikeArray*) -- Postsynaptic spikes emitted on this step.
                 * **kernel** (*FloatArray*) -- Current synaptic weights, read from the synapse.

   :Output Ports: **kernel** (*FloatArray*) -- The updated weights, written back onto the synapse as an effect.

   .. rubric:: Notes

   With :math:`y` the postsynaptic trace,

   .. math::
       \Delta W = \eta \left( y \, s_{\mathrm{pre}} - W y^2 \right)

   applied as :math:`W \leftarrow \max(W + \Delta t \, \Delta W, 0)`.

   .. rubric:: References

   .. [1] E. Oja, "A Simplified Neuron Model as a Principal Component Analyzer", Journal of
          Mathematical Biology 15(3), 267-273, 1982. https://doi.org/10.1007/BF00275687

   .. seealso::

      :py:obj:`HebbianRule`
          The same potentiation without the normalizing term.


   .. py:attribute:: config
      :type:  OjaRuleConfig


   .. py:method:: build(pre_spikes, post_spikes, kernel)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



.. py:class:: ModulatedHebbianRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticityConfig`, :py:obj:`HebbianRuleConfig`


   Configuration for `ModulatedHebbianRule`.

   Union of `HebbianRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of its
   own, so the defaults are those of `HebbianRuleConfig`.


.. py:class:: ModulatedHebbianRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticity`, :py:obj:`HebbianRule`


   Three-factor Hebbian rule.

   `HebbianRule` composed with `ModulatedPlasticity`. The pair-based update is unchanged and
   the whole of it is scaled by the signal on the ``modulation`` port. The signal gates
   learning in time: the coincidence of the two spike trains sets which weights would change,
   and the modulation sets whether and how much they do.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ModulatedHebbianRuleConfig, optional

   :Input Ports: * **modulation** (*FloatArray*) -- Third factor scaling the whole update.
                 * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay.
                 * **post_spikes** (*SpikeArray*) -- Postsynaptic spikes emitted on this step.
                 * **kernel** (*FloatArray*) -- Current synaptic weights, read from the synapse.

   :Output Ports: **kernel** (*FloatArray*) -- The updated weights, written back onto the synapse as an effect.

   .. rubric:: Notes

   With :math:`x` the presynaptic trace and :math:`y` the postsynaptic trace,

   .. math::
       \Delta W = \eta \left( y \, s_{\mathrm{pre}} + x \, s_{\mathrm{post}} \right)

   applied as :math:`W \leftarrow \max(W + \Delta t \, M \, \Delta W, 0)`. A modulation of
   zero freezes the weights, and a negative modulation reverses the sign of the update.

   .. rubric:: References

   .. [1] W. Gerstner, M. Lehmann, V. Liakoni, D. Corneil and J. Brea, "Eligibility Traces and
          Plasticity on Behavioral Time Scales", Frontiers in Neural Circuits 12, 53, 2018.
          https://doi.org/10.3389/fncir.2018.00053

   .. seealso::

      :py:obj:`HebbianRule`
          The same rule without the third factor.

      :py:obj:`ModulatedPlasticity`
          The mixin supplying the third factor.

      :py:obj:`ModulatedQuadrupletRule`
          Modulated rule with separate spike and trace terms.


   .. py:attribute:: config
      :type:  ModulatedHebbianRuleConfig


.. py:class:: ModulatedOjaRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticityConfig`, :py:obj:`OjaRuleConfig`


   Configuration for `ModulatedOjaRule`.

   Union of `OjaRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of its own,
   so the defaults are those of `OjaRuleConfig`.


.. py:class:: ModulatedOjaRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticity`, :py:obj:`OjaRule`


   Oja's rule scaled by a third factor.

   `OjaRule` composed with `ModulatedPlasticity`. Both terms of the rule are scaled together
   by the signal on the ``modulation`` port, so the normalization follows the potentiation
   rather than acting on its own.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ModulatedOjaRuleConfig, optional

   :Input Ports: * **modulation** (*FloatArray*) -- Third factor scaling the whole update.
                 * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay.
                 * **post_spikes** (*SpikeArray*) -- Postsynaptic spikes emitted on this step.
                 * **kernel** (*FloatArray*) -- Current synaptic weights, read from the synapse.

   :Output Ports: **kernel** (*FloatArray*) -- The updated weights, written back onto the synapse as an effect.

   .. rubric:: Notes

   With :math:`y` the postsynaptic trace,

   .. math::
       W \leftarrow \max\left(
           W + \Delta t \, (\eta M) \left( y \, s_{\mathrm{pre}} - W y^2 \right), 0 \right)

   A modulation of zero freezes the weights entirely, the decay term included, so the weight
   vector is not renormalized while learning is gated off. A negative modulation reverses
   both terms, which grows the weights rather than bounding them.

   .. seealso::

      :py:obj:`OjaRule`
          The same rule without the third factor.

      :py:obj:`ModulatedPlasticity`
          The mixin supplying the third factor.

      :py:obj:`ModulatedHebbianRule`
          Modulated potentiation without the normalizing term.


   .. py:attribute:: config
      :type:  ModulatedOjaRuleConfig


