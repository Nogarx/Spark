spark.nn.components.plasticity
==============================

.. py:module:: spark.nn.components.plasticity


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/plasticity/base/index
   /autoapi/spark/nn/components/plasticity/btsp/index
   /autoapi/spark/nn/components/plasticity/hebbian_rule/index
   /autoapi/spark/nn/components/plasticity/modulated/index
   /autoapi/spark/nn/components/plasticity/quadruplet_rule/index
   /autoapi/spark/nn/components/plasticity/short_term/index
   /autoapi/spark/nn/components/plasticity/zenke_rule/index


Classes
-------

.. autoapisummary::

   spark.nn.components.plasticity.Plasticity
   spark.nn.components.plasticity.PlasticityConfig
   spark.nn.components.plasticity.PlasticityOutput
   spark.nn.components.plasticity.ModulatedPlasticity
   spark.nn.components.plasticity.ModulatedPlasticityConfig
   spark.nn.components.plasticity.ZenkeRule
   spark.nn.components.plasticity.ZenkeRuleConfig
   spark.nn.components.plasticity.ModulatedZenkeRule
   spark.nn.components.plasticity.ModulatedZenkeRuleConfig
   spark.nn.components.plasticity.HebbianRule
   spark.nn.components.plasticity.HebbianRuleConfig
   spark.nn.components.plasticity.OjaRule
   spark.nn.components.plasticity.OjaRuleConfig
   spark.nn.components.plasticity.ModulatedHebbianRule
   spark.nn.components.plasticity.ModulatedHebbianRuleConfig
   spark.nn.components.plasticity.ModulatedOjaRule
   spark.nn.components.plasticity.ModulatedOjaRuleConfig
   spark.nn.components.plasticity.QuadrupletRule
   spark.nn.components.plasticity.QuadrupletRuleConfig
   spark.nn.components.plasticity.ModulatedQuadrupletRule
   spark.nn.components.plasticity.ModulatedQuadrupletRuleConfig
   spark.nn.components.plasticity.BTSPRule
   spark.nn.components.plasticity.BTSPRuleConfig
   spark.nn.components.plasticity.ModulatedBTSPRule
   spark.nn.components.plasticity.ModulatedBTSPRuleConfig


Package Contents
----------------

.. py:class:: Plasticity(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for plasticity rules.

   A plasticity rule reads the presynaptic spikes, the postsynaptic spikes and the current
   weights, and returns the weights for the next step. It does not own the weights: the
   result is written back onto the synapse through an effect declared in the enclosing
   controller.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: PlasticityConfig, optional

   :Input Ports: * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay.
                 * **post_spikes** (*SpikeArray*) -- Postsynaptic spikes emitted on this step.
                 * **kernel** (*FloatArray*) -- Current synaptic weights, read from the synapse.

   :Output Ports: **kernel** (*FloatArray*) -- The updated weights, written back onto the synapse as an effect.

   .. rubric:: Notes

   A rule parameter may be given per connection type rather than as a single value. Passing
   a 4-tuple selects one value for each of the excitatory-excitatory, excitatory-inhibitory,
   inhibitory-excitatory and inhibitory-inhibitory pairs, in the order ``(EE, EI, IE, II)``.
   The pairing is read from the inhibition masks the spikes carry and is available as the
   ``synaptic_mask`` property, with ``synaptic_mask_map`` giving the names.

   Spikes are reshaped to broadcast against the kernel before the rule is evaluated: the
   presynaptic axes are placed last and the postsynaptic axes first. A rule is therefore
   written as an elementwise expression over the kernel shape.

   .. seealso::

      :py:obj:`HebbianRule`
          Pair-based potentiation.

      :py:obj:`OjaRule`
          Hebbian potentiation with multiplicative normalization.

      :py:obj:`QuadrupletRule`
          Four-term rule driven by an external signal.

      :py:obj:`ModulatedHebbianRule`
          Hebbian rule gated by an external signal.

      :py:obj:`ZenkeRule`
          Triplet rule with heterosynaptic and transmitter-induced terms.

      :py:obj:`RUShortTermPlasticity`
          Short term depression of the current, not of the weights.


   .. py:property:: synaptic_mask
      :type: spark.core.payloads.IntegerMask



   .. py:property:: synaptic_mask_map
      :type: dict[int, str]



   .. py:method:: __call__(pre_spikes, post_spikes, kernel)

      Computes the weights for the next step.

      :param pre_spikes: Presynaptic spikes, after any conduction delay.
      :type pre_spikes: SpikeArray
      :param post_spikes: Postsynaptic spikes emitted on this step.
      :type post_spikes: SpikeArray
      :param kernel: Current synaptic weights.
      :type kernel: FloatArray

      :returns: Dictionary with one entry, ``kernel``, the updated weights.
      :rtype: PlasticityOutput



.. py:class:: PlasticityConfig

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Base configuration for plasticity rules.

   Carries no field of its own. Concrete rules declare their own parameters.


.. py:class:: PlasticityOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a plasticity rule.

   .. attribute:: kernel

      The updated synaptic weights, to be written back onto the synapse.

      :type: FloatArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: kernel
      :type:  spark.core.payloads.FloatArray


.. py:class:: ModulatedPlasticity(config = None, **kwargs)

   Third factor for plasticity rules.

   Mixin adding a ``modulation`` input port to a `Plasticity` subclass and scaling the
   weight change by whatever arrives on it. It must precede the rule in the base list::

       class ModulatedHebbianRule(ModulatedPlasticity, HebbianRule): ...

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ModulatedPlasticityConfig, optional

   :Input Ports: * **modulation** (*FloatArray*) -- Third factor scaling the whole update.
                 * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay.
                 * **post_spikes** (*SpikeArray*) -- Postsynaptic spikes emitted on this step.
                 * **kernel** (*FloatArray*) -- Current synaptic weights, read from the synapse.

   :Output Ports: **kernel** (*FloatArray*) -- The updated weights, written back onto the synapse as an effect.

   .. rubric:: Notes

   The rule supplies its terms through `Plasticity._kernel_delta`, and the mixin folds the
   third factor into the learning rate rather than into the result:

   .. math::
       W \leftarrow \mathrm{clip}\left(
           W + \Delta t \, (\eta M) \, \Delta W \right)

   A modulation of zero freezes the weights, and a negative modulation reverses the sign of
   the update. The bounds are those of the rule, so an upper bound declared by the rule is
   still applied.

   Scaling the rate rather than the result is what keeps the arithmetic associated the same
   way as in an unmodulated rule, which matters in half precision.

   The signal is expected in a shape that broadcasts against the kernel: a scalar, the
   presynaptic shape, the postsynaptic shape, or the kernel shape itself.

   The port is added by overriding ``__call__``, whose signature is what the framework reads
   the input ports from. `build` drops the extra argument before handing the rest to the
   rule.

   .. seealso::

      :py:obj:`Plasticity`
          The base the rule derives from, and the hooks it fills in.

      :py:obj:`ModulatedHebbianRule`
          `HebbianRule` with this mixin.

      :py:obj:`ModulatedQuadrupletRule`
          `QuadrupletRule` with this mixin.


   .. py:attribute:: config
      :type:  ModulatedPlasticityConfig


   .. py:method:: build(modulation, pre_spikes, post_spikes, kernel)


   .. py:method:: __call__(modulation, pre_spikes, post_spikes, kernel)

      Computes the weights for the next step, scaled by the third factor.

      :param modulation: Third factor scaling the whole update.
      :type modulation: FloatArray
      :param pre_spikes: Presynaptic spikes, after any conduction delay.
      :type pre_spikes: SpikeArray
      :param post_spikes: Postsynaptic spikes emitted on this step.
      :type post_spikes: SpikeArray
      :param kernel: Current synaptic weights.
      :type kernel: FloatArray

      :returns: Dictionary with one entry, ``kernel``, the updated weights.
      :rtype: PlasticityOutput



.. py:class:: ModulatedPlasticityConfig

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   Configuration for `ModulatedPlasticity`.

   Carries no field of its own. The third factor arrives on a port rather than through the
   configuration.


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


.. py:class:: ModulatedZenkeRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticityConfig`, :py:obj:`ZenkeRuleConfig`


   Configuration for `ModulatedZenkeRule`.

   Union of `ZenkeRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of its
   own, so the defaults are those of `ZenkeRuleConfig`.


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


.. py:class:: ModulatedHebbianRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticityConfig`, :py:obj:`HebbianRuleConfig`


   Configuration for `ModulatedHebbianRule`.

   Union of `HebbianRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of its
   own, so the defaults are those of `HebbianRuleConfig`.


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


.. py:class:: ModulatedOjaRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticityConfig`, :py:obj:`OjaRuleConfig`


   Configuration for `ModulatedOjaRule`.

   Union of `OjaRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of its own,
   so the defaults are those of `OjaRuleConfig`.


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


.. py:class:: ModulatedQuadrupletRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticityConfig`, :py:obj:`QuadrupletRuleConfig`


   Configuration for `ModulatedQuadrupletRule`.

   Union of `QuadrupletRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of
   its own.


.. py:class:: BTSPRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Behavioral timescale synaptic plasticity.

   Weights are moved by the overlap of two slow signals, both held by the rule: an
   eligibility trace laid down locally by presynaptic activity, and an instructive signal
   laid down by the dendritic plateau. Both last on the order of a second, so an input
   active seconds before or after a plateau is still modified by it.

   The update is bidirectional and depends on the weight rather than on postsynaptic
   firing: weak synapses potentiate towards ``w_max`` and strong ones depress towards
   zero, which makes the rule stable without a separate homeostatic term.

   The plateau carries no sign. An optional ``sign`` port gates the rule with the sign
   of the instructive event: positive runs the fitted rule, negative depresses every
   eligible input, zero leaves the weights alone.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: BTSPRuleConfig, optional

   :Input Ports: * **modulation** (*FloatArray*) -- The third factor, in ``[0, 1]``. Wire the ``plateau`` port of a dendrite onto it.
                   It drives the instructive signal.
                 * **pre_spikes** (*SpikeArray*) -- Presynaptic spikes, after any conduction delay. They drive the eligibility trace.
                 * **kernel** (*FloatArray*) -- Current synaptic weights, read from the synapse.
                 * **sign** (*FloatArray, optional*) -- Sign of the instructive event, in ``[-1, 1]``, one entry per postsynaptic unit or
                   a scalar. Values in between scale the gated terms, so pass the sign alone for a
                   hard gate. Without the port the rule is unsigned, as fitted.
                 * **inhibition_mask** (*BooleanMask, optional*) -- Marks the inhibitory units of the pool. Supplied by the enclosing `Neuron` without
                   being wired, and read only to type the connections. The update never reads the
                   postsynaptic state, which is what separates BTSP from a Hebbian rule.

   :Output Ports: **kernel** (*FloatArray*) -- The updated weights, written back onto the synapse as an effect.

   .. rubric:: Notes

   The rule holds both traces: the eligibility one per presynaptic unit, the instructive
   one per dendrite.

   .. math::
       \tau_{ET} \frac{dET}{dt} = -ET + \lambda_{ET} R \\
       \tau_{IS} \frac{dIS}{dt} = -IS + \lambda_{IS} P

   The two are driven differently because their inputs are. Presynaptic spikes are events,
   so each one adds ``et_scale`` to the eligibility trace. The plateau is a sustained
   level, so the instructive signal relaxes towards ``is_scale`` times it.

   Their pointwise product is the overlap that drives both processes:

   .. math::
       \frac{dW}{dt} = (W_{\max} - W) \, k_+ \, q_+(ET \cdot IS)
                     - W \, k_- \, q_-(ET \cdot IS)

   with :math:`q_\pm` a sigmoid rescaled to pass through 0 at an overlap of 0 and 1 at an
   overlap of 1:

   .. math::
       q(x) = \frac{\sigma(\beta (x - \alpha)) - \sigma(-\beta \alpha)}
                   {\sigma(\beta (1 - \alpha)) - \sigma(-\beta \alpha)}

   The overlap is clipped to ``[0, 1]``, which is the range the sigmoids were fitted on.
   The rates are given per second and the weights are bounded to ``[0, w_max]``. Both
   ``lambda`` factors are free normalizations, chosen so neither trace exceeds 1.

   For a constant overlap :math:`x` the weight relaxes to the fixed point

   .. math::
       W^*(x) = W_{\max} \frac{k_+ q_+(x)}{k_+ q_+(x) + k_- q_-(x)}

   at rate :math:`k_+ q_+(x) + k_- q_-(x)`. The target rises with the overlap, so an
   input active close to the plateau is pulled towards ``w_max`` and an eligible input far
   from it towards a low weight, whatever the postsynaptic activity. This is what keeps
   the total drive bounded without a homeostatic term.

   With the ``sign`` port wired, the sign :math:`s` gates the terms:

   .. math::
       \frac{dW}{dt} = [s]_+ \left( (W_{\max} - W) k_+ q_+ - W k_- q_- \right)
                     - [s]_- \, W k_{\mathrm{neg}} \, q_+

   A negative event therefore depresses eligible inputs in proportion to their weight,
   following the potentiation profile so that the inputs closest to the event lose the
   most. A zero sign gates plasticity off.

   The reference implementation applies each induction lap as one step with the weight
   held fixed during the lap. Here the equation is integrated every step, so one plateau
   closes a fraction :math:`1 - \exp(-k \int q \, dt)` of the gap to the target rather
   than :math:`k \int q \, dt` of it, and cannot overshoot.

   The modulation and the sign are expected in a shape that broadcasts against the
   kernel: a scalar, the postsynaptic shape, or the kernel shape itself.

   .. rubric:: References

   .. [1] A. D. Milstein et al., "Bidirectional synaptic plasticity rapidly modifies
          hippocampal representations", eLife 10, e73046, 2021.
          https://doi.org/10.7554/eLife.73046


   .. py:attribute:: config
      :type:  BTSPRuleConfig


   .. py:method:: build(modulation, pre_spikes, kernel, sign = None, inhibition_mask = None, post_spikes = None)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(modulation, pre_spikes, kernel, sign = None, inhibition_mask = None, post_spikes = None)

      Computes the weights for the next step.

      :param modulation: The third factor, in ``[0, 1]``. Wire a dendritic ``plateau`` onto it.
      :type modulation: FloatArray
      :param pre_spikes: Presynaptic spikes, after any conduction delay.
      :type pre_spikes: SpikeArray
      :param kernel: Current synaptic weights.
      :type kernel: FloatArray
      :param sign: Sign of the instructive event, in ``[-1, 1]``. Gates the rule.
      :type sign: FloatArray, optional
      :param inhibition_mask: Marks the inhibitory units of the pool. Not read by the update.
      :type inhibition_mask: BooleanMask, optional
      :param post_spikes: Postsynaptic spikes. Read only when ``et_post_tau`` > 0, for the coincidence
                          eligibility.
      :type post_spikes: SpikeArray, optional

      :returns: Dictionary with one entry, ``kernel``, the updated weights.
      :rtype: PlasticityOutput



.. py:class:: BTSPRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   Configuration for `BTSPRule`.

   Every parameter may be given as a single value or as a 4-tuple holding one value per
   connection type, in the order ``(EE, EI, IE, II)``.

   :param et_tau: Decay constant of the eligibility trace, in ms.
   :type et_tau: PlasticityParamLike or Initializer, default 863.91
   :param et_scale: Increment added to the eligibility trace by each presynaptic spike. Pick it so the
                    trace peaks near 1 at the presynaptic rates in use: a rate ``f`` sustained for
                    longer than ``et_tau`` settles near ``et_scale * f * et_tau / 1000``.
   :type et_scale: PlasticityParamLike or Initializer, default 0.05
   :param is_tau: Decay constant of the instructive signal, in ms.
   :type is_tau: PlasticityParamLike or Initializer, default 542.76
   :param is_scale: Gain on the plateau entering the instructive signal. The default normalizes a
                    300 ms plateau to a peak of 1, which is the plateau duration the rule was fitted
                    on. For a plateau of duration ``D`` use ``1 / (1 - exp(-D / is_tau))``.
   :type is_scale: PlasticityParamLike or Initializer, default 2.36
   :param w_max: Upper bound on the weights, and the target the potentiation term pulls towards. The
                 fitted value is normalized, in multiples of the initial weight, while a synaptic
                 kernel is in pA. Scale it into the units of the kernel it bounds, or the first step
                 clips the whole kernel down to it.
   :type w_max: PlasticityParamLike or Initializer, default 4.02
   :param k_pot: Potentiation rate, per second.
   :type k_pot: PlasticityParamLike or Initializer, default 2.27
   :param k_dep: Depression rate, per second.
   :type k_dep: PlasticityParamLike or Initializer, default 0.33
   :param k_neg: Depression rate under a negative instructive sign, per second. Read only when the
                 ``sign`` port is wired. Defaults to the potentiation rate, so that one negative
                 event unlearns as much as one positive event learns.
   :type k_neg: PlasticityParamLike or Initializer, default 2.27
   :param alpha_pot: Half maximum of the potentiation sigmoid, in units of the overlap.
   :type alpha_pot: PlasticityParamLike or Initializer, default 0.24
   :param beta_pot: Slope of the potentiation sigmoid.
   :type beta_pot: PlasticityParamLike or Initializer, default 30.32
   :param alpha_dep: Half maximum of the depression sigmoid, in units of the overlap.
   :type alpha_dep: PlasticityParamLike or Initializer, default 0.09
   :param beta_dep: Slope of the depression sigmoid. The fitted value is steep enough that depression
                    is effectively switched on at ``alpha_dep``.
   :type beta_dep: PlasticityParamLike or Initializer, default 2260.61
   :param eta: Overall gain on the update. The rates are carried by ``k_pot`` and ``k_dep``, so
               this is left at one unless the whole rule is being scaled.
   :type eta: float or jax.Array, default 1.0


   .. py:attribute:: et_tau
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: et_scale
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: is_tau
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: is_scale
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: w_max
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: k_pot
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: k_dep
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: k_neg
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: et_post_tau
      :type:  float


   .. py:attribute:: alpha_pot
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: beta_pot
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: alpha_dep
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: beta_dep
      :type:  spark.nn.components.plasticity.base.PlasticityParamLike | spark.nn.initializers.base.Initializer


   .. py:attribute:: eta
      :type:  float | jax.Array


.. py:class:: ModulatedBTSPRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticity`, :py:obj:`BTSPRule`


   Three-factor BTSP rule.

   `BTSPRule` composed with `ModulatedPlasticity`.

   .. seealso::

      :py:obj:`BTSPRule`
          The same rule without the third factor.

      :py:obj:`ModulatedPlasticity`
          The mixin supplying the third factor.


   .. py:attribute:: config
      :type:  ModulatedBTSPRuleConfig


.. py:class:: ModulatedBTSPRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticityConfig`, :py:obj:`BTSPRuleConfig`


   Configuration for `ModulatedBTSPRule`.

   Union of `BTSPRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of its
   own, so the defaults are those of `BTSPRuleConfig`.


