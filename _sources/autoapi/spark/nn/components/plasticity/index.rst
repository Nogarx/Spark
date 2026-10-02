spark.nn.components.plasticity
==============================

.. py:module:: spark.nn.components.plasticity


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/plasticity/base/index
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


