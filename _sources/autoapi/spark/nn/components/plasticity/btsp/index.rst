spark.nn.components.plasticity.btsp
===================================

.. py:module:: spark.nn.components.plasticity.btsp


Classes
-------

.. autoapisummary::

   spark.nn.components.plasticity.btsp.BTSPRuleConfig
   spark.nn.components.plasticity.btsp.BTSPRule
   spark.nn.components.plasticity.btsp.ModulatedBTSPRuleConfig
   spark.nn.components.plasticity.btsp.ModulatedBTSPRule


Module Contents
---------------

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



.. py:class:: ModulatedBTSPRuleConfig

   Bases: :py:obj:`spark.nn.components.plasticity.modulated.ModulatedPlasticityConfig`, :py:obj:`BTSPRuleConfig`


   Configuration for `ModulatedBTSPRule`.

   Union of `BTSPRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of its
   own, so the defaults are those of `BTSPRuleConfig`.


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


