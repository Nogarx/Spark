spark.nn.components.somas.adaptive
==================================

.. py:module:: spark.nn.components.somas.adaptive


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.adaptive.AdaptiveSomaConfig
   spark.nn.components.somas.adaptive.AdaptiveSoma


Module Contents
---------------

.. py:class:: AdaptiveSomaConfig

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Configuration for `AdaptiveSoma`.

   Each mechanism is enabled by its trigger parameter and disabled while that parameter is
   None. A disabled mechanism contributes no operations to the step.

   :param cooldown: Absolute refractory period, in ms. Enables refractoriness.
   :type cooldown: float or jax.Array or Initializer or None, default None
   :param clamp_duration: Time the membrane potential is held at the reset value after a spike, in ms. Enables
                          the potential clamp.
   :type clamp_duration: float or jax.Array or Initializer or None, default None
   :param threshold_delta: Threshold increment per spike, in mV. Enables threshold adaptation.
   :type threshold_delta: float or jax.Array or Initializer or None, default None
   :param threshold_tau: Decay constant of the threshold offset, in ms. Read only when ``threshold_delta`` is
                         set.
   :type threshold_tau: float or jax.Array or Initializer, default 20.0
   :param adaptation_delta: Adaptation current increment per spike, in pA. Enables the adaptation current.
   :type adaptation_delta: float or jax.Array or Initializer or None, default None
   :param adaptation_tau: Decay constant of the adaptation current, in ms. Read only when ``adaptation_delta``
                          is set.
   :type adaptation_tau: float or jax.Array or Initializer, default 100.0
   :param adaptation_subthreshold: Coupling of the adaptation current to the membrane potential, in nS. Read only when
                                   ``adaptation_delta`` is set.
   :type adaptation_subthreshold: float or jax.Array or Initializer, default 0.5


   .. py:attribute:: cooldown
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer | None


   .. py:attribute:: clamp_duration
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer | None


   .. py:attribute:: threshold_delta
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer | None


   .. py:attribute:: threshold_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: adaptation_delta
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer | None


   .. py:attribute:: adaptation_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: adaptation_subthreshold
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: AdaptiveSoma(config = None, **kwargs)

   Adaptation mechanisms for soma models.

   Mixin adding an absolute refractory period, a potential clamp, an adaptive threshold and
   an adaptation current to a `Soma` subclass. It must precede the model in the base list::

       class AdaptiveLeakySoma(AdaptiveSoma, LeakySoma): ...

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: AdaptiveSomaConfig, optional

   :Input Ports: * **current** (*CurrentArray*) -- Current delivered to the membrane, in pA. Gated off while the refractory period is active.
                 * **inhibition_mask** (*BooleanMask, optional*) -- Marks the inhibitory units. Passed through to the model unchanged.

   :Output Ports: **spikes** (*SpikeArray*) -- Non-zero where the potential crossed the adapted threshold and no veto applied.

   :Properties: **potential** (*PotentialArray*) -- Membrane potential, as held by the model this mixin extends. Read only.

   .. rubric:: Notes

   The mixin supplies four terms of the soma step. The membrane integration
   :math:`\Phi` comes from the model it is mixed into:

   .. math::
       I_{\mathrm{eff}} &= g_{\mathrm{in}} I + I_{\mathrm{off}} \\
       V' &= \Phi(V, I_{\mathrm{eff}}) \\
       V'' &= g_{\mathrm{dyn}} V' + (1 - g_{\mathrm{dyn}}) V_{\mathrm{reset}} \\
       s &= (V'' > \theta + \theta_{\mathrm{off}}) \wedge m

   Each trigger parameter drives the terms beside it:

   * ``cooldown`` drives :math:`g_{\mathrm{in}}` and :math:`m`.
   * ``clamp_duration`` drives :math:`g_{\mathrm{dyn}}`.
   * ``threshold_delta`` drives :math:`\theta_{\mathrm{off}}`.
   * ``adaptation_delta`` drives :math:`I_{\mathrm{off}}`.

   Refractoriness gates the input current off and vetoes the spikes for ``cooldown`` after a
   spike. The potential clamp holds the potential at ``potential_reset`` for
   ``clamp_duration``. Both share one spike counter, saturating at the longer of the two.

   The threshold offset decays exponentially and is incremented by ``threshold_delta`` on
   every spike, with :math:`\alpha_\theta = \exp(-\Delta t / \tau_\theta)`:

   .. math::
       \theta_{\mathrm{off}} \leftarrow \alpha_\theta \theta_{\mathrm{off}}
                                      + \Delta\theta \, s

   The adaptation current is subtracted from the input current, so
   :math:`I_{\mathrm{off}} = -w`. It couples to the post-reset potential and is incremented
   on every spike, with :math:`a` the subthreshold coupling and :math:`b` the increment:

   .. math::
       w \leftarrow w + \frac{\Delta t}{\tau_w} \left( -w + a V \right) + b \, s

   Mechanism state is read at the start of the step and updated in `_after_spike` from the
   spikes and the membrane potential of that step. The hooks call `super`, so a model that
   overrides the same hooks, such as `IzhikevichSoma`, keeps working when extended.

   .. seealso::

      :py:obj:`AdaptiveLeakySoma`
          Leaky membrane with these mechanisms.

      :py:obj:`AdaptiveExponentialSoma`
          Exponential membrane with these mechanisms (AdEx).

      :py:obj:`AdaptiveIzhikevichSoma`
          Izhikevich membrane with these mechanisms.


   .. py:attribute:: config
      :type:  AdaptiveSomaConfig


   .. py:attribute:: has_refraction


   .. py:attribute:: has_potential_clamp


   .. py:attribute:: has_adaptive_threshold


   .. py:attribute:: has_adaptation_current


   .. py:method:: build(**abc_args)


   .. py:method:: reset()

      Resets component state.



