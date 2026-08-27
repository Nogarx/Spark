spark.nn.components.somas
=========================

.. py:module:: spark.nn.components.somas


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/somas/adaptive/index
   /autoapi/spark/nn/components/somas/base/index
   /autoapi/spark/nn/components/somas/exponential/index
   /autoapi/spark/nn/components/somas/izhikevich/index
   /autoapi/spark/nn/components/somas/leaky/index


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.Soma
   spark.nn.components.somas.SomaConfig
   spark.nn.components.somas.SomaOutput
   spark.nn.components.somas.AdaptiveSoma
   spark.nn.components.somas.AdaptiveSomaConfig
   spark.nn.components.somas.LeakySoma
   spark.nn.components.somas.LeakySomaConfig
   spark.nn.components.somas.AdaptiveLeakySoma
   spark.nn.components.somas.AdaptiveLeakySomaConfig
   spark.nn.components.somas.ExponentialSoma
   spark.nn.components.somas.ExponentialSomaConfig
   spark.nn.components.somas.AdaptiveExponentialSoma
   spark.nn.components.somas.AdaptiveExponentialSomaConfig
   spark.nn.components.somas.IzhikevichSoma
   spark.nn.components.somas.IzhikevichSomaConfig
   spark.nn.components.somas.AdaptiveIzhikevichSoma
   spark.nn.components.somas.AdaptiveIzhikevichSomaConfig


Package Contents
----------------

.. py:class:: Soma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for soma models.

   Base component representing membrane potential dynamics. Soma models subclass this component
   in order to implement specific some dynamics. Standard auxiliary methods are provided by this
   component.

       I_eff = _effective_current(I)
       V'    = _integrate(V, I_eff)
       V''   = _post_integrate(V')
       s     = (V'' > _effective_threshold()) and _spike_mask()
       V     = s ? V_reset : V''
               _after_spike(s)

   Only `_integrate` is required. The other hooks return their argument unchanged, so a
   model that is a membrane integration and nothing else defines one method.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: SomaConfig, optional

   :Input Ports: * **current** (*CurrentArray*) -- Current delivered to the membrane, in pA.
                 * **inhibition_mask** (*BooleanMask, optional*) -- Marks the inhibitory units. Supplied by the enclosing `Neuron`. It is stamped onto the
                   emitted spikes and does not affect the integration.

   :Output Ports: **spikes** (*SpikeArray*) -- Non-zero where the potential crossed the threshold on this step.

   :Properties: **potential** (*PotentialArray*) -- Membrane potential, relative to ``potential_rest``. Read only.

   .. rubric:: Notes

   A subclass owns a ``threshold`` and a ``potential_reset`` constant. A mechanism written
   against those, rather than against a particular integration, composes with every model in
   this package.

   Two kinds of subclass use the hooks. A model whose own state is integrated alongside the
   membrane, such as the recovery variable of `IzhikevichSoma`, updates that state in
   `_post_integrate` and `_after_spike`. `AdaptiveSoma` wraps an existing model by driving
   the current, the potential, the threshold and the spike veto.

   .. seealso::

      :py:obj:`AdaptiveSoma`
          Refractoriness, potential clamp, threshold and current adaptation.

      :py:obj:`LeakySoma`
          Leaky integrate-and-fire membrane.

      :py:obj:`ExponentialSoma`
          Exponential integrate-and-fire membrane.

      :py:obj:`IzhikevichSoma`
          Quadratic membrane with a recovery variable.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: build(current, inhibition_mask = None)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: potential()


   .. py:method:: reset()

      Resets neuron states to their initial values.



   .. py:method:: __call__(current, inhibition_mask = None)

      Advances the membrane one step and emits the spikes.

      :param current: Current delivered to the membrane, in pA.
      :type current: CurrentArray
      :param inhibition_mask: Marks the inhibitory units. Stamped onto the emitted spikes without affecting the
                              integration.
      :type inhibition_mask: BooleanMask or bool, optional

      :returns: Dictionary with one entry, ``spikes``, non-zero where the potential crossed the
                threshold.
      :rtype: SomaOutput



.. py:class:: SomaConfig

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Base configuration for soma models.

   Carries no field of its own. Concrete models declare their own parameters.


.. py:class:: SomaOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a soma model.

   .. attribute:: spikes

      One entry per unit, non-zero where the membrane potential crossed the threshold.

      :type: SpikeArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: spikes
      :type:  spark.core.payloads.SpikeArray


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


.. py:class:: LeakySoma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.base.Soma`


   Leaky integrate-and-fire soma.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: LeakySomaConfig

   :Input Ports: * **current** (*CurrentArray*) -- Current delivered to the membrane, in pA.
                 * **inhibition_mask** (*BooleanMask, optional*) -- Marks the inhibitory units. Supplied by the enclosing `Neuron`.

   :Output Ports: **spikes** (*SpikeArray*) -- Non-zero where the potential crossed the threshold on this step.

   :Properties: **potential** (*PotentialArray*) -- Membrane potential, relative to ``potential_rest``. Read only.

   .. rubric:: Notes

   Potentials are stored relative to ``potential_rest``, so a stored value of zero is rest.
   The membrane is integrated in closed form, with
   :math:`\alpha = \exp(-\Delta t / \tau_V)`:

   .. math::
       V_{t+1} = \alpha V_t + (1 - \alpha) R I_t

   .. rubric:: References

   .. [1] W. Gerstner, W. M. Kistler, R. Naud and L. Paninski, "Neuronal Dynamics: From
          Single Neurons to Networks and Models of Cognition", Chapter 1.3, Integrate-And-Fire
          Models. https://neuronaldynamics.epfl.ch/online/Ch1.S3.html

   .. seealso::

      :py:obj:`AdaptiveLeakySoma`
          This model with the adaptation mechanisms.


   .. py:attribute:: config
      :type:  LeakySomaConfig


   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



.. py:class:: LeakySomaConfig

   Bases: :py:obj:`spark.nn.components.somas.base.SomaConfig`


   Configuration for `LeakySoma`.

   :param potential_rest: Membrane rest potential, in mV.
   :type potential_rest: float or jax.Array or Initializer, default -60.0
   :param potential_reset: Membrane potential after a spike, in mV.
   :type potential_reset: float or jax.Array or Initializer, default -50.0
   :param potential_tau: Membrane potential decay constant, in ms.
   :type potential_tau: float or jax.Array or Initializer, default 20.0
   :param resistance: Membrane resistance, in GΩ.
   :type resistance: float or jax.Array or Initializer, default 0.1
   :param threshold: Spike threshold, in mV.
   :type threshold: float or jax.Array or Initializer, default -40.0


   .. py:attribute:: potential_rest
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: potential_reset
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: potential_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: resistance
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: threshold
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: AdaptiveLeakySoma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.adaptive.AdaptiveSoma`, :py:obj:`LeakySoma`


   Leaky integrate-and-fire soma with the adaptation mechanisms.

   `LeakySoma` composed with `AdaptiveSoma`, which adds an absolute refractory period, a
   potential clamp, an adaptive threshold and an adaptation current. Each is enabled by its
   trigger parameter and costs nothing while that parameter is None.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: AdaptiveLeakySomaConfig, optional

   :Input Ports: * **current** (*CurrentArray*) -- Current delivered to the membrane, in pA.
                 * **inhibition_mask** (*BooleanMask, optional*) -- Marks the inhibitory units. Supplied by the enclosing `Neuron`.

   :Output Ports: **spikes** (*SpikeArray*) -- Non-zero where the potential crossed the threshold on this step.

   :Properties: **potential** (*PotentialArray*) -- Membrane potential, relative to ``potential_rest``. Read only.

   .. seealso::

      :py:obj:`LeakySoma`
          The membrane integration, without the mechanisms.

      :py:obj:`AdaptiveSoma`
          The mechanisms and their equations.


   .. py:attribute:: config
      :type:  AdaptiveLeakySomaConfig


.. py:class:: AdaptiveLeakySomaConfig

   Bases: :py:obj:`spark.nn.components.somas.adaptive.AdaptiveSomaConfig`, :py:obj:`LeakySomaConfig`


   Configuration for `AdaptiveLeakySoma`.

   Union of `LeakySomaConfig` and `AdaptiveSomaConfig`. It declares no field of its own.


.. py:class:: ExponentialSoma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.base.Soma`


   Exponential integrate-and-fire soma.

   A leaky membrane with an added exponential term, which reproduces the upswing of a spike
   rather than firing on a hard threshold crossing. A spike is registered when the potential
   reaches ``threshold``, after which the potential is set to ``potential_reset``.

   Note that this is not the adaptive exponential (AdEx) model.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ExponentialSomaConfig, optional

   :Input Ports: * **current** (*CurrentArray*) -- Current delivered to the membrane, in pA.
                 * **inhibition_mask** (*BooleanMask, optional*) -- Marks the inhibitory units. Supplied by the enclosing `Neuron`.

   :Output Ports: **spikes** (*SpikeArray*) -- Non-zero where the potential crossed the threshold on this step.

   :Properties: **potential** (*PotentialArray*) -- Membrane potential, relative to ``potential_rest``. Read only.

   .. rubric:: Notes

   Potentials are stored relative to ``potential_rest``, so a stored value of zero is rest.
   With :math:`\Delta_T` the slope factor and :math:`V_{rh}` the rheobase threshold, the step
   is a forward Euler integration:

   .. math::
       V_{t+1} = V_t + \frac{\Delta t}{\tau_V} \left(
           -V_t + \Delta_T \exp\!\left(\frac{V_t - V_{rh}}{\Delta_T}\right) + R I_t \right)

   .. rubric:: References

   .. [1] N. Fourcaud-Trocmé, D. Hansel, C. van Vreeswijk and N. Brunel, "How Spike Generation
          Mechanisms Determine the Neuronal Response to Fluctuating Inputs", Journal of
          Neuroscience 23(37), 11628-11640, 2003.
          https://www.jneurosci.org/content/23/37/11628
   .. [2] W. Gerstner, W. M. Kistler, R. Naud and L. Paninski, "Neuronal Dynamics: From
          Single Neurons to Networks and Models of Cognition", Chapter 5.2, Exponential
          Integrate-and-Fire Model. https://neuronaldynamics.epfl.ch/online/Ch5.S2.html

   .. seealso::

      :py:obj:`AdaptiveExponentialSoma`
          This model with the adaptation mechanisms (AdEx).


   .. py:attribute:: config
      :type:  ExponentialSomaConfig


   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



.. py:class:: ExponentialSomaConfig

   Bases: :py:obj:`spark.nn.components.somas.base.SomaConfig`


   Configuration for `ExponentialSoma`.

   :param potential_rest: Membrane rest potential, in mV.
   :type potential_rest: float or jax.Array or Initializer, default -70.0
   :param potential_reset: Membrane potential after a spike, in mV.
   :type potential_reset: float or jax.Array or Initializer, default -51.0
   :param potential_tau: Membrane potential decay constant, in ms.
   :type potential_tau: float or jax.Array or Initializer, default 5.0
   :param resistance: Membrane resistance, in GΩ.
   :type resistance: float or jax.Array or Initializer, default 0.5
   :param threshold: Potential at which a spike is registered, in mV. This is the cutoff of the exponential
                     upswing, not the point where firing begins.
   :type threshold: float or jax.Array or Initializer, default -30.0
   :param rheobase_threshold: Rheobase threshold, in mV. The potential above which the exponential term dominates
                              and the upswing becomes irreversible.
   :type rheobase_threshold: float or jax.Array or Initializer, default -50.0
   :param spike_slope: Sharpness of spike initiation, in mV. Smaller values approach a hard threshold.
   :type spike_slope: float or jax.Array or Initializer, default 2.0


   .. py:attribute:: potential_rest
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: potential_reset
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: potential_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: resistance
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: threshold
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: rheobase_threshold
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: spike_slope
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: AdaptiveExponentialSoma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.adaptive.AdaptiveSoma`, :py:obj:`ExponentialSoma`


   Adaptive exponential integrate-and-fire soma (AdEx).

   `ExponentialSoma` composed with `AdaptiveSoma`. Setting ``adaptation_delta`` and
   ``adaptation_subthreshold`` gives the adaptation current of the AdEx model; the refractory
   period, the potential clamp and the adaptive threshold are available on the same terms as
   for any other soma.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: AdaptiveExponentialSomaConfig, optional

   :Input Ports: * **current** (*CurrentArray*) -- Current delivered to the membrane, in pA.
                 * **inhibition_mask** (*BooleanMask, optional*) -- Marks the inhibitory units. Supplied by the enclosing `Neuron`.

   :Output Ports: **spikes** (*SpikeArray*) -- Non-zero where the potential crossed the threshold on this step.

   :Properties: **potential** (*PotentialArray*) -- Membrane potential, relative to ``potential_rest``. Read only.

   .. rubric:: References

   .. [1] R. Brette and W. Gerstner, "Adaptive Exponential Integrate-and-Fire Model as an
          Effective Description of Neuronal Activity", Journal of Neurophysiology 94(5),
          3637-3642, 2005. https://doi.org/10.1152/jn.00686.2005

   .. seealso::

      :py:obj:`ExponentialSoma`
          The membrane integration, without the mechanisms.

      :py:obj:`AdaptiveSoma`
          The mechanisms and their equations.


   .. py:attribute:: config
      :type:  AdaptiveExponentialSomaConfig


.. py:class:: AdaptiveExponentialSomaConfig

   Bases: :py:obj:`spark.nn.components.somas.adaptive.AdaptiveSomaConfig`, :py:obj:`ExponentialSomaConfig`


   Configuration for `AdaptiveExponentialSoma`.

   Union of `ExponentialSomaConfig` and `AdaptiveSomaConfig`. It declares no field of its own.


.. py:class:: IzhikevichSoma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.base.Soma`


   Izhikevich soma.

   A quadratic membrane paired with a recovery variable. The pair reproduces a wide range of
   firing patterns, selected through ``recovery_timescale``, ``recovery_sensitivity``,
   ``recovery_update`` and ``potential_reset``. A spike is registered when the potential
   reaches ``threshold``, after which the potential is set to ``potential_reset`` and the
   recovery variable is incremented.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: IzhikevichSomaConfig, optional

   :Input Ports: * **current** (*CurrentArray*) -- Current delivered to the membrane, in pA.
                 * **inhibition_mask** (*BooleanMask, optional*) -- Marks the inhibitory units. Supplied by the enclosing `Neuron`.

   :Output Ports: **spikes** (*SpikeArray*) -- Non-zero where the potential crossed the threshold on this step.

   :Properties: **potential** (*PotentialArray*) -- Membrane potential, in mV. Read only.

   .. rubric:: Notes

   Unlike the other somas in this package, potentials are stored in absolute mV rather than
   relative to rest: the quadratic term is not invariant under a shift of the potential.

   With :math:`u` the recovery variable, the step is

   .. math::
       V_{t+1} &= V_t + \Delta t \left( 0.04 V_t^2 + 5 V_t + 140 - u_t + R I_t \right) \\
       u_{t+1} &= u_t + \Delta t \, a \left( b V_{t+1} - u_t \right)

   and a spike adds :math:`d` to :math:`u`. The constants 0.04, 5 and 140 are those of the
   original model and assume :math:`V` in mV and :math:`\Delta t` in ms.

   The recovery variable reads the potential produced by the same step, so it is updated in
   `_post_integrate` and `_after_spike` rather than alongside the membrane. Those hooks call
   `super`, which leaves `AdaptiveSoma` free to extend the model.

   .. rubric:: References

   .. [1] E. M. Izhikevich, "Simple Model of Spiking Neurons", IEEE Transactions on Neural
          Networks 14(6), 1569-1572, 2003. https://doi.org/10.1109/TNN.2003.820440

   .. seealso::

      :py:obj:`AdaptiveIzhikevichSoma`
          This model with the adaptation mechanisms.


   .. py:attribute:: config
      :type:  IzhikevichSomaConfig


   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



.. py:class:: IzhikevichSomaConfig

   Bases: :py:obj:`spark.nn.components.somas.base.SomaConfig`


   Configuration for `IzhikevichSoma`.

   :param potential_rest: Membrane rest potential, in mV. The membrane and the recovery variable start here.
   :type potential_rest: float or jax.Array or Initializer, default -65.0
   :param potential_reset: Membrane potential after a spike, in mV. Parameter ``c`` of the Izhikevich model.
   :type potential_reset: float or jax.Array or Initializer, default -65.0
   :param resistance: Membrane resistance, in GΩ.
   :type resistance: float or jax.Array or Initializer, default 0.1
   :param threshold: Peak potential at which a spike is registered, in mV.
   :type threshold: float or jax.Array or Initializer, default 30.0
   :param recovery_timescale: Time scale of the recovery variable. Parameter ``a`` of the Izhikevich model.
   :type recovery_timescale: float or jax.Array or Initializer, default 0.02
   :param recovery_sensitivity: Sensitivity of the recovery variable to subthreshold fluctuations of the membrane
                                potential. Parameter ``b`` of the Izhikevich model.
   :type recovery_sensitivity: float or jax.Array or Initializer, default 0.2
   :param recovery_update: Recovery increment per spike. Parameter ``d`` of the Izhikevich model.
   :type recovery_update: float or jax.Array or Initializer, default 2


   .. py:attribute:: potential_rest
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: potential_reset
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: resistance
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: threshold
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: recovery_timescale
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: recovery_sensitivity
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: recovery_update
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: AdaptiveIzhikevichSoma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.adaptive.AdaptiveSoma`, :py:obj:`IzhikevichSoma`


   Izhikevich soma with the adaptation mechanisms.

   `IzhikevichSoma` composed with `AdaptiveSoma`, which adds an absolute refractory period, a
   potential clamp, an adaptive threshold and an adaptation current. The recovery variable of
   the Izhikevich model is unaffected and keeps its own dynamics.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: AdaptiveIzhikevichSomaConfig, optional

   :Input Ports: * **current** (*CurrentArray*) -- Current delivered to the membrane, in pA.
                 * **inhibition_mask** (*BooleanMask, optional*) -- Marks the inhibitory units. Supplied by the enclosing `Neuron`.

   :Output Ports: **spikes** (*SpikeArray*) -- Non-zero where the potential crossed the threshold on this step.

   :Properties: **potential** (*PotentialArray*) -- Membrane potential, in mV. Read only.

   .. seealso::

      :py:obj:`IzhikevichSoma`
          The membrane integration, without the mechanisms.

      :py:obj:`AdaptiveSoma`
          The mechanisms and their equations.


   .. py:attribute:: config
      :type:  AdaptiveIzhikevichSomaConfig


.. py:class:: AdaptiveIzhikevichSomaConfig

   Bases: :py:obj:`spark.nn.components.somas.adaptive.AdaptiveSomaConfig`, :py:obj:`IzhikevichSomaConfig`


   Configuration for `AdaptiveIzhikevichSoma`.

   Union of `IzhikevichSomaConfig` and `AdaptiveSomaConfig`. It declares no field of its own.


