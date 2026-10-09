spark.nn.components.somas.izhikevich
====================================

.. py:module:: spark.nn.components.somas.izhikevich


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.izhikevich.IzhikevichSomaConfig
   spark.nn.components.somas.izhikevich.IzhikevichSoma
   spark.nn.components.somas.izhikevich.AdaptiveIzhikevichSomaConfig
   spark.nn.components.somas.izhikevich.AdaptiveIzhikevichSoma


Module Contents
---------------

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



.. py:class:: AdaptiveIzhikevichSomaConfig

   Bases: :py:obj:`spark.nn.components.somas.adaptive.AdaptiveSomaConfig`, :py:obj:`IzhikevichSomaConfig`


   Configuration for `AdaptiveIzhikevichSoma`.

   Union of `IzhikevichSomaConfig` and `AdaptiveSomaConfig`. It declares no field of its own.


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


