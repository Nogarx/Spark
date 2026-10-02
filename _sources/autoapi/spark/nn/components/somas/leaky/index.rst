spark.nn.components.somas.leaky
===============================

.. py:module:: spark.nn.components.somas.leaky


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.leaky.LeakySomaConfig
   spark.nn.components.somas.leaky.LeakySoma
   spark.nn.components.somas.leaky.AdaptiveLeakySomaConfig
   spark.nn.components.somas.leaky.AdaptiveLeakySoma


Module Contents
---------------

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



.. py:class:: AdaptiveLeakySomaConfig

   Bases: :py:obj:`spark.nn.components.somas.adaptive.AdaptiveSomaConfig`, :py:obj:`LeakySomaConfig`


   Configuration for `AdaptiveLeakySoma`.

   Union of `LeakySomaConfig` and `AdaptiveSomaConfig`. It declares no field of its own.


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


