spark.nn.components.somas.exponential
=====================================

.. py:module:: spark.nn.components.somas.exponential


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.exponential.ExponentialSomaConfig
   spark.nn.components.somas.exponential.ExponentialSoma
   spark.nn.components.somas.exponential.AdaptiveExponentialSomaConfig
   spark.nn.components.somas.exponential.AdaptiveExponentialSoma


Module Contents
---------------

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



.. py:class:: AdaptiveExponentialSomaConfig

   Bases: :py:obj:`spark.nn.components.somas.adaptive.AdaptiveSomaConfig`, :py:obj:`ExponentialSomaConfig`


   Configuration for `AdaptiveExponentialSoma`.

   Union of `ExponentialSomaConfig` and `AdaptiveSomaConfig`. It declares no field of its own.


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


