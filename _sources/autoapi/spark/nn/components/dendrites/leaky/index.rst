spark.nn.components.dendrites.leaky
===================================

.. py:module:: spark.nn.components.dendrites.leaky


Classes
-------

.. autoapisummary::

   spark.nn.components.dendrites.leaky.LeakyDendriteConfig
   spark.nn.components.dendrites.leaky.LeakyDendrite


Module Contents
---------------

.. py:class:: LeakyDendriteConfig

   Bases: :py:obj:`spark.nn.components.dendrites.base.DendriteConfig`


   Configuration for `LeakyDendrite`.

   :param capacitance: Membrane capacitance, in pF.
   :type capacitance: float or jax.Array or Initializer, default 23.674
   :param leak_conductance: Leak conductance, in nS.
   :type leak_conductance: float or jax.Array or Initializer, default 3.378
   :param The remaining parameters are those of `DendriteConfig`.:

   .. rubric:: Notes

   The membrane is described by a capacitance and a leak conductance rather than by a time
   constant and a resistance, as the somas of the package are, because every other term of
   a dendrite is a conductance as well: the coupling, the back-propagating action potential
   and the active channels of the subclasses. The time constant of the isolated membrane
   is ``capacitance / leak_conductance``.


   .. py:attribute:: capacitance
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: leak_conductance
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: LeakyDendrite(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.dendrites.base.Dendrite`


   Passive dendritic compartment.

   A leaky membrane coupled to the soma through an axial conductance and depolarized by the
   back-propagating action potential. It has no active current of its own: it filters its
   input and passes the somatic spike back into the dendritic tree.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: LeakyDendriteConfig, optional

   :Input Ports: * **in_current** (*CurrentArray*) -- Current delivered to the dendrite, in pA.
                 * **soma_potential** (*PotentialArray, optional*) -- Potential of the coupled soma, relative to the soma's rest.
                 * **spikes** (*SpikeArray, optional*) -- Somatic spikes, driving the back-propagating action potential.

   :Output Ports: * **out_current** (*CurrentArray*) -- Axial current delivered to the soma, in pA.
                  * **plateau** (*FloatArray*) -- How far the dendrite is into a plateau, in ``[0, 1]``.

   :Properties: **potential** (*PotentialArray*) -- Dendritic membrane potential, relative to ``potential_rest``. Read only.

   .. rubric:: Notes

   Every term of the membrane equation is a conductance times a driving force, so the
   step is integrated exactly for conductances held over the step. With
   :math:`G = g_L + g_C + g_{BAP} + \sum_i g_i` the total conductance and
   :math:`J = I + g_C V_s + g_{BAP} E_{BAP} + \sum_i g_i E_i` the reversal weighted drive,
   both in the frame where the leak reversal is zero,

   .. math::
       V_\infty = J / G, \qquad
       V_{t+1} = V_\infty + (V_t - V_\infty) \exp(-\Delta t \, G / C)

   The step is applied as :math:`V_t + (1 - e^{-\Delta t G / C}) (V_\infty - V_t)`, with the
   gain computed at single precision, so it stays accurate at half precision and small
   steps. The channel sums come from `_channel_terms`, which is empty here and filled in by
   active models such as `CalciumDendrite`. This exponential Euler step stays stable at any
   step size, which matters because the coupling makes the dendrite much faster than its
   own leak: with the default parameters the isolated time constant is 7 ms, the coupled
   one 1 ms.

   .. seealso::

      :py:obj:`CalciumDendrite`
          This membrane with the calcium and calcium activated potassium currents.


   .. py:attribute:: config
      :type:  LeakyDendriteConfig


   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



