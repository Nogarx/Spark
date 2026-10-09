spark.nn.components.dendrites
=============================

.. py:module:: spark.nn.components.dendrites


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/dendrites/base/index
   /autoapi/spark/nn/components/dendrites/leaky/index


Classes
-------

.. autoapisummary::

   spark.nn.components.dendrites.Dendrite
   spark.nn.components.dendrites.DendriteConfig
   spark.nn.components.dendrites.DendriteOutput
   spark.nn.components.dendrites.LeakyDendrite
   spark.nn.components.dendrites.LeakyDendriteConfig


Package Contents
----------------

.. py:class:: Dendrite(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for dendrite models.

   A dendrite owns a membrane potential, exchanges an axial current with the soma it is
   attached to, and receives the somatic spikes as a back-propagating action potential. The
   step is fixed here and a subclass fills in the membrane::

       V_s    = soma_potential + (E_soma - E_dendrite)
       g_BAP  = _bap_conductance(spikes)
       I_eff  = _effective_current(I)
       V'     = _integrate(V, I_eff, V_s, g_BAP)
                _update_state(V, V')
       p      = _plateau_computation(V', _effective_threshold())
       I_out  = coupling * (V' - V_s)

   Only `_integrate` is required. The other hooks return their argument unchanged or do
   nothing, so a model that is a membrane integration and nothing else defines one method.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: DendriteConfig, optional

   :Input Ports: * **in_current** (*CurrentArray*) -- Current delivered to the dendrite, in pA. Synaptic input and any injected current.
                 * **soma_potential** (*PotentialArray, optional*) -- Potential of the soma the dendrite is coupled to, relative to the soma's rest. Wire
                   the ``potential`` property of the soma onto it. Without it the dendrite is an
                   isolated compartment: no axial current flows in either direction.
                 * **spikes** (*SpikeArray, optional*) -- Somatic spikes. Each one triggers a back-propagating action potential. Without it
                   no back-propagation takes place.
                 * **injected_current** (*CurrentArray, optional*) -- Current injected into the dendrite besides the synaptic one, in pA, summed with
                   ``in_current``. Meant for instructive or teaching currents that are not synaptic.

   :Output Ports: * **out_current** (*CurrentArray*) -- Axial current delivered to the soma, in pA. Inside a `Neuron` it is written onto the
                    ``coupling_current`` property of a `CoupledSoma` as an effect, which closes the loop
                    between the two compartments without a cycle in the wiring.
                  * **plateau** (*FloatArray*) -- How far the dendrite is into a plateau, in ``[0, 1]``.

   :Properties: **potential** (*PotentialArray*) -- Dendritic membrane potential, relative to ``potential_rest``. Read only.

   .. rubric:: Notes

   The back-propagating action potential follows the reference implementation of the
   Ca-AdEx neuron: a conductance with a double exponential time course, normalized so that
   one spike peaks at ``bap_conductance``, driving the membrane towards ``bap_reversal``.

   There is no conduction delay parameter. The spike takes on the order of a millisecond
   to travel from the soma to the calcium hot zone, which is at or below the step sizes
   Spark is typically run at, and the reference fit put its delay at the smallest value
   its simulator allowed, one 0.1 ms step. A somatic spike therefore enters the kernel on
   the step it arrives and, since the kernel starts from zero, its conductance acts from
   the next step on. To model a longer conduction delay, put an `NDelays` between the
   ``spikes`` output of the soma and the ``spikes`` port of the dendrite.

   Potentials are stored relative to ``potential_rest``. The soma potential arrives
   relative to the soma's own rest and is shifted by ``soma_potential_rest -
   potential_rest`` before use, so ``soma_potential_rest`` has to name the rest potential
   of the soma actually wired in.

   The plateau is reported as a smooth reading of the threshold crossing rather than as a
   flag, so it can drive a trace or gate a plasticity rule directly. It reads 0.5 at
   ``threshold``, so comparing it against 0.5 recovers the hard flag.

   .. seealso::

      :py:obj:`LeakyDendrite`
          Passive membrane.

      :py:obj:`CalciumDendrite`
          Calcium hot zone with a regenerative calcium spike.

      :py:obj:`CoupledSoma`
          Soma mechanism receiving ``out_current``.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: build(in_current, soma_potential = None, spikes = None, injected_current = None)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: potential()


   .. py:method:: reset()

      Resets the dendrite state to its initial value.



   .. py:method:: __call__(in_current, soma_potential = None, spikes = None, injected_current = None)

      Advances the dendrite one step.

      :param in_current: Current delivered to the dendrite, in pA.
      :type in_current: CurrentArray
      :param soma_potential: Potential of the coupled soma, relative to the soma's rest.
      :type soma_potential: PotentialArray, optional
      :param spikes: Somatic spikes, driving the back-propagating action potential.
      :type spikes: SpikeArray, optional
      :param injected_current: Additional current injected into the dendrite, in pA.
      :type injected_current: CurrentArray, optional

      :returns: Dictionary with ``out_current`` and ``plateau``.
      :rtype: DendriteOutput



.. py:class:: DendriteConfig

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Base configuration for dendrite models.

   Holds what every dendrite in this package shares: its rest potential, the axial coupling
   to the soma, the back-propagating action potential and the plateau reading. Concrete
   models add the parameters of their membrane.

   :param potential_rest: Dendritic leak reversal potential, in mV. Potentials are stored relative to it.
   :type potential_rest: float or jax.Array or Initializer, default -55.0
   :param coupling: Axial conductance between the soma and the dendrite, in nS. The dendrite receives
                    ``coupling * (V_soma - V_dendrite)`` and delivers the opposite current to the soma.
   :type coupling: float or jax.Array or Initializer, default 19.777
   :param soma_potential_rest: Rest potential of the soma the dendrite is coupled to, in mV. A soma reports its
                               potential relative to its own rest, so this is needed to place both compartments on
                               one scale. ``None`` reads as ``potential_rest``, i.e. no offset.
   :type soma_potential_rest: float or jax.Array or Initializer or None, default None
   :param bap_conductance: Peak conductance reached by one back-propagating action potential, in nS.
   :type bap_conductance: float or jax.Array or Initializer, default 27.996
   :param bap_reversal: Reversal potential of the back-propagating conductance, in mV.
   :type bap_reversal: float or jax.Array or Initializer, default 0.0
   :param bap_rise: Rise constant of the back-propagating conductance, in ms. Must be shorter than
                    ``bap_decay``.
   :type bap_rise: float or jax.Array or Initializer, default 0.2
   :param bap_decay: Decay constant of the back-propagating conductance, in ms.
   :type bap_decay: float or jax.Array or Initializer, default 3.0
   :param threshold: Potential above which the dendrite counts as being in a plateau, in mV.
   :type threshold: float or jax.Array or Initializer, default -25.0
   :param plateau_slope: Slope of the plateau reading, in mV. Smaller values approach a hard threshold.
   :type plateau_slope: float or jax.Array or Initializer, default 2.0

   .. rubric:: Notes

   The defaults are the distal compartment of the Ca-AdEx neuron of Pastorelli et al.
   (2025), Supplementary Table S1. The table lists ``w_BAP`` in mV; the reference
   implementation uses it as the peak of a conductance, in nS, and so does this package.


   .. py:attribute:: potential_rest
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: coupling
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: soma_potential_rest
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer | None


   .. py:attribute:: bap_conductance
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: bap_reversal
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: bap_rise
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: bap_decay
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: threshold
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: plateau_slope
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: DendriteOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a dendrite model.

   .. attribute:: out_current

      One entry per unit, the axial current the dendrite delivers to the soma, in pA.
      Positive when the dendrite is more depolarized than the soma.

      :type: CurrentArray

   .. attribute:: plateau

      One entry per unit, how far the dendrite is into a plateau. Rises from 0 to 1 as
      the membrane crosses ``threshold``, and reads 0.5 exactly at it.

      :type: FloatArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: out_current
      :type:  spark.core.payloads.CurrentArray


   .. py:attribute:: plateau
      :type:  spark.core.payloads.FloatArray


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


