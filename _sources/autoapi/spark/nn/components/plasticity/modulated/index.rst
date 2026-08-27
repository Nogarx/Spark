spark.nn.components.plasticity.modulated
========================================

.. py:module:: spark.nn.components.plasticity.modulated


Classes
-------

.. autoapisummary::

   spark.nn.components.plasticity.modulated.ModulatedPlasticityConfig
   spark.nn.components.plasticity.modulated.ModulatedPlasticity


Module Contents
---------------

.. py:class:: ModulatedPlasticityConfig

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   Configuration for `ModulatedPlasticity`.

   Carries no field of its own. The third factor arrives on a port rather than through the
   configuration.


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



