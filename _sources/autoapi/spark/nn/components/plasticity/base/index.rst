spark.nn.components.plasticity.base
===================================

.. py:module:: spark.nn.components.plasticity.base


Attributes
----------

.. autoapisummary::

   spark.nn.components.plasticity.base.PlasticityParamLike
   spark.nn.components.plasticity.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.components.plasticity.base.PlasticityOutput
   spark.nn.components.plasticity.base.PlasticityConfig
   spark.nn.components.plasticity.base.Plasticity


Module Contents
---------------

.. py:data:: PlasticityParamLike

.. py:class:: PlasticityOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a plasticity rule.

   .. attribute:: kernel

      The updated synaptic weights, to be written back onto the synapse.

      :type: FloatArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: kernel
      :type:  spark.core.payloads.FloatArray


.. py:class:: PlasticityConfig

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Base configuration for plasticity rules.

   Carries no field of its own. Concrete rules declare their own parameters.


.. py:data:: ConfigT

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



