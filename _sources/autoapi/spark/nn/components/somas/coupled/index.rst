spark.nn.components.somas.coupled
=================================

.. py:module:: spark.nn.components.somas.coupled


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.coupled.CoupledSomaConfig
   spark.nn.components.somas.coupled.CoupledSoma


Module Contents
---------------

.. py:class:: CoupledSomaConfig

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Configuration for `CoupledSoma`.

   Carries no field of its own. The coupling current is state written by another module,
   not a parameter.


.. py:class:: CoupledSoma(config = None, **kwargs)

   Coupling of a soma to a dendritic compartment.

   Mixin adding a writable ``coupling_current`` property to a `Soma` subclass, the current
   another compartment injects into the membrane. It must precede the model in the base
   list::

       class CoupledLeakySoma(CoupledSoma, LeakySoma): ...

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: CoupledSomaConfig, optional

   :Input Ports: * **current** (*CurrentArray*) -- Current delivered to the membrane, in pA. The coupling current is added to it before
                   the model sees it.
                 * **inhibition_mask** (*BooleanMask, optional*) -- Marks the inhibitory units. Passed through to the model unchanged.

   :Output Ports: **spikes** (*SpikeArray*) -- Non-zero where the potential crossed the threshold.

   :Properties: * **potential** (*PotentialArray*) -- Membrane potential, as held by the model this mixin extends. Read only.
                * **coupling_current** (*CurrentArray*) -- Current injected by the coupled compartment, in pA. Writable, so a `Neuron` can set
                  it as an effect from the ``out_current`` port of a dendrite.

   .. rubric:: Notes

   The mixin supplies one term of the soma step, the current offset:

   .. math::
       I_{\mathrm{eff}} = \Phi_{\mathrm{model}}(I + I_C)

   where :math:`I_C = g_C (V_d - V_s)` is what the dendrite reports. The current is added
   before the mechanisms of the model act on the input, so a refractory gate of
   `AdaptiveSoma` silences the dendritic drive along with the synaptic one, as the reset
   clamp of the reference model does.

   Within a `Neuron` the dendrite reads the ``potential`` property and the ``spikes``
   output of the soma, and the soma receives the dendritic current as an effect at the end
   of the step. The soma therefore integrates the current the dendrite computed on the
   previous step, one step of lag that keeps the wiring free of cycles and the execution
   order fixed. The dendrite sees the soma of the same step.

   .. seealso::

      :py:obj:`CoupledLeakySoma`
          Leaky membrane with the coupling.

      :py:obj:`CoupledAdaptiveExponentialSoma`
          AdEx membrane with the coupling, the soma of the Ca-AdEx neuron.

      :py:obj:`Dendrite`
          The compartment producing the coupling current.


   .. py:attribute:: config
      :type:  CoupledSomaConfig


   .. py:method:: build(**abc_args)


   .. py:method:: coupling_current()


   .. py:method:: reset()

      Resets component state.



