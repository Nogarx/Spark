spark.nn.components.somas.base
==============================

.. py:module:: spark.nn.components.somas.base


Attributes
----------

.. autoapisummary::

   spark.nn.components.somas.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.base.SomaOutput
   spark.nn.components.somas.base.SomaConfig
   spark.nn.components.somas.base.Soma


Module Contents
---------------

.. py:class:: SomaOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a soma model.

   .. attribute:: spikes

      One entry per unit, non-zero where the membrane potential crossed the threshold.

      :type: SpikeArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: spikes
      :type:  spark.core.payloads.SpikeArray


.. py:class:: SomaConfig

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Base configuration for soma models.

   Carries no field of its own. Concrete models declare their own parameters.


.. py:data:: ConfigT

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



