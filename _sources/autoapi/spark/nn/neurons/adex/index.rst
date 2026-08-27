spark.nn.neurons.adex
=====================

.. py:module:: spark.nn.neurons.adex


Classes
-------

.. autoapisummary::

   spark.nn.neurons.adex.AdExNeuronConfig
   spark.nn.neurons.adex.AdExNeuron


Module Contents
---------------

.. py:class:: AdExNeuronConfig

   Bases: :py:obj:`spark.nn.controllers.NeuronConfig`


   Configuration for `AdExNeuron`.

   :param modules_specs: The four components listed under `AdExNeuron`, prewired. Replace an entry to swap a
                         component, or edit its configuration to retune one.
   :type modules_specs: tuple of ModuleSpecs
   :param units: Shape of the pool of neurons.
   :type units: tuple of int
   :param inhibitory_rate: Fraction of the pool that is inhibitory.
   :type inhibitory_rate: float, default 0.2
   :param seed: Seed for the random draws of the neuron and its modules.
   :type seed: int, optional
   :param dt: Integration step, in ms.
   :type dt: float, default 1.0

   .. rubric:: Notes

   The default parameters of the components have not been calibrated against any particular
   dataset or firing regime.


   .. py:attribute:: modules_specs
      :type:  tuple[spark.core.specs.ModuleSpecs, ...]


.. py:class:: AdExNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.Neuron`


   Adaptive exponential integrate-and-fire neuron with plastic synapses.

   A prewired `Neuron` holding four components:

   * ``delays``, an `N2NDelays` conduction delay, one per connection.
   * ``synapses``, `TracedSynapses`, giving each spike an exponential postsynaptic current.
   * ``soma``, an `AdaptiveExponentialSoma` with an adaptation current rising by 7 pA per
     spike.
   * ``hebbian_rule``, a `HebbianRule` reading the delayed presynaptic spikes, the emitted
     spikes and the current weights.

   The adaptation current subtracts from the input as the unit fires, which is the AdEx
   mechanism behind spike-frequency adaptation and bursting.

   :param config: Controller configuration. Its fields may also be given as keyword arguments.
   :type config: AdExNeuronConfig, optional

   :Input Ports: **in_spikes** (*SpikeArray*) -- Spikes arriving at the pool.

   :Output Ports: **out_spikes** (*SpikeArray*) -- Spikes emitted by the pool on this step.

   :Properties: **inhibition_mask** (*BooleanMask*) -- Marks the inhibitory units of the pool. Read only.

   .. seealso::

      :py:obj:`ALIFNeuron`
          Adaptation through the threshold rather than through a current.

      :py:obj:`LIFNeuron`
          Leaky soma without adaptation.


   .. py:attribute:: config
      :type:  AdExNeuronConfig


