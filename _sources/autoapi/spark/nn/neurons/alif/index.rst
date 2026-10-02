spark.nn.neurons.alif
=====================

.. py:module:: spark.nn.neurons.alif


Classes
-------

.. autoapisummary::

   spark.nn.neurons.alif.ALIFNeuronConfig
   spark.nn.neurons.alif.ALIFNeuron


Module Contents
---------------

.. py:class:: ALIFNeuronConfig

   Bases: :py:obj:`spark.nn.controllers.NeuronConfig`


   Configuration for `ALIFNeuron`.

   :param modules_specs: The four components listed under `ALIFNeuron`, prewired. Replace an entry to swap a
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


.. py:class:: ALIFNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.Neuron`


   Adaptive leaky integrate-and-fire neuron with plastic synapses.

   A prewired `Neuron` holding four components:

   * ``delays``, an `N2NDelays` conduction delay, one per connection.
   * ``synapses``, `TracedSynapses`, giving each spike an exponential postsynaptic current.
   * ``soma``, an `AdaptiveLeakySoma` with a 3 ms refractory period and a threshold that
     rises by 100 mV per spike and decays back.
   * ``hebbian_rule``, a `HebbianRule` reading the delayed presynaptic spikes, the emitted
     spikes and the current weights.

   The rising threshold makes the unit progressively harder to drive as it fires, so a
   constant input produces a rate that falls rather than one that holds.

   :param config: Controller configuration. Its fields may also be given as keyword arguments.
   :type config: ALIFNeuronConfig, optional

   :Input Ports: **in_spikes** (*SpikeArray*) -- Spikes arriving at the pool.

   :Output Ports: **out_spikes** (*SpikeArray*) -- Spikes emitted by the pool on this step.

   :Properties: **inhibition_mask** (*BooleanMask*) -- Marks the inhibitory units of the pool. Read only.

   .. seealso::

      :py:obj:`LIFNeuron`
          The same neuron without threshold adaptation.

      :py:obj:`AdExNeuron`
          Adaptation through a current rather than through the threshold.


   .. py:attribute:: config
      :type:  ALIFNeuronConfig


