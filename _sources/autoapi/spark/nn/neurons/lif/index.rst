spark.nn.neurons.lif
====================

.. py:module:: spark.nn.neurons.lif


Classes
-------

.. autoapisummary::

   spark.nn.neurons.lif.LIFNeuronConfig
   spark.nn.neurons.lif.LIFNeuron


Module Contents
---------------

.. py:class:: LIFNeuronConfig

   Bases: :py:obj:`spark.nn.controllers.NeuronConfig`


   Configuration for `LIFNeuron`.

   :param modules_specs: The four components listed under `LIFNeuron`, prewired. Replace an entry to swap a
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


.. py:class:: LIFNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.Neuron`


   Leaky integrate-and-fire neuron with plastic synapses.

   A prewired `Neuron` holding four components:

   * ``delays``, an `N2NDelays` conduction delay, one per connection.
   * ``synapses``, `LinearSynapses`, whose weights the plasticity rule writes back.
   * ``soma``, an `AdaptiveLeakySoma` with a 3 ms refractory period.
   * ``hebbian_rule``, a `HebbianRule` reading the delayed presynaptic spikes, the emitted
     spikes and the current weights.

   :param config: Controller configuration. Its fields may also be given as keyword arguments.
   :type config: LIFNeuronConfig, optional

   :Input Ports: **in_spikes** (*SpikeArray*) -- Spikes arriving at the pool.

   :Output Ports: **out_spikes** (*SpikeArray*) -- Spikes emitted by the pool on this step.

   :Properties: **inhibition_mask** (*BooleanMask*) -- Marks the inhibitory units of the pool. Read only.

   .. rubric:: Notes

   Only the refractory period is enabled on the soma. Threshold adaptation and the adaptation
   current are available by setting their trigger parameters on the soma configuration.

   .. seealso::

      :py:obj:`ALIFNeuron`
          The same neuron with threshold adaptation.

      :py:obj:`AdExNeuron`
          Adaptive exponential soma in place of the leaky one.


   .. py:attribute:: config
      :type:  LIFNeuronConfig


