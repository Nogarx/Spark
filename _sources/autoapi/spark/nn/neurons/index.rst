spark.nn.neurons
================

.. py:module:: spark.nn.neurons


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/neurons/adex/index
   /autoapi/spark/nn/neurons/alif/index
   /autoapi/spark/nn/neurons/base/index
   /autoapi/spark/nn/neurons/lif/index


Classes
-------

.. autoapisummary::

   spark.nn.neurons.Neuron
   spark.nn.neurons.NeuronConfig
   spark.nn.neurons.NeuronOutput
   spark.nn.neurons.ALIFNeuron
   spark.nn.neurons.ALIFNeuronConfig
   spark.nn.neurons.AdExNeuron
   spark.nn.neurons.AdExNeuronConfig


Package Contents
----------------

.. py:class:: Neuron(config = None, **kwargs)

   Bases: :py:obj:`spark.core.module.SparkModule`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract Neuron model.

   This is a convenience class used to synchronize data more easily.
   Can be thought as the equivalent of Sequential in standard ML frameworks.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:attribute:: units


   .. py:method:: reset()

      Resets neuron states to their initial values.



   .. py:method:: __call__(in_spikes)
      :abstractmethod:


      Execution method.



.. py:class:: NeuronConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.core.config.SparkConfig`


   Abstract Neuron model configuration class.


   .. py:attribute:: units
      :type:  tuple[int, Ellipsis]


.. py:class:: NeuronOutput

   Bases: :py:obj:`TypedDict`


   Generic Neuron model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: out_spikes
      :type:  spark.core.payloads.SpikeArray


.. py:class:: ALIFNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.neurons.Neuron`


   Leaky integrate and fire neuronal model.


   .. py:attribute:: config
      :type:  ALIFNeuronConfig


   .. py:attribute:: soma
      :type:  spark.nn.components.somas.leaky.AdaptiveLeakySoma


   .. py:attribute:: delays
      :type:  spark.nn.components.delays.base.Delays


   .. py:attribute:: synapses
      :type:  spark.nn.components.synapses.base.Synanpses


   .. py:attribute:: learning_rule
      :type:  spark.nn.components.learning_rules.base.LearningRule


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(in_spikes)

      Update neuron's states and compute spikes.



.. py:class:: ALIFNeuronConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.neurons.NeuronConfig`


   ALIFNeuron configuration class.


   .. py:attribute:: inhibitory_rate
      :type:  float


   .. py:attribute:: soma
      :type:  spark.nn.components.somas.leaky.AdaptiveLeakySomaConfig


   .. py:attribute:: synapses
      :type:  spark.nn.components.synapses.base.SynanpsesConfig


   .. py:attribute:: delays
      :type:  spark.nn.components.delays.base.DelaysConfig | None


   .. py:attribute:: learning_rule
      :type:  spark.nn.components.learning_rules.base.LearningRuleConfig | None


.. py:class:: AdExNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.neurons.Neuron`


   Leaky Integrate and Fire neuronal model.


   .. py:attribute:: config
      :type:  AdExNeuronConfig


   .. py:attribute:: soma
      :type:  spark.nn.components.somas.exponential.AdaptiveExponentialSoma


   .. py:attribute:: delays
      :type:  spark.nn.components.delays.base.Delays


   .. py:attribute:: synapses
      :type:  spark.nn.components.synapses.base.Synanpses


   .. py:attribute:: learning_rule
      :type:  spark.nn.components.learning_rules.base.LearningRule


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(in_spikes)

      Update neuron's states and compute spikes.



.. py:class:: AdExNeuronConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.neurons.NeuronConfig`


   AdExNeuron configuration class.


   .. py:attribute:: inhibitory_rate
      :type:  float


   .. py:attribute:: soma
      :type:  spark.nn.components.somas.exponential.AdaptiveExponentialSomaConfig


   .. py:attribute:: synapses
      :type:  spark.nn.components.synapses.base.SynanpsesConfig


   .. py:attribute:: delays
      :type:  spark.nn.components.delays.base.DelaysConfig | None


   .. py:attribute:: learning_rule
      :type:  spark.nn.components.learning_rules.base.LearningRuleConfig | None


