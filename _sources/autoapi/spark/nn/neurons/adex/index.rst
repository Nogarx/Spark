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



