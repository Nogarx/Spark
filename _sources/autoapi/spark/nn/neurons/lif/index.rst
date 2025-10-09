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

.. py:class:: LIFNeuronConfig(**kwargs)

   Bases: :py:obj:`spark.nn.neurons.NeuronConfig`


   LIFNeuron configuration class.


   .. py:attribute:: max_delay
      :type:  float


   .. py:attribute:: inhibitory_rate
      :type:  float


   .. py:attribute:: async_spikes
      :type:  bool


   .. py:attribute:: soma_config
      :type:  spark.nn.components.somas.leaky.LeakySomaConfig


   .. py:attribute:: synapses_config
      :type:  spark.nn.components.synapses.base.SynanpsesConfig


   .. py:attribute:: delays_config
      :type:  spark.nn.components.delays.base.DelaysConfig


   .. py:attribute:: learning_rule_config
      :type:  spark.nn.components.learning_rules.base.LearningRuleConfig


.. py:class:: LIFNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.neurons.Neuron`


   Leaky Integrate and Fire neuronal model.


   .. py:attribute:: config
      :type:  LIFNeuronConfig


   .. py:attribute:: soma
      :type:  spark.nn.components.somas.leaky.LeakySoma


   .. py:attribute:: delays
      :type:  spark.nn.components.delays.base.Delays


   .. py:attribute:: synapses
      :type:  spark.nn.components.synapses.base.Synanpses


   .. py:attribute:: learning_rule
      :type:  spark.nn.components.learning_rules.base.LearningRule


   .. py:attribute:: max_delay


   .. py:attribute:: inhibitory_rate


   .. py:attribute:: async_spikes


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(in_spikes)

      Update neuron's states and compute spikes.



