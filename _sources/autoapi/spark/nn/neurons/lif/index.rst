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

.. py:class:: LIFNeuronConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.neurons.NeuronConfig`


   LIFNeuron configuration class.


   .. py:attribute:: inhibitory_rate
      :type:  float


   .. py:attribute:: soma
      :type:  spark.nn.components.somas.leaky.LeakySomaConfig


   .. py:attribute:: synapses
      :type:  spark.nn.components.synapses.base.SynanpsesConfig


   .. py:attribute:: delays
      :type:  spark.nn.components.delays.base.DelaysConfig | None


   .. py:attribute:: learning_rule
      :type:  spark.nn.components.learning_rules.base.LearningRuleConfig | None


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


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(in_spikes)

      Update neuron's states and compute spikes.



