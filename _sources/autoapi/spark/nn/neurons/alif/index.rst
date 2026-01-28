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



