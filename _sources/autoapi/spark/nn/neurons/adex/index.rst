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

.. py:class:: AdExNeuronConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.NeuronConfig`


   Standard Adaptive Exponential (AdEx) neuron model with traced synapses, neuron-to-neuron delays and Hebbian learning.

           NOTE: Parameter calibration is still necessary.


   .. py:attribute:: modules_specs
      :type:  list[spark.core.specs.ModuleSpecs]


.. py:class:: AdExNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.Neuron`


   Neuron model.

   A neuron is a pipeline object used to represent and coordinate a collection of neurons and interfaces.


   .. py:attribute:: config
      :type:  AdExNeuronConfig


