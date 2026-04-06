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

.. py:class:: LIFNeuronConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.NeuronConfig`


   Standard Leaky-and-Integrate (LIF) neuron model with linear synapses, neuron-to-neuron delays and Hebbian learning.

           NOTE: Parameter calibration is still necessary.


   .. py:attribute:: modules_specs
      :type:  list[spark.core.specs.ModuleSpecs]


.. py:class:: LIFNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.Neuron`


   Neuron model.

   A neuron is a pipeline object used to represent and coordinate a collection of neurons and interfaces.


   .. py:attribute:: config
      :type:  LIFNeuronConfig


