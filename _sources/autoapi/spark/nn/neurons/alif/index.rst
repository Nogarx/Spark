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

.. py:class:: ALIFNeuronConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.NeuronConfig`


   Standard Adaptive Leaky-and-Integrate (ALIF) neuron model with traced synapses, neuron-to-neuron delays and Hebbian learning.

           NOTE: Parameter calibration is still necessary.


   .. py:attribute:: modules_specs
      :type:  list[spark.core.specs.ModuleSpecs]


.. py:class:: ALIFNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.Neuron`


   Neuron model.

   A neuron is a pipeline object used to represent and coordinate a collection of neurons and interfaces.


   .. py:attribute:: config
      :type:  ALIFNeuronConfig


