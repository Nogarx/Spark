spark.nn.neurons
================

.. py:module:: spark.nn.neurons


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/neurons/adex/index
   /autoapi/spark/nn/neurons/alif/index
   /autoapi/spark/nn/neurons/lif/index
   /autoapi/spark/nn/neurons/module/index


Classes
-------

.. autoapisummary::

   spark.nn.neurons.LIFNeuron
   spark.nn.neurons.LIFNeuronConfig
   spark.nn.neurons.ALIFNeuron
   spark.nn.neurons.ALIFNeuronConfig
   spark.nn.neurons.AdExNeuron


Package Contents
----------------

.. py:class:: LIFNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.Neuron`


   Neuron model.

   A neuron is a pipeline object used to represent and coordinate a collection of neurons and interfaces.


   .. py:attribute:: config
      :type:  LIFNeuronConfig


.. py:class:: LIFNeuronConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.NeuronConfig`


   Standard Leaky-and-Integrate (LIF) neuron model with linear synapses, neuron-to-neuron delays and Hebbian learning.

           NOTE: Parameter calibration is still necessary.


   .. py:attribute:: modules_specs
      :type:  list[spark.core.specs.ModuleSpecs]


.. py:class:: ALIFNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.Neuron`


   Neuron model.

   A neuron is a pipeline object used to represent and coordinate a collection of neurons and interfaces.


   .. py:attribute:: config
      :type:  ALIFNeuronConfig


.. py:class:: ALIFNeuronConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.NeuronConfig`


   Standard Adaptive Leaky-and-Integrate (ALIF) neuron model with traced synapses, neuron-to-neuron delays and Hebbian learning.

           NOTE: Parameter calibration is still necessary.


   .. py:attribute:: modules_specs
      :type:  list[spark.core.specs.ModuleSpecs]


.. py:class:: AdExNeuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.Neuron`


   Neuron model.

   A neuron is a pipeline object used to represent and coordinate a collection of neurons and interfaces.


   .. py:attribute:: config
      :type:  AdExNeuronConfig


