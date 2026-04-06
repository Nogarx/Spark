spark.nn.neurons.module
=======================

.. py:module:: spark.nn.neurons.module


Attributes
----------

.. autoapisummary::

   spark.nn.neurons.module.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.neurons.module.NeuronOutput
   spark.nn.neurons.module.NeuronModuleConfig
   spark.nn.neurons.module.NeuronModule


Module Contents
---------------

.. py:class:: NeuronOutput

   Bases: :py:obj:`TypedDict`


   Generic Neuron model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: out_spikes
      :type:  spark.core.payloads.SpikeArray


.. py:class:: NeuronModuleConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.core.config.DefaultSparkConfig`


   Abstract Neuron model configuration class.


   .. py:attribute:: units
      :type:  tuple[int, Ellipsis]


   .. py:method:: __post_init__()


.. py:data:: ConfigT

.. py:class:: NeuronModule(config = None, **kwargs)

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



