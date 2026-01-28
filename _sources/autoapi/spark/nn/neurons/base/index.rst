spark.nn.neurons.base
=====================

.. py:module:: spark.nn.neurons.base


Attributes
----------

.. autoapisummary::

   spark.nn.neurons.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.neurons.base.NeuronOutput
   spark.nn.neurons.base.NeuronConfig
   spark.nn.neurons.base.Neuron


Module Contents
---------------

.. py:class:: NeuronOutput

   Bases: :py:obj:`TypedDict`


   Generic Neuron model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: out_spikes
      :type:  spark.core.payloads.SpikeArray


.. py:class:: NeuronConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.core.config.SparkConfig`


   Abstract Neuron model configuration class.


   .. py:attribute:: units
      :type:  tuple[int, Ellipsis]


.. py:data:: ConfigT

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



