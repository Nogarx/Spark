spark.nn.controllers.neuron
===========================

.. py:module:: spark.nn.controllers.neuron


Classes
-------

.. autoapisummary::

   spark.nn.controllers.neuron.NeuronMeta
   spark.nn.controllers.neuron.NeuronConfig
   spark.nn.controllers.neuron.Neuron


Module Contents
---------------

.. py:class:: NeuronMeta

   Bases: :py:obj:`spark.nn.controllers.base.ControllerMeta`


   Neuron metaclass.


.. py:class:: NeuronConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.base.ControllerConfig`


   Configuration class for Neuron's.


   .. py:attribute:: units
      :type:  tuple[int, Ellipsis]


   .. py:attribute:: inhibitory_rate
      :type:  float


   .. py:method:: __post_init__()


.. py:class:: Neuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.base.Controller`


   Neuron model.

   A neuron is a pipeline object used to represent and coordinate a collection of neurons and interfaces.


   .. py:attribute:: config
      :type:  NeuronConfig


   .. py:attribute:: units


   .. py:method:: recurrent_contract()

      Returns the expected specs for the outputs and properties of the module.

      This function is a binding contract that allows the modules to accept self connections.



   .. py:method:: has_recurrent_contract()
      :classmethod:


      Returns True if the modules defines a recurrent contract, False otherwise.



   .. py:method:: inhibition_mask()


   .. py:method:: build(input_specs)


   .. py:method:: __call__(**inputs)

      Update neuron's states.



   .. py:method:: read_state(port_list)

      Returns the current state of the modules/cache.



