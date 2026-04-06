spark.nn.controllers.brain
==========================

.. py:module:: spark.nn.controllers.brain


Classes
-------

.. autoapisummary::

   spark.nn.controllers.brain.BrainMeta
   spark.nn.controllers.brain.BrainConfig
   spark.nn.controllers.brain.Brain


Module Contents
---------------

.. py:class:: BrainMeta

   Bases: :py:obj:`spark.nn.controllers.base.ControllerMeta`


   Brain metaclass.


.. py:class:: BrainConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.base.ControllerConfig`


   Configuration class for Brain's.


.. py:class:: Brain(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.base.Controller`


   Brain model.

   A brain is a pipeline object used to represent and coordinate a collection of neurons and interfaces.
   This implementation relies on a cache system to simplify parallel computations; every timestep all the modules
   in the Brain read from the cache, update its internal state and update the cache state.
   Note that this introduces a small latency between elements of the brain, which for most cases is negligible, and for
   such a reason it is recommended that only full neuron models and interfaces are used within a Brain.


   .. py:attribute:: config
      :type:  BrainConfig


   .. py:method:: build(input_specs)


   .. py:method:: __call__(**inputs)

      Update brain's states.



   .. py:method:: read_state(port_list)

      Returns the current state of the modules/cache.



