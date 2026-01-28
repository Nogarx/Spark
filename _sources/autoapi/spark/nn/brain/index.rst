spark.nn.brain
==============

.. py:module:: spark.nn.brain


Classes
-------

.. autoapisummary::

   spark.nn.brain.BrainMeta
   spark.nn.brain.BrainConfig
   spark.nn.brain.Brain


Module Contents
---------------

.. py:class:: BrainMeta

   Bases: :py:obj:`spark.core.module.SparkMeta`


   Brain metaclass.


.. py:class:: BrainConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.core.config.BaseSparkConfig`


   Configuration class for Brain's.


   .. py:attribute:: input_map
      :type:  dict[str, spark.core.specs.PortSpecs]


   .. py:attribute:: output_map
      :type:  dict[str, dict]


   .. py:attribute:: modules_map
      :type:  dict[str, spark.core.specs.ModuleSpecs]


   .. py:method:: validate(is_partial = False, errors = None, current_path = ['brain'])

      Validates all fields in the configuration class.



   .. py:method:: refresh_seeds()

      Utility method to recompute all seed variables within the SparkConfig.
      Useful when creating several populations from the same config.



.. py:class:: Brain(config = None, **kwargs)

   Bases: :py:obj:`spark.core.module.SparkModule`


   Brain model.

   A brain is a pipeline object used to represent and coordinate a collection of neurons and interfaces.
   This implementation relies on a cache system to simplify parallel computations; every timestep all the modules
   in the Brain read from the cache, update its internal state and update the cache state.
   Note that this introduces a small latency between elements of the brain, which for most cases is negligible, and for
   such a reason it is recommended that only full neuron models and interfaces are used within a Brain.


   .. py:attribute:: config
      :type:  BrainConfig


   .. py:method:: resolve_initialization_order()

      Resolves the initialization order of the modules.



   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets all the modules to its initial state.



   .. py:method:: __call__(**inputs)

      Update brain's states.



   .. py:method:: get_spikes_from_cache()

      Collect the brain's spikes.



