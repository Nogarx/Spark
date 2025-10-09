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


.. py:class:: BrainConfig(**kwargs)

   Bases: :py:obj:`spark.core.config.BaseSparkConfig`


   Configuration class for Brain's.


   .. py:attribute:: input_map
      :type:  dict[str, spark.core.specs.InputSpec]


   .. py:attribute:: output_map
      :type:  dict[str, dict[str, spark.core.specs.OutputSpec]]


   .. py:attribute:: modules_map
      :type:  dict[str, spark.core.specs.ModuleSpecs]


   .. py:method:: validate()

      Validates all fields in the configuration class.



.. py:class:: Brain(config = None, **kwargs)

   Bases: :py:obj:`spark.core.module.SparkModule`


   Abstract brain model.
   This is more a convenience class used to synchronize data more easily.


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



