spark.nn
========

.. py:module:: spark.nn


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/brain/index
   /autoapi/spark/nn/components/index
   /autoapi/spark/nn/initializers/index
   /autoapi/spark/nn/interfaces/index
   /autoapi/spark/nn/neurons/index


Classes
-------

.. autoapisummary::

   spark.nn.Module
   spark.nn.Config
   spark.nn.BaseConfig
   spark.nn.Brain
   spark.nn.BrainConfig


Package Contents
----------------

.. py:class:: Module(*, config = None, name = None, **kwargs)

   Bases: :py:obj:`flax.nnx.Module`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ , :py:obj:`InputT`\ ]


   Base class for Spark Modules


   .. py:attribute:: name
      :type:  str
      :value: 'name'



   .. py:attribute:: config
      :type:  ConfigT


   .. py:attribute:: default_config
      :type:  type[ConfigT]


   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:



   .. py:attribute:: __built__
      :type:  bool
      :value: False



   .. py:attribute:: __allow_cycles__
      :type:  bool
      :value: False



   .. py:method:: get_config_spec()
      :classmethod:


      Returns the default configuratio class associated with this module.



   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: set_recurrent_shape_contract(shape = None, output_shapes = None)

      Recurrent shape policy pre-defines expected shapes for the output specs.

      This is function is a binding contract that allows the modules to accept self connections.

      Input:
          shape: Shape, A common shape for all the outputs.
          output_shapes: dict[str, Shape], A specific policy for every single output variable.

      NOTE: If both, shape and output_specs, are provided, output_specs takes preference over shape.



   .. py:method:: get_recurrent_shape_contract()

      Retrieve the recurrent shape policy of the module.



   .. py:method:: get_input_specs()

      Returns a dictionary of the SparkModule's input port specifications.



   .. py:method:: get_output_specs()

      Returns a dictionary of the SparkModule's input port specifications.



   .. py:method:: get_rng_keys(num_keys)

      Generates a new collection of random keys for the JAX's random engine.



   .. py:method:: __call__(**kwargs)
      :abstractmethod:


      Execution method.



.. py:class:: Config(**kwargs)

   Bases: :py:obj:`BaseSparkConfig`


   Default class for module configuration.


   .. py:attribute:: seed
      :type:  int


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


   .. py:attribute:: dt
      :type:  float


.. py:class:: BaseConfig(**kwargs)

   Bases: :py:obj:`abc.ABC`


   Base class for module configuration.


   .. py:attribute:: __config_delimiter__
      :type:  str
      :value: '__'



   .. py:attribute:: __shared_config_delimiter__
      :type:  str
      :value: '_s_'



   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:



   .. py:method:: merge(partial = {})

      Update config with partial overrides.



   .. py:method:: diff(other)

      Return differences from another config.



   .. py:method:: validate()

      Validates all fields in the configuration class.



   .. py:method:: get_metadata()

      Returns all the metadata in the configuration class, indexed by the attribute name.



   .. py:property:: class_ref
      :type: type


      Returns the type of the associated Module/Initializer.

      NOTE: It is recommended to set the __class_ref__ to the name of the associated module/initializer
      when defining custom configuration classes. The current class_ref solver is extremely brittle and
      likely to fail in many different custom scenarios.


   .. py:method:: __post_init__()


   .. py:method:: to_dict()

      Serialize config to dictionary



   .. py:method:: from_dict(dct)
      :classmethod:


      Create config instance from dictionary.



   .. py:method:: to_file(file_path)

      Export a config instance from a .scfg file.



   .. py:method:: from_file(file_path)
      :classmethod:


      Create config instance from a .scfg file.



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



