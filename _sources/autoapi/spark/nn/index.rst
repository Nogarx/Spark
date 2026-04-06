spark.nn
========

.. py:module:: spark.nn


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/index
   /autoapi/spark/nn/controllers/index
   /autoapi/spark/nn/initializers/index
   /autoapi/spark/nn/interfaces/index
   /autoapi/spark/nn/neurons/index


Classes
-------

.. autoapisummary::

   spark.nn.Module
   spark.nn.DefaultConfig
   spark.nn.Config
   spark.nn.Brain
   spark.nn.BrainConfig
   spark.nn.Neuron
   spark.nn.NeuronConfig


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



   .. py:attribute:: rng


   .. py:attribute:: __built__
      :type:  bool
      :value: False



   .. py:attribute:: __allow_cycles__
      :type:  bool
      :value: False



   .. py:method:: get_config_spec()
      :classmethod:


      Returns the default configuration class associated with this module.



   .. py:method:: recurrent_contract()

      Returns the expected specs for the outputs and properties of the module.

      This function is a binding contract that allows the modules to accept self connections.



   .. py:method:: has_recurrent_contract()
      :classmethod:


      Returns True if the modules defines a recurrent contract, False otherwise.



   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: set_recurrent_contract(output_contract_specs, property_contract_specs)

      Recurrent shape policy pre-defines expected shapes for the output/property specs.

      This is function is a binding contract that allows the modules to accept self connections.

      Input:
          output_contract_specs: dict[str, PortSpecs], A dictionary with a contract for the output specs.
          property_contract_specs: dict[str, PortSpecs], A dictionary with a contract for the property specs.



   .. py:method:: get_contract_specs()

      Retrieve the recurrent spec policy of the module.



   .. py:method:: get_input_specs()

      Returns a dictionary of the SparkModule's input port specifications.



   .. py:method:: get_output_specs()

      Returns a dictionary of the SparkModule's input port specifications.



   .. py:method:: get_property_specs()

      Returns a dictionary of the SparkModule's property port specifications.



   .. py:method:: get_rng_keys(num_keys)

      Generates a new collection of random keys for the JAX's random engine.



   .. py:method:: get_properties()
      :classmethod:


      Returns all the attributes names wrapped by the spark_property wrapper.



   .. py:method:: __call__(**kwargs)
      :abstractmethod:


      Execution method.



   .. py:method:: __repr__()


   .. py:method:: inspect()

      Returns a formated string of the datastructure.



   .. py:method:: checkpoint(path, overwrite=False)


   .. py:method:: from_checkpoint(path, safe=True)
      :classmethod:



.. py:class:: DefaultConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`SparkConfig`


   Default class for module configuration.


   .. py:attribute:: seed
      :type:  int


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


   .. py:attribute:: dt
      :type:  float


.. py:class:: Config(skip_validation = False, **kwargs)

   Bases: :py:obj:`abc.ABC`


   Base class for module configuration.


   .. py:attribute:: __config_delimiter__
      :type:  str
      :value: '__'



   .. py:attribute:: __shared_config_delimiter__
      :type:  str
      :value: '_s_'



   .. py:attribute:: __metadata__
      :type:  dict


   .. py:attribute:: __graph_editor_metadata__
      :type:  dict


   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:



   .. py:method:: __eq__(other)


   .. py:method:: merge(partial = None, skip_validation = False)

      Update config with partial overrides.



   .. py:method:: diff(other)

      Return differences from another config.



   .. py:method:: validate(is_partial = False, errors = None, current_path = None)

      Validates all fields in the configuration class.



   .. py:method:: get_field_errors(field_name)

      Validates all fields in the configuration class.



   .. py:method:: get_metadata()

      Returns all the metadata in the configuration class, indexed by the attribute name.



   .. py:property:: class_ref
      :type: type


      Returns the type of the associated Module/Initializer.

      NOTE: It is recommended to set the __class_ref__ to the name of the associated module/initializer
      when defining custom configuration classes. The automatic class_ref solver is extremely brittle and
      likely to fail in many different custom scenarios.


   .. py:method:: __post_init__()


   .. py:method:: to_dict(is_partial = False)

      Serialize config to dictionary



   .. py:method:: get_kwargs()

      Returns a dictionary with pairs of key, value fields (skips metadata).



   .. py:method:: from_dict(dct)
      :classmethod:


      Create config instance from dictionary.



   .. py:method:: to_file(file_path, is_partial = False, compress = True)

      Export a config instance from a .scfg file.



   .. py:method:: from_file(file_path, is_partial = False)
      :classmethod:


      Create config instance from a .scfg file.



   .. py:method:: __iter__()

      Custom iterator to simplify SparkConfig inspection across the entire ecosystem.
      This iterator excludes private fields.

      Output:
          field_name: str, field name
          field_value: tp.Any, field value



   .. py:method:: __repr__()


   .. py:method:: inspect(simplified=False)

      Returns a formated string of the datastructure.



   .. py:method:: with_new_seeds(seed = None)

      Utility method to recompute all seed variables within the SparkConfig.
      Useful when creating several populations from the same config.



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



.. py:class:: BrainConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.base.ControllerConfig`


   Configuration class for Brain's.


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



.. py:class:: NeuronConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.base.ControllerConfig`


   Configuration class for Neuron's.


   .. py:attribute:: units
      :type:  tuple[int, Ellipsis]


   .. py:attribute:: inhibitory_rate
      :type:  float


   .. py:method:: __post_init__()


