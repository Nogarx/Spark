spark.nn.controllers
====================

.. py:module:: spark.nn.controllers


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/controllers/base/index
   /autoapi/spark/nn/controllers/brain/index
   /autoapi/spark/nn/controllers/neuron/index


Classes
-------

.. autoapisummary::

   spark.nn.controllers.Controller
   spark.nn.controllers.ControllerConfig
   spark.nn.controllers.Brain
   spark.nn.controllers.BrainConfig
   spark.nn.controllers.Neuron
   spark.nn.controllers.NeuronConfig


Package Contents
----------------

.. py:class:: Controller(config = None, **kwargs)

   Bases: :py:obj:`flax.nnx.Module`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Controller model.

   A controller is a pipeline object used to represent and coordinate a collection of Spark modules.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:attribute:: default_config
      :type:  type[ConfigT]


   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:



   .. py:method:: get_properties()
      :classmethod:


      Returns all the attributes names wrapped by the spark_property wrapper.



   .. py:method:: recurrent_contract()

      Returns the expected specs for the outputs and properties of the module.

      This function is a binding contract that allows the modules to accept self connections.



   .. py:method:: has_recurrent_contract()
      :classmethod:


      Returns True if the modules defines a recurrent contract, False otherwise.



   .. py:method:: get_config_spec()
      :classmethod:


      Returns the default configuration class associated with this module.



   .. py:method:: build(input_specs)


   .. py:method:: get_controller_inputs()

      Returns the names of the controller's input variables



   .. py:method:: get_controller_outputs()

      Returns the names of the controller's output variables



   .. py:method:: refresh_seeds(seed = None)

      Utility method to recompute all seed variables within the SparkConfig.
      Useful when creating several populations from the same config.

      NOTE: This method has no effect after the model has been built.



   .. py:method:: reset()

      Resets all the modules to its initial state.



   .. py:method:: __call__(**inputs)
      :abstractmethod:


      Update controller's states.



   .. py:method:: read_state(port_list)
      :abstractmethod:


      Utility function to read internal controller's variables.



   .. py:method:: get_rng_keys(num_keys)

      Generates a new collection of random keys for the JAX's random engine.



.. py:class:: ControllerConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.core.config.SparkConfig`


   Base class for module configuration.


   .. py:attribute:: modules_specs
      :type:  list[spark.core.specs.ModuleSpecs]


   .. py:attribute:: seed
      :type:  int


   .. py:attribute:: dt
      :type:  float


   .. py:method:: __post_init__()


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


