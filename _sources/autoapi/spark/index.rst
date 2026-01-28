spark
=====

.. py:module:: spark


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/core/index
   /autoapi/spark/graph_editor/index
   /autoapi/spark/nn/index


Attributes
----------

.. autoapisummary::

   spark.register_module
   spark.register_initializer
   spark.register_payload
   spark.register_config
   spark.register_cfg_validator
   spark.REGISTRY


Classes
-------

.. autoapisummary::

   spark.Constant
   spark.Variable
   spark.SparkPayload
   spark.SpikeArray
   spark.CurrentArray
   spark.PotentialArray
   spark.FloatArray
   spark.IntegerArray
   spark.BooleanMask
   spark.PortSpecs
   spark.PortMap
   spark.ModuleSpecs
   spark.GraphEditor


Functions
---------

.. autoapisummary::

   spark.split
   spark.merge


Package Contents
----------------

.. py:class:: Constant(data, dtype = None)

   Jax.Array wrapper for constant arrays.


   .. py:attribute:: value
      :type:  jax.Array


   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:property:: shape
      :type: tuple[int, Ellipsis]



   .. py:property:: dtype
      :type: Any



   .. py:property:: ndim
      :type: int



   .. py:property:: size
      :type: int



   .. py:property:: T
      :type: jax.Array



   .. py:method:: __neg__()


   .. py:method:: __pos__()


   .. py:method:: __abs__()


   .. py:method:: __invert__()


   .. py:method:: __add__(other)


   .. py:method:: __sub__(other)


   .. py:method:: __mul__(other)


   .. py:method:: __truediv__(other)


   .. py:method:: __floordiv__(other)


   .. py:method:: __mod__(other)


   .. py:method:: __matmul__(other)


   .. py:method:: __pow__(other)


   .. py:method:: __radd__(other)


   .. py:method:: __rsub__(other)


   .. py:method:: __rmul__(other)


   .. py:method:: __rtruediv__(other)


   .. py:method:: __rfloordiv__(other)


   .. py:method:: __rmod__(other)


   .. py:method:: __rmatmul__(other)


   .. py:method:: __rpow__(other)


.. py:class:: Variable(value, dtype = None, **metadata)

   Bases: :py:obj:`flax.nnx.Variable`


   The base class for all ``Variable`` types.
   Note that this is just a convinience wrapper around Flax's nnx.Variable to simplify imports.


   .. py:attribute:: value
      :type:  jax.Array


   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:property:: shape
      :type: tuple[int, Ellipsis]



.. py:class:: SparkPayload

   Bases: :py:obj:`abc.ABC`


   Abstract payload definition to validate exchanges between SparkModule's.


   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



   .. py:property:: shape
      :type: Any



   .. py:property:: dtype
      :type: Any



.. py:class:: SpikeArray(spikes, inhibition_mask = False, async_spikes = False)

   Bases: :py:obj:`SparkPayload`


   Representation of a collection of spike events.

   Init:
       spikes: jax.Array[bool], True if neuron spiked, False otherwise
       inhibition_mask: jax.Array[bool], True if neuron is inhibitory, False otherwise

   The async_spikes flag is automatically set True by delay mechanisms that perform neuron-to-neuron specific delays.
   Note that when async_spikes is True the shape of the spikes changes from (origin_units,) to (origin_units, target_units).
   This is important when implementing new synaptic models, since fully valid synaptic models should be able to handle both cases.


   .. py:attribute:: async_spikes
      :type:  bool
      :value: False



   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:property:: spikes
      :type: jax.Array



   .. py:property:: inhibition_mask
      :type: jax.Array



   .. py:property:: value
      :type: jax.Array



   .. py:property:: shape
      :type: tuple[int, Ellipsis]



   .. py:property:: dtype
      :type: jax.typing.DTypeLike



.. py:class:: CurrentArray

   Bases: :py:obj:`ValueSparkPayload`


   Representation of a collection of currents.


.. py:class:: PotentialArray

   Bases: :py:obj:`ValueSparkPayload`


   Representation of a collection of membrane potentials.


.. py:class:: FloatArray

   Bases: :py:obj:`ValueSparkPayload`


   Representation of a float array.


.. py:class:: IntegerArray

   Bases: :py:obj:`ValueSparkPayload`


   Representation of an integer array.


.. py:class:: BooleanMask

   Bases: :py:obj:`ValueSparkPayload`


   Representation of an inhibitory boolean mask.


.. py:class:: PortSpecs(payload_type, shape, dtype, description = None, async_spikes = None, inhibition_mask = None)

   Base specification for a port of an SparkModule.


   .. py:attribute:: payload_type
      :type:  type[spark.core.payloads.SparkPayload] | None


   .. py:attribute:: shape
      :type:  tuple[int, Ellipsis] | list[tuple[int, Ellipsis]] | None


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike | None


   .. py:attribute:: description
      :type:  str | None
      :value: None



   .. py:attribute:: async_spikes
      :type:  bool | None
      :value: (None,)



   .. py:attribute:: inhibition_mask
      :type:  bool | None
      :value: (None,)



   .. py:method:: to_dict(is_partial = False)

      Serialize PortSpecs to dictionary



   .. py:method:: from_dict(dct, is_partial = False)
      :classmethod:


      Deserialize dictionary to  PortSpecs



   .. py:method:: from_portspecs_list(portspec_list, validate_async = True)
      :classmethod:


      Merges a list of PortSpecs into a single PortSpecs



.. py:class:: PortMap(origin, port)

   Specification for an output port of an SparkModule.


   .. py:attribute:: origin
      :type:  str


   .. py:attribute:: port
      :type:  str


   .. py:method:: to_dict(is_partial = False)

      Serialize PortMap to dictionary



   .. py:method:: from_dict(dct, is_partial = False)
      :classmethod:


      Deserialize dictionary to PortMap



.. py:class:: ModuleSpecs(name, module_cls, inputs, config)

   Specification for SparkModule automatic constructor.


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: module_cls
      :type:  type[spark.core.module.SparkModule]


   .. py:attribute:: inputs
      :type:  dict[str, list[PortMap]]


   .. py:attribute:: config
      :type:  spark.core.config.BaseSparkConfig


   .. py:method:: to_dict(is_partial = False)

      Serialize ModuleSpecs to dictionary



   .. py:method:: from_dict(dct, is_partial = False)
      :classmethod:


      Deserialize dictionary to ModuleSpecs



.. py:function:: split(node, *filters)

   Wrapper around flax.nnx.split to simply imports.


.. py:function:: merge(graphdef, state, /, *states)

   Wrapper around flax.nnx.merge to simply imports.


.. py:data:: register_module

   Decorator used to register a new SparkModule.
   Note that module must inherit from spark.nn.Module (spark.core.module.SparkModule)

.. py:data:: register_initializer

   Decorator used to register a new Initializer.
   Note that module must inherit from spark.nn.initializers.base.Initializer

.. py:data:: register_payload

   Decorator used to register a new SparkPayload.
   Note that module must inherit from spark.SparkPayload (spark.core.payloads.SparkPayload)

.. py:data:: register_config

   Decorator used to register a new SparkConfig.
   Note that module must inherit from spark.nn.BaseConfig (spark.core.config.BaseSparkConfig)

.. py:data:: register_cfg_validator

   Decorator used to register a new ConfigurationValidator.
   Note that module must inherit from spark.core.config_validation.ConfigurationValidator

.. py:data:: REGISTRY

   Registry singleton.

.. py:class:: GraphEditor

   .. py:attribute:: app


   .. py:method:: launch()

      Creates and shows the editor window without blocking.
      This method is safe to call multiple times.



   .. py:method:: exit_editor()

      Exit editor.



   .. py:method:: closeEvent(event)

      Overrides the default close event to check for unsaved changes.



   .. py:method:: new_session()

      Clears the current session after checking for unsaved changes.



   .. py:method:: save_session()

      Saves the current session to a Spark Graph Editor file.



   .. py:method:: save_session_as()

      Saves the current session to a new Spark Graph Editor file.



   .. py:method:: load_session()

      Loads a graph state from a Spark Graph Editor file after checking for unsaved changes.



   .. py:method:: load_from_model()

      Loads a graph state from a Spark configuration file after checking for unsaved changes.



   .. py:method:: export_model()

      Exports the graph state to a Spark configuration file.



   .. py:method:: export_model_as()

      Exports the graph state to a new Spark configuration file.



