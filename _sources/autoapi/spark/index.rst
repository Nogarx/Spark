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
   spark.register_neuron
   spark.register_initializer
   spark.register_payload
   spark.register_config
   spark.register_cfg_validator
   spark.register_interface
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
   spark.property


Functions
---------

.. autoapisummary::

   spark.jit
   spark.eval_shape
   spark.split
   spark.merge
   spark.register_neuron_from_config
   spark.register_neuron_from_config_file


Package Contents
----------------

.. py:class:: Constant(data, dtype = None)

   Representation of a constant array/object.

   Holds a quantity fixed at build time, such as a decay constant or a delay kernel.
   Assigning to ``value`` raises `AttributeError`; a quantity that changes belongs in a
   `Variable`.

   Registered as a static pytree node, so it travels in the treedef rather than as a traced
   leaf.


   .. py:property:: value
      :type: jax.Array



   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:property:: shape
      :type: tuple[int, ...]



   .. py:property:: dtype
      :type: Any



   .. py:property:: ndim
      :type: int



   .. py:property:: size
      :type: int



   .. py:property:: T
      :type: jax.Array



.. py:class:: Variable(value, dtype = None, **metadata)

   Bases: :py:obj:`flax.nnx.Variable`


   Representation of a variable array/object.

   Wrapper around the Flax variable, to simplify imports. The dtype given at construction is
   applied once, to the initial value; a later assignment to ``value`` is converted to an
   array but keeps its own dtype.


   .. py:property:: value
      :type: jax.Array



   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:property:: shape
      :type: tuple[int, ...]



.. py:class:: SparkPayload

   Bases: :py:obj:`abc.ABC`


   Base class for the values modules exchange.

   A payload names what a quantity is, not only how it is shaped. A port declares the payload
   it carries, and the framework refuses a connection between two ports of different types,
   so a current cannot be wired where a potential is expected.

   Every payload is registered as a pytree, so it crosses a jit boundary as data.

   .. seealso::

      :py:obj:`ValueSparkPayload`
          Payloads holding a single array.

      :py:obj:`SpikeArray`
          Spikes and their inhibition mask, bit-packed.


   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



   .. py:property:: shape
      :type: Any



   .. py:property:: dtype
      :type: Any



.. py:class:: SpikeArray(spikes, inhibition_mask = None, async_spikes = False)

   Bases: :py:obj:`SparkPayload`


   Spike events of a pool, with the sign of each unit.

   The spike bit and the inhibition bit of every unit are packed into one ``uint8`` array,
   so the two travel together and a downstream module cannot read one without the other.

   Inhibition schema
   Excitatory -> + or 0
   Inhibitory -> - or 1

   Encoding schema
   (Spike bit, Inhibition bit)
   0: (False, False) ->  0
   1: (True,  False) ->  1
   2: (False, True)  -> -0
   3: (True,  True)  -> -1

   :param spikes: Non-zero where a unit spiked.
   :type spikes: jax.Array
   :param inhibition_mask: True where a unit is inhibitory. Broadcast against ``spikes`` when it has fewer
                           dimensions. Defaults to all excitatory.
   :type inhibition_mask: BooleanMask or jax.Array or bool, optional
   :param async_spikes: Marks one entry per (target, origin) pair rather than one per origin.
   :type async_spikes: bool, default False

   .. attribute:: spikes

      The spike bit, as bool.

      :type: jax.Array

   .. attribute:: inhibition_mask

      The inhibition bit, as bool.

      :type: jax.Array

   .. attribute:: value

      The signed spikes: ``+1`` for an excitatory spike, ``-1`` for an inhibitory one, ``0``
      for no spike.

      :type: jax.Array

   .. rubric:: Notes

   ``async_spikes`` is set by the delay models that give every connection its own delay, such
   as `N2NDelays`. The shape then grows from ``(origin_units,)`` to
   ``(target_units, origin_units)``. A synapse model that means to accept both forms has to
   read the flag and sum over the origin axes only.


   .. py:attribute:: async_spikes
      :type:  bool
      :value: False



   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:method:: __eq__(other)


   .. py:property:: spikes
      :type: jax.Array



   .. py:property:: inhibition_mask
      :type: jax.Array



   .. py:property:: value
      :type: jax.Array



   .. py:property:: shape
      :type: tuple[int, ...]



   .. py:property:: dtype
      :type: jax.typing.DTypeLike



.. py:class:: CurrentArray

   Bases: :py:obj:`ValueSparkPayload`


   Synaptic current, in pA.

   Produced by a synapse model and consumed by a soma.


.. py:class:: PotentialArray

   Bases: :py:obj:`ValueSparkPayload`


   Membrane potential, in mV.

   Exposed by a soma as its ``potential`` property. Whether it is measured from rest or in
   absolute mV depends on the soma model.


.. py:class:: FloatArray

   Bases: :py:obj:`ValueSparkPayload`


   Array of floats, with no unit attached.

   Used for synaptic weights and for the signals that are neither currents nor potentials,
   such as the third factor of a modulated plasticity rule.


.. py:class:: IntegerArray

   Bases: :py:obj:`ValueSparkPayload`


   Array of integers, with no unit attached.

   Used for conduction delays, which are counted in steps.


.. py:class:: BooleanMask

   Bases: :py:obj:`ValueSparkPayload`


   Boolean mask over the units of a pool.

   Used for the inhibition mask a `Neuron` hands to its modules.


.. py:class:: PortSpecs(payload_type, shape, dtype, description = None)

   Module port specification.

   Names the payload type, and the shape and dtype once they are known. Shape and dtype are
   None until the module is built, since they are inferred from the values that reach it.

   :param payload_type: Type the port carries. Two ports connect only if this matches.
   :type payload_type: type of SparkPayload or None
   :param shape: Shape of the payload. A list when several values arrive on the port.
   :type shape: tuple of int or list of tuple of int or None
   :param dtype: Dtype of the payload.
   :type dtype: DTypeLike or None
   :param description: Human readable description of the port.
   :type description: str, optional


   .. py:attribute:: payload_type
      :type:  type[spark.core.payloads.SparkPayload] | None


   .. py:attribute:: shape
      :type:  tuple[int, ...] | list[tuple[int, ...]] | None


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike | None


   .. py:attribute:: description
      :type:  str | None


   .. py:method:: to_dict()

      Serializes the specification to a dictionary.

      :returns: The fields of the specification, with the payload type as its registered name.
      :rtype: dict



   .. py:method:: from_dict(dct)
      :classmethod:


      Builds a specification from a dictionary.

      :param dct: As produced by `to_dict`.
      :type dct: dict

      :rtype: PortSpecs



   .. py:method:: from_payload(payload)
      :classmethod:


      Builds a specification describing an existing payload.

      :param payload: Payload to read the type, shape and dtype from.
      :type payload: SparkPayload

      :rtype: PortSpecs



   .. py:method:: from_portspecs_list(portspec_list)
      :classmethod:


      Merges several specifications into one.

      Used for a port fed by more than one source, whose values are concatenated.

      :param portspec_list: Specifications to merge. All must carry the same payload type.
      :type portspec_list: list of PortSpecs

      :returns: The shared payload type, the promoted dtype and the merged shape. The list itself is
                returned unchanged when it holds a single entry.
      :rtype: PortSpecs

      :raises TypeError: If the specifications do not all carry the same payload type.



.. py:class:: PortMap(origin, port, is_property = False)

   Module's connections specification.

   A pair of the form (module_name, module_port_name) that specifies a connection within a controller.
   ``'__call__'`` and ``'__self__'`` are speciail module_names used to refer to the controller's inputs
   and the same module defining the mapping.


   :param origin: Name of the module the value comes from. ``'__call__'`` for an input of the enclosing
                  controller, ``'__self__'`` for a property of the controller itself.
   :type origin: str
   :param port: Name of the port on that module.
   :type port: str
   :param is_property: Read a property of the origin rather than one of its outputs.
   :type is_property: bool, default False


   .. py:attribute:: origin
      :type:  str


   .. py:attribute:: port
      :type:  str


   .. py:attribute:: is_property
      :type:  bool


   .. py:method:: to_dict()

      Serializes the map to a dictionary.

      :returns: The origin, the port and the property flag.
      :rtype: dict



   .. py:method:: from_dict(dct)
      :classmethod:


      Builds a map from a dictionary.

      :param dct: As produced by `to_dict`.
      :type dct: dict

      :rtype: PortMap



   .. py:method:: __hash__()


   .. py:method:: __eq__(other)


.. py:class:: ModuleSpecs(name, module_cls, inputs, config = None, outputs = None, effects = None)

   Specification of a module within a controller.

   This is what makes a model data rather than code: a controller holds a tuple of these, so
   it can be written to a file, edited, and instantiated again without a Python definition.

   :param name: Name the module answers to inside the controller. Also the attribute it is bound to.
   :type name: str
   :param module_cls: Class to instantiate. Must be registered.
   :type module_cls: type of SparkModule
   :param inputs: For each input port of the module, where its value comes from. Several entries for one
                  port are concatenated in order.
   :type inputs: dict of str to PortMap or list of PortMap
   :param config: Configuration of the module. The default configuration is used when omitted.
   :type config: SparkConfig, optional
   :param outputs: Output ports of the module to expose as output ports of the controller, as
                   ``{controller port: module port}``.
   :type outputs: dict of str to str, optional
   :param effects: Properties of this module to write after the step, as ``{property: source}``. A
                   plasticity rule writes weights back onto a synapse this way. The property must have a
                   setter.
   :type effects: dict of str to PortMap or list of PortMap, optional


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: module_cls
      :type:  type[spark.core.module.SparkModule]


   .. py:attribute:: inputs
      :type:  dict[str, Iterable[PortMap]]


   .. py:attribute:: outputs
      :type:  dict[str, str]


   .. py:attribute:: effects
      :type:  dict[str, Iterable[PortMap]]


   .. py:attribute:: config
      :type:  spark.core.config.SparkConfig


   .. py:method:: to_dict()

      Serializes the specification to a dictionary.

      :returns: The name, the registered name of the module class, the wiring and the configuration.
      :rtype: dict



   .. py:method:: from_dict(dct)
      :classmethod:


      Builds a specification from a dictionary.

      :param dct: As produced by `to_dict`. The module class is looked up in the registry by name.
      :type dct: dict

      :rtype: ModuleSpecs



.. py:class:: property(fget=None, fset=None, fdel=None, doc=None)

   Declares a property port on a module.

   Behaves like the built-in property, and additionally marks the attribute as a port the
   framework can wire. The getter must be annotated with the `SparkPayload` it returns, which
   is what the port carries.

   A property with no setter is read only: other modules may read it, but it cannot be the
   target of an effect.

   .. rubric:: Examples

   >>> class Synapses(Component):
   ...     @spark_property
   ...     def kernel(self) -> FloatArray:
   ...         return FloatArray(self._kernel.value)
   ...
   ...     @kernel.setter
   ...     def kernel(self, new_kernel: FloatArray) -> None:
   ...         self._kernel.value = new_kernel.value


   .. py:attribute:: fget
      :value: None



   .. py:attribute:: fset
      :value: None



   .. py:attribute:: fdel
      :value: None



   .. py:attribute:: __doc__
      :value: None



   .. py:method:: __set_name__(owner, name)


   .. py:method:: __get__(obj, objtype=None)


   .. py:method:: __set__(obj, value)


   .. py:method:: __delete__(obj)


   .. py:method:: getter(fget)


   .. py:method:: setter(fset)


   .. py:method:: deleter(fdel)


.. py:function:: jit(*args, **kwargs)

   Wrapper around flax.nnx.jit, to simplify imports.


.. py:function:: eval_shape(*args, **kwargs)

   Wrapper around flax.nnx.eval_shape, to simplify imports.


.. py:function:: split(*args, **kwargs)

   Wrapper around flax.nnx.split, to simplify imports.


.. py:function:: merge(*args, **kwargs)

   Wrapper around flax.nnx.merge, to simplify imports.


.. py:data:: register_module

   Decorator used to register a new SparkModule.
   Note that module must inherit from spark.nn.Module (spark.core.module.SparkModule)

.. py:data:: register_neuron

   Decorator used to register a new Neuron model.
   Note that module must inherit from spark.nn.Neuron (spark.nn.controllers.neuron.Neuron)

.. py:data:: register_initializer

   Decorator used to register a new Initializer.
   Note that module must inherit from spark.nn.initializers.base.Initializer

.. py:data:: register_payload

   Decorator used to register a new SparkPayload.
   Note that module must inherit from spark.SparkPayload (spark.core.payloads.SparkPayload)

.. py:data:: register_config

   Decorator used to register a new SparkConfig.
   Note that module must inherit from spark.nn.BaseConfig (spark.core.config.SparkConfig)

.. py:data:: register_cfg_validator

   Decorator used to register a new ConfigurationValidator.
   Note that module must inherit from spark.core.config_validation.ConfigurationValidator

.. py:data:: register_interface

   Decorator used to register a new Interface.
   Note that module must inherit from spark.nn.interfaces.base.Interface

.. py:function:: register_neuron_from_config(cls_name, config)

   Registers a (Neuron, NeuronConfig) pair built from a configuration instance.

   This is what turns a model designed in the editor into a class that can be placed in a
   `Brain` like any other neuron.

   :param name: Name the pair answers to.
   :type name: str
   :param config: Configuration whose modules and values become the defaults of the pair.
   :type config: NeuronConfig

   :returns: The registered class.
   :rtype: type of Neuron


.. py:function:: register_neuron_from_config_file(cls_name, path)

   Registers a (Neuron, NeuronConfig) pair built from a .scfg file.

   :param name: Name the pair answers to.
   :type name: str
   :param file_path: File holding the configuration.
   :type file_path: str

   :returns: The registered class.
   :rtype: type of Neuron


.. py:data:: REGISTRY

   Registry singleton.

