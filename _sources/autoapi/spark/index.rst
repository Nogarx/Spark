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
   /autoapi/spark/recording/index


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
   spark.Partition


Functions
---------

.. autoapisummary::

   spark.jit
   spark.scan
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


.. py:function:: jit(fun = None, /, **options)

   Compiles a function with ``jax.jit``. Its calls are recorded while a recorder is open.

   :param fun: The function to compile. Without it, returns a decorator taking ``options``.
   :type fun: callable, optional
   :param \*\*options: Passed to ``jax.jit``, such as ``static_argnames`` or ``donate_argnames``.

   :returns: The compiled function. Without an open recorder, it is ``jax.jit(fun, **options)``, or
             ``flax.nnx.jit(fun, **options)`` for calls with modules among their arguments.
   :rtype: Jit

   .. seealso::

      :py:obj:`Jit`
          How a call is recorded.

      :py:obj:`scan`
          ``jax.lax.scan``, recorded within a `Jit` while a recorder is open.

   .. rubric:: Examples

   >>> @partial(spark.jit, static_argnames=['steps'])
   ... def run(graph, state, steps, **inputs):
   ...     def step(state, _):
   ...         model = spark.merge(graph, state)
   ...         outputs = model(**inputs)
   ...         return spark.split(model)[1], outputs
   ...     return spark.scan(step, state, length=steps)


.. py:function:: scan(f, init, xs = None, length = None, reverse = False, unroll = 1, _split_transpose = False)

   ``jax.lax.scan``, recorded within a call of a `Jit` while a recorder is open.

   Otherwise, it is ``jax.lax.scan``, and traces to the same program.

   Each step of the scan is one step of the model: ``f`` calls the model once. Within a
   recorded call, the probes of the call are recorded on every step, and the records are
   returned by the `Jit` to the recorder. What ``f`` returns is unchanged.

   :param f: As for ``jax.lax.scan``.
   :param init: As for ``jax.lax.scan``.
   :param xs: As for ``jax.lax.scan``.
   :param length: As for ``jax.lax.scan``.
   :param reverse: As for ``jax.lax.scan``.
   :param unroll: As for ``jax.lax.scan``.
   :param _split_transpose: As for ``jax.lax.scan``.

   :returns: As ``jax.lax.scan`` returns them.
   :rtype: carry, ys

   :raises RuntimeError: Within a recorded call, when ``f`` does not call the model once per step, or when the
       scan runs within another transformation.
   :raises ValueError: Within a recorded call, with ``reverse``.

   .. seealso::

      :py:obj:`jit`
          Compiles a function whose calls a recorder records.


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


.. py:class:: Partition(brain, parts)

   A brain divided among devices by its modules.

   Each device runs a part of the brain: a sub-brain, a `Brain` holding the modules of that
   part, with their configurations and their state. A module input wired to a module of
   another part becomes an input of the sub-brain, and the output it reads becomes an output of
   the sub-brain of that other part, both named ``'<origin>:<port>'``. The inputs and the
   outputs of the brain stay with the modules reading and producing them.

   In a brain, every module reads what the others produced on the step before, so the
   sub-brains of a step do not wait for each other: each device runs its own, and `exchange`
   copies the outputs that cross devices once every sub-brain has run. A step of every
   sub-brain followed by an exchange is a step of the brain.

   :param brain: A built brain. The sub-brains hold its modules.
   :type brain: Brain
   :param parts: The names of the modules each device runs. Every module of the brain is in one part.
   :type parts: dict of jax.Device to iterable of str

   :raises RuntimeError: If the brain is not built.
   :raises ValueError: If a module is in no part or in two, if a part is empty or names a module the brain
       does not have, or if a module reads a property of a module of another part, or has a
       property written by one.

   .. rubric:: Notes

   Properties are read and effects are applied within a step, so a module reading a property
   of another, or whose property another writes, is on the same device as that module.

   A sub-brain is called as the brain is, within ``jax.jit`` and ``jax.lax.scan``; a call
   compiles for the device its state is on and returns at once, so the devices run together.

   The devices may belong to several processes (``jax.distributed``). Every process builds
   the same brain, from a configuration with its seed, and the same partition, and runs the
   sub-brains of its own devices, `local_devices`. `initial`, `exchange` and `merge` gather
   what crosses processes with an all-gather, which every process calls in the same order;
   an output read within its own process is copied from device to device. On the processor,
   the processes exchange through gloo (``jax_cpu_collectives_implementation``).

   `run` calls the sub-brains of the devices of the process, and records them while a recorder
   of `spark.recording` is open, as a call running the brain is recorded.

   .. rubric:: Examples

   >>> partition = spark.Partition(brain, {gpu0: ['spiker', 'first_pool'], gpu1: ['second_pool', 'readout']})
   >>> graphs, states = partition.split(brain)
   >>> received = partition.initial(states)
   >>> for _ in range(steps):
   ...     outputs, states = partition.run(run, graphs, states, received, inputs, 1)
   ...     received = partition.exchange(outputs)
   >>> brain = partition.merge(states)

   ``run`` is the function running the brain, ``run(graph, state, steps, **inputs)``, which
   returns the outputs and the state. Called by hand, the loop over the devices is

   >>> outputs = {}
   >>> for device in partition.local_devices:
   ...     outputs[device], states[device] = run(graphs[device], states[device], 1,
   ...                                           **partition.inputs(device, inputs), **received[device])


   .. py:method:: balanced(brain, devices, tolerance = 0.1)
      :classmethod:


      Returns a partition of a brain over devices that balances their work and keeps what
      crosses devices small.

      :param brain: A built brain.
      :type brain: Brain
      :param devices: The devices to divide the brain among, in order of preference. A device gets no part
                      if the brain has fewer modules to place than devices.
      :type devices: sequence of jax.Device
      :param tolerance: How much more work than the lightest device a device may take, as a fraction, to hold
                        a module next to those it exchanges outputs with.
      :type tolerance: float, default 0.1

      :rtype: Partition

      :raises RuntimeError: If the brain is not built.
      :raises ValueError: If no device is given.

      .. rubric:: Notes

      The work of a module is the number of values in its state, which its weights, traces and
      buffers make up. Modules that read a property of one another, or write one, are placed
      together. The modules are placed from the heaviest, each on the device that keeps the
      work within ``tolerance`` and exchanges the most with it; then a module is moved to
      another device while the move lowers what crosses devices and keeps every device within
      ``1 + tolerance`` times the heaviest load, or an even share if that is more. What crosses
      is counted in values per step: the size of an output, once for every other device
      reading it.

      The parts depend on the brain and the devices only, so every process gets the same.



   .. py:property:: devices
      :type: tuple[jax.Device, ...]


      The devices of the partition, in the order the parts were given, in every process.


   .. py:property:: local_devices
      :type: tuple[jax.Device, ...]


      The devices of the partition that belong to this process, in the order the parts were
      given: every device, with one process.


   .. py:property:: parts
      :type: dict[jax.Device, tuple[str, ...]]


      The names of the modules each device runs, in the order of the brain.


   .. py:property:: configs
      :type: dict[jax.Device, spark.nn.controllers.brain.BrainConfig]


      The configuration of the sub-brain of each device.


   .. py:method:: device_of(name)

      Returns the device that runs the module ``name``.



   .. py:method:: split(brain)

      Returns the graph and the state of the sub-brain of every device of this process, the
      state placed on its device.

      :param brain: The brain the partition was made from, or one of the same structure, such as one
                    returned by `merge`. Its modules give the state of the sub-brains.
      :type brain: Brain

      :returns: * **graphs** (*dict of jax.Device to GraphDef*) -- The graph of each sub-brain.
                * **states** (*dict of jax.Device to State*) -- The state of each sub-brain, on its device.



   .. py:method:: merge(states, device = None)

      Returns the brain whose modules have the states of the sub-brains.

      With several processes, every process calls it, and every process gets the brain: the
      states of the sub-brains of the other processes are gathered.

      :param states: The state of the sub-brain of each device of this process.
      :type states: dict of jax.Device to State
      :param device: The device the brain is placed on. The first device of the partition in this process
                     when omitted.
      :type device: jax.Device, optional

      :returns: The brain, as a brain that ran unpartitioned would be.
      :rtype: Brain



   .. py:method:: run(function, graphs, states, received, inputs, *args, **kwargs)

      Runs the sub-brain of every device of this process, and returns their outputs and states.

      :param function: Runs a model: ``function(graph, state, *args, **inputs, **kwargs)`` returns its outputs and
                       its state, as the loop running the brain does. A `spark.jit` function is recorded.
      :type function: callable
      :param graphs: The graph of each sub-brain, as `split` gives them.
      :type graphs: dict of jax.Device to GraphDef
      :param states: The state of each sub-brain.
      :type states: dict of jax.Device to State
      :param received: What each sub-brain reads from the others, as `initial` or `exchange` gives it.
      :type received: dict of jax.Device to dict of str to SparkPayload
      :param inputs: The inputs of the brain, by name.
      :type inputs: dict of str to SparkPayload
      :param \*args: Passed to ``function`` after the graph and the state, such as the steps of the call.
      :param \*\*kwargs: Passed to ``function`` after the graph and the state, such as the steps of the call.

      :returns: * **outputs** (*dict of jax.Device to dict of str to SparkPayload*) -- The outputs ``function`` returned for each device.
                * **states** (*dict of jax.Device to State*) -- The state of each sub-brain after the call.

      :raises ValueError: With an open recorder, when a probe addresses the brain rather than a module or an
          input of the brain.
      :raises NotImplementedError: With an open recorder, when the partition spans several processes.

      .. rubric:: Notes

      While a recorder of `spark.recording` is open, the calls of a `spark.jit` function on the
      devices are one call of the recorder, as a call running the brain is. Each records the
      probes of the modules of its sub-brain, and an input of the brain is recorded by the first
      device reading it. The records are joined on the first device and handed over once. The
      calls on the devices run the same steps.



   .. py:method:: inputs(device, inputs)

      Returns the inputs of the brain that the sub-brain of ``device`` reads, placed on that device.

      :param device: A device of the partition in this process.
      :type device: jax.Device
      :param inputs: The inputs of the brain, by name.
      :type inputs: dict of str to SparkPayload

      :rtype: dict of str to SparkPayload



   .. py:method:: initial(states)

      Returns what each sub-brain reads from the others on its next step, from their states.

      The cache of a sub-brain holds the outputs of its last step, so this is also what
      `exchange` returns after that step.

      :param states: The state of the sub-brain of each device of this process.
      :type states: dict of jax.Device to State

      :returns: For each device of this process, the outputs of the other parts its sub-brain reads,
                on that device.
      :rtype: dict of jax.Device to dict of str to SparkPayload



   .. py:method:: exchange(outputs)

      Copies the outputs that cross devices to the devices reading them.

      :param outputs: The outputs of the step of the sub-brain of each device of this process.
      :type outputs: dict of jax.Device to dict of str to SparkPayload

      :returns: For each device of this process, the outputs of the other parts its sub-brain reads,
                on that device. The copies between devices of one process are asynchronous; those
                between processes wait for every process to have run its step.
      :rtype: dict of jax.Device to dict of str to SparkPayload



   .. py:method:: outputs(outputs)

      Returns the outputs of the brain, from the outputs of the sub-brains.

      :param outputs: The outputs of the sub-brain of each device of this process.
      :type outputs: dict of jax.Device to dict of str to SparkPayload

      :returns: One entry per output port of the brain produced in this process, on the device
                producing it.
      :rtype: dict of str to SparkPayload



.. py:data:: REGISTRY

   Registry singleton.

