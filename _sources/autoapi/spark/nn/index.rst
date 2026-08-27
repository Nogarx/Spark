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

   Bases: :py:obj:`spark.core.backend.Module`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ , :py:obj:`InputT`\ ]


   Base class for the modules of a network.

   A module owns its state, declares its ports through the signature of ``__call__``, and is
   built the first time it is called. Its configuration is a `SparkConfig` subclass, declared
   through the ``config`` annotation.

   :param config: Module configuration. Its fields may also be given as keyword arguments.
   :type config: SparkConfig

   .. rubric:: Notes

   Ports are read from the signature. The parameters of ``__call__`` are the input ports,
   the TypedDict it returns names the output ports, and every `spark_property` is a property port.

   Shape inference happens once, on the first call: `build` receives the payloads that
   arrived and initializes whatever depends on their shape.

   .. seealso::

      :py:obj:`SparkConfig`
          The configuration a module is built from.

      :py:obj:`spark_property`
          Declares a property port.


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


      Returns the configuration class this module is built from.

      :returns: The class named by the ``config`` annotation.
      :rtype: type of SparkConfig



   .. py:method:: recurrent_contract()

      Returns what the outputs and properties of this module look like before it has run.

      A module that defines this method may form a closed cycles.

      :returns: * **output_contract_specs** (*dict of str to SparkPayload*) -- Mock payload per output port.
                * **property_contract_specs** (*dict of str to SparkPayload*) -- Mock payload per property port.

      :raises NotImplementedError: If the module declares no contract. Check `has_recurrent_contract` first.



   .. py:method:: has_recurrent_contract()
      :classmethod:


      Whether the module declares a recurrent contract.

      :returns: True if `recurrent_contract` may be called.
      :rtype: bool



   .. py:method:: build(**abc_kwargs)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Returns the state of the module to its initial value.

      Leaves the parameters drawn at build time untouched.



   .. py:method:: set_recurrent_contract(output_contract_specs, property_contract_specs)

      Fixes the specifications a module presents while a cycle is being resolved.

      :param output_contract_specs: Specification per output port.
      :type output_contract_specs: dict of str to PortSpecs
      :param property_contract_specs: Specification per property port.
      :type property_contract_specs: dict of str to PortSpecs



   .. py:method:: get_contract_specs()

      Returns the recurrent contract set on this module.

      :returns: * **output_contract_specs** (*dict of str to PortSpecs*)
                * **property_contract_specs** (*dict of str to PortSpecs*)



   .. py:method:: get_input_specs()

      Returns the input port specifications of this instance.

      :returns: One entry per input port, with the shape and dtype seen at build time.
      :rtype: dict of str to PortSpecs

      :raises RuntimeError: If the module has not been built.



   .. py:method:: get_output_specs()

      Returns the output port specifications of this instance.

      :returns: One entry per output port, with the shape and dtype seen at build time.
      :rtype: dict of str to PortSpecs

      :raises RuntimeError: If the module has not been built.



   .. py:method:: get_property_specs()

      Returns the property port specifications of this instance.

      :returns: One entry per property port, with the shape and dtype seen at build time.
      :rtype: dict of str to PortSpecs

      :raises RuntimeError: If the module has not been built.



   .. py:method:: get_rng_keys(num_keys)

      Draws new keys from the random engine of the module.

      Advances the internal key, so two calls never return the same keys.

      :param num_keys: Number of keys to draw.
      :type num_keys: int

      :returns: A single key when ``num_keys`` is 1, a list otherwise.
      :rtype: jax.Array or list of jax.Array

      :raises RuntimeError: If the configuration of the module declares no seed.



   .. py:method:: get_properties()
      :classmethod:


      Returns the names of every `spark_property` of this class.

      :rtype: tuple of str



   .. py:method:: get_readonly_properties()
      :classmethod:


      Returns the names of every `spark_property` of this class that has no setter.

      A read-only property can be read by other modules but cannot be the target of an effect.

      :rtype: tuple of str



   .. py:method:: __call__(**kwargs)
      :abstractmethod:


      Execution method.



   .. py:method:: __repr__()


   .. py:method:: inspect()

      Prints the tree of modules held by this one.



   .. py:method:: checkpoint(path, overwrite=False)


   .. py:method:: from_checkpoint(path, safe=True)
      :classmethod:



.. py:class:: DefaultConfig

   Bases: :py:obj:`SparkConfig`


   Configuration of a module, with the fields every module needs.

   :param seed: Seed for the random draws of the module. Drawn from the operating system when omitted.
   :type seed: int, optional
   :param dtype: Dtype used for the internal state.
   :type dtype: DTypeLike, default jnp.float16
   :param dt: Integration step, in ms.
   :type dt: float, default 1.0


   .. py:attribute:: seed
      :type:  int


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


   .. py:attribute:: dt
      :type:  float


.. py:class:: Config

   Bases: :py:obj:`abc.ABC`


   Base class for the configuration of a module.

   A configuration is a frozen dataclass carrying the parameters of a module. It is
   serializable, so a model can be written to a file and read back without a Python
   definition, and it validates its fields as they are set.

   .. rubric:: Notes

   Every annotated attribute becomes a field. The metadata of a field may declare
   ``validators``, ``units``, a ``description`` and ``value_options``, which the editor and
   the validators read.

   A field may be given an `Initializer` in place of a value. The array is then drawn at
   build time, once the shape is known, and is reached through ``config.init.<field>``.

   Fields named ``dt`` and ``units`` are handed down by a controller to every configuration
   it contains, so a pool is sized and clocked in one place.

   .. seealso::

      :py:obj:`DefaultSparkConfig`
          Adds the seed, dtype and dt every module needs.


   .. py:method:: partial(**kwargs)
      :classmethod:



   .. py:method:: merge(**kwargs)

      Returns a copy of this configuration with the given values written over it.

      :param \*\*kwargs: Values by field name.

      :returns: A new configuration. This one is left unchanged.
      :rtype: SparkConfig



   .. py:property:: init


   .. py:property:: class_ref
      :type: type


      Returns the module or initializer class this configuration belongs to.

      :rtype: type


   .. py:method:: __iter__()

      Iterates over the fields of the configuration.

      :Yields: * **field_name** (*str*) -- Name of the field.
               * **field_value** (*Any*) -- Value the field holds.



   .. py:method:: inspect(simplified=False)

      Prints the tree of fields of this configuration.



   .. py:method:: with_new_seeds(seed = None)

      Returns a copy of this configuration with every seed redrawn.

      :returns: A new configuration. This one is left unchanged.
      :rtype: SparkConfig



   .. py:method:: to_dict()

      Serializes the configuration to a dictionary.

      :rtype: dict



   .. py:method:: from_dict(dct)
      :classmethod:


      Builds a configuration from a dictionary.

      :param dct: As produced by `to_dict`.
      :type dct: dict

      :rtype: SparkConfig



   .. py:method:: to_file(file_path, compress = True, verbose = True, metadata = None)

      Writes the configuration to a .scfg file.

      :param file_path: Where to write.
      :type file_path: str
      :param compress: Compress the file.
      :type compress: bool, default True
      :param verbose: Log where the file was written.
      :type verbose: bool, default True
      :param metadata: Written beside the configuration. `from_file` does not read it back; use
                       `metadata_from_file` for that. The editor stores node positions here.
      :type metadata: dict, optional



   .. py:method:: metadata_from_file(file_path)
      :classmethod:


      Reads the metadata written beside the configuration of a file.

      The configuration itself is not decoded, so this works for a file naming models that are
      not registered.

      :param file_path: File to read.
      :type file_path: str

      :returns: What the writer stored, empty when it stored nothing.
      :rtype: dict



   .. py:method:: from_file(file_path)
      :classmethod:


      Builds a configuration from a .scfg file.

      :param file_path: File to read.
      :type file_path: str

      :rtype: SparkConfig



.. py:class:: Brain(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.base.Controller`


   Network of neurons and interfaces.

   Every module reads from a cache holding the outputs of the previous step, updates its own
   state, and writes its outputs back. Because no module waits for another, any wiring is
   legal, cycles included, and the modules of one step are independent of each other.
   Note that due to implementation details, modules within this controller have a one step
   latency per connection; which, for most cases, is negligible.

   :param config: Controller configuration. Its fields may also be given as keyword arguments.
   :type config: BrainConfig, optional

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Derived from the modules: a module input wired to ``__call__`` becomes an input port of
                 the controller, under the name the port map gives it.

   :Output Ports: **\*\*outputs** (*SparkPayload*) -- Derived from the modules: a module output named in ``outputs`` becomes an output port of
                  the controller.

   .. rubric:: Notes

   Input and output ports are derived from the modules, as for any controller.

   .. seealso::

      :py:obj:`Neuron`
          Controller without the cache, where a step is ordered by its dependencies.


   .. py:attribute:: config
      :type:  BrainConfig


   .. py:method:: build(**abc_args)


   .. py:method:: __call__(**inputs)

      Advances every module one step against the cache.

      Every module reads the outputs of the previous step, so the modules of one step are
      independent of each other. The cache is written once all of them have run.

      :param \*\*inputs: One entry per input port of the brain, as derived from the modules.
      :type \*\*inputs: SparkPayload

      :returns: One entry per output port of the brain, as derived from the modules.
      :rtype: dict of str to SparkPayload



   .. py:method:: read_state(port_list)

      Returns the current state of the modules/cache.



.. py:class:: BrainConfig

   Bases: :py:obj:`spark.nn.controllers.base.ControllerConfig`


   Configuration for `Brain`.

   :param modules_specs: The neurons and interfaces the brain holds, and how their ports are wired.
   :type modules_specs: tuple of ModuleSpecs
   :param seed: Seed for the random draws of the brain and its modules. Drawn from the operating
                system when omitted.
   :type seed: int, optional
   :param dt: Integration step, in ms. Handed down to every module.
   :type dt: float, default 1.0


.. py:class:: Neuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.base.Controller`


   Pool of neurons built from components.

   A neuron holds the components one neuron model is made of, such as delays, synapses, a
   soma and a plasticity rule, and steps them in dependency order within a single timestep.
   A module therefore reads what the modules before it produced on the same step, rather than
   on the previous one.

   Because the step is ordered, a cycle in the wiring can only be resolved if the module
   closing it declares a recurrent contract, which states what its outputs look like before
   it has run.

   :param config: Controller configuration. Its fields may also be given as keyword arguments.
   :type config: NeuronConfig

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Derived from the modules: a module input wired to ``__call__`` becomes an input port of
                 the controller, under the name the port map gives it.

   :Output Ports: **\*\*outputs** (*SparkPayload*) -- Derived from the modules: a module output named in ``outputs`` becomes an output port of
                  the controller.

   :Properties: **inhibition_mask** (*BooleanMask*) -- Marks the inhibitory units of the pool. Read only, and supplied to any module declaring
                an ``inhibition_mask`` input without being wired.

   .. rubric:: Notes

   Which units are inhibitory is a property of the pool, not of the module that emits the
   spikes. The mask is drawn once from ``inhibitory_rate`` and handed to any module declaring
   an ``inhibition_mask`` input, without being wired. From there the spikes carry the
   distinction themselves. A declared connection takes precedence.

   ``inhibition_mask`` is exposed as a read-only property, so it can be read by the graph but
   not written.

   .. seealso::

      :py:obj:`Brain`
          Controller that steps its modules against a cache of the previous step.

      :py:obj:`LIFNeuron`
          Prebuilt leaky integrate-and-fire neuron.


   .. py:attribute:: config
      :type:  NeuronConfig


   .. py:attribute:: units
      :value: ()



   .. py:method:: recurrent_contract()

      Returns expected-like outputs and properties of the module.

      This function is a binding contract that allows the modules to accept self connections.



   .. py:method:: has_recurrent_contract()
      :classmethod:


      Returns True if the modules defines a recurrent contract, False otherwise.



   .. py:method:: inhibition_mask()


   .. py:method:: build(**abc_args)


   .. py:method:: __call__(**inputs)

      Advances every module one step, in dependency order.

      A module reads what the modules before it produced on this same step. Effects are applied
      once every module has run.

      :param \*\*inputs: One entry per input port of the neuron, as derived from the modules.
      :type \*\*inputs: SparkPayload

      :returns: One entry per output port of the neuron, as derived from the modules.
      :rtype: dict of str to SparkPayload



   .. py:method:: read_state(port_list)

      Returns the current state of the modules/cache.



.. py:class:: NeuronConfig

   Bases: :py:obj:`spark.nn.controllers.base.ControllerConfig`


   Configuration for `Neuron`.

   :param modules_specs: The components the neuron holds, and how their ports are wired.
   :type modules_specs: tuple of ModuleSpecs
   :param units: Shape of the pool of neurons.
   :type units: tuple of int
   :param inhibitory_rate: Fraction of the pool that is inhibitory. Must lie in ``[0, 1]``.
   :type inhibitory_rate: float, default 0.2
   :param seed: Seed for the random draws of the neuron and its modules. Drawn from the operating
                system when omitted.
   :type seed: int, optional
   :param dt: Integration step, in ms.
   :type dt: float, default 1.0

   .. rubric:: Notes

   ``dt`` and ``units`` are handed down to every configuration the neuron contains, so a pool
   is sized and clocked in one place. Both names are reserved for that purpose.


   .. py:attribute:: units
      :type:  tuple[int, ...]


   .. py:attribute:: inhibitory_rate
      :type:  float


   .. py:method:: __post_init__()


