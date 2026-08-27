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

   Bases: :py:obj:`spark.core.backend.Module`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for controllers.

   A controller holds a set of modules and the wiring between them, and steps them in order.

   :param config: Controller configuration. Its fields may also be given as keyword arguments.
   :type config: ControllerConfig, optional

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Derived from the modules: a module input wired to ``__call__`` becomes an input port of
                 the controller, under the name the port map gives it.

   :Output Ports: **\*\*outputs** (*SparkPayload*) -- Derived from the modules: a module output named in ``outputs`` becomes an output port of
                  the controller.

   .. rubric:: Notes

   The input and output ports of a controller are derived from its modules. A module input
   wired to ``__call__`` becomes an input port of the controller, and a module output named
   in ``outputs`` becomes an output port.

   Beyond ports, a module may declare an effect: a value written onto a property of another
   module after the step, which is how a plasticity rule writes weights back onto a synapse.

   .. seealso::

      :py:obj:`Neuron`
          Controller whose modules step in dependency order within one timestep.

      :py:obj:`Brain`
          Controller whose modules read the previous timestep from a cache.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:attribute:: default_config
      :type:  type[ConfigT]


   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:



   .. py:method:: get_properties()
      :classmethod:


      Returns all the attributes names wrapped by the spark_property wrapper.



   .. py:method:: get_readonly_properties()
      :classmethod:


      Returns all the attributes names wrapped by the spark_property wrapper that do not define a setter.



   .. py:method:: recurrent_contract()

      Returns expected-like outputs and properties of the module.

      This function is a binding contract that allows the modules to accept self connections.



   .. py:method:: has_recurrent_contract()
      :classmethod:


      Returns True if the modules defines a recurrent contract, False otherwise.



   .. py:method:: get_config_spec()
      :classmethod:


      Returns the default configuration class associated with this module.



   .. py:method:: build(**abc_args)


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


      Advances every module one step.

      :param \*\*inputs: One entry per input port of the controller, as derived from the modules.
      :type \*\*inputs: SparkPayload

      :returns: One entry per output port of the controller, as derived from the modules.
      :rtype: dict of str to SparkPayload



   .. py:method:: read_state(port_list)
      :abstractmethod:


      Utility function to read internal controller's variables.



   .. py:method:: get_rng_keys(num_keys)

      Generates a new collection of random keys for the JAX's random engine.



.. py:class:: ControllerConfig

   Bases: :py:obj:`spark.core.config.SparkConfig`


   Base configuration for controllers.

   :param modules_specs: The modules the controller holds and how their ports are wired. Each entry names a
                         module, its class, its configuration, and where each of its inputs comes from.
   :type modules_specs: tuple of ModuleSpecs
   :param seed: Seed for the random draws of the controller and its modules. Drawn from the operating
                system when omitted.
   :type seed: int, optional
   :param dt: Integration step, in ms.
   :type dt: float, default 1.0

   .. rubric:: Notes

   ``dt`` is handed down to every configuration the controller contains, so the modules of
   one controller always integrate on the same clock. A ``dt`` set on a module directly is
   overwritten.


   .. py:attribute:: modules_specs
      :type:  tuple[spark.core.specs.ModuleSpecs, ...]


   .. py:attribute:: seed
      :type:  int


   .. py:attribute:: dt
      :type:  float


   .. py:method:: __post_init__()


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


