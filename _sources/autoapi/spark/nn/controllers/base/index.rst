spark.nn.controllers.base
=========================

.. py:module:: spark.nn.controllers.base


Attributes
----------

.. autoapisummary::

   spark.nn.controllers.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.controllers.base.ControllerMeta
   spark.nn.controllers.base.ControllerConfig
   spark.nn.controllers.base.Controller


Module Contents
---------------

.. py:class:: ControllerMeta

   Bases: :py:obj:`spark.core.module.SparkMeta`


   Metaclass for controllers.


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


.. py:data:: ConfigT

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



