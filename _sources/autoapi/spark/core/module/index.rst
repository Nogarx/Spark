spark.core.module
=================

.. py:module:: spark.core.module


Attributes
----------

.. autoapisummary::

   spark.core.module.ConfigT
   spark.core.module.InputT


Classes
-------

.. autoapisummary::

   spark.core.module.ModuleOutput
   spark.core.module.SparkMeta
   spark.core.module.SparkModule


Module Contents
---------------

.. py:data:: ConfigT

.. py:data:: InputT

.. py:class:: ModuleOutput

   Bases: :py:obj:`TypedDict`


   Base class for the output ports of a module.

   A module declares its output ports by annotating the return of its ``__call__`` with a
   TypedDict. The names and payload types of that TypedDict become the output specification.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: SparkMeta

   Bases: :py:obj:`spark.core.backend.ModuleMeta`


   Metaclass for `SparkModule`.

   Wraps ``__call__`` so that the first call builds the module. Shapes are not known until
   values arrive, so a module is constructed from its configuration alone and completes on
   the first call.


.. py:class:: SparkModule(*, config = None, name = None, **kwargs)

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



