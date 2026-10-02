spark.core.registry
===================

.. py:module:: spark.core.registry


Attributes
----------

.. autoapisummary::

   spark.core.registry.logger
   spark.core.registry.REGISTRY
   spark.core.registry.register_module
   spark.core.registry.register_neuron
   spark.core.registry.register_payload
   spark.core.registry.register_interface
   spark.core.registry.register_initializer
   spark.core.registry.register_config
   spark.core.registry.register_cfg_validator
   spark.core.registry.MRO_PATH_ALIAS_MAP
   spark.core.registry.INITIALIZERS_ALIAS_MAP


Classes
-------

.. autoapisummary::

   spark.core.registry.RegistryNamespace
   spark.core.registry.RegistryEntry
   spark.core.registry.SubRegistry
   spark.core.registry.Registry
   spark.core.registry.SparkRegistry


Functions
---------

.. autoapisummary::

   spark.core.registry.create_registry_decorator
   spark.core.registry.register_neuron_from_config
   spark.core.registry.register_models_from_payload
   spark.core.registry.register_neuron_from_config_file


Module Contents
---------------

.. py:data:: logger

.. py:class:: RegistryNamespace(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   The namespaces a class can be registered under.

   Each names one kind of registered class and the base class its members must derive from.


   .. py:attribute:: Components


   .. py:attribute:: Initializers


   .. py:attribute:: Payloads


   .. py:attribute:: Interfaces


   .. py:attribute:: Neurons


   .. py:attribute:: Configs


   .. py:attribute:: Validators


   .. py:method:: base_module(namespace)
      :classmethod:



.. py:class:: RegistryEntry

   Registered class.

   :param name: Normalized name the class answers to.
   :type name: str
   :param path: Category path, used to group the class in the palette of the editor.
   :type path: tuple of str
   :param cls: The registered class, or a thunk resolving to it.
   :type cls: type


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: module
      :type:  str


   .. py:attribute:: qualname
      :type:  str


   .. py:attribute:: namespace
      :type:  RegistryNamespace


   .. py:attribute:: path
      :type:  list[str]


   .. py:attribute:: metadata
      :type:  dict[str, Any]


   .. py:method:: get_cls()


.. py:class:: SubRegistry(instance, namespace)

   One namespace of a registry, as a mapping of name to entry.

   Reached as ``REGISTRY.<Namespace>``, for example ``REGISTRY.Components``.


   .. py:method:: get(key, default = None)

      Returns the entry registered under a name.

      :param key: Name to look up. Normalized before the lookup.
      :type key: str or None
      :param default: Returned when the name is not registered.
      :type default: Any, optional

      :rtype: RegistryEntry or None



   .. py:method:: get_by_cls(cls, default = None)

      Returns the entry a class is registered under.

      :param cls: Registered class.
      :type cls: type
      :param default: Returned when the class is not registered.
      :type default: Any, optional

      :rtype: RegistryEntry or None



   .. py:method:: __getitem__(key)


   .. py:method:: __setitem__(key, value)


   .. py:method:: __contains__(key)


   .. py:method:: __iter__()


   .. py:method:: __len__()


   .. py:method:: values()


   .. py:method:: keys()


   .. py:method:: items()


   .. py:method:: register(name, cls, path = None)


   .. py:method:: exists(name)


.. py:class:: Registry

   Mapping from a name to the class registered under it.

   Entries are collected as modules are imported and resolved once, when the registry is
   built. Nothing may be read before that, since a class registered later would be missed.

   .. seealso::

      :py:obj:`SparkRegistry`
          The registry of the framework, with its namespaces.


   .. py:attribute:: __built__
      :value: False



   .. py:method:: __getattr__(name)

      Serves ``REGISTRY.<Namespace>`` as a view of that namespace.

      :param name: Name of a namespace.
      :type name: str

      :rtype: SubRegistry

      :raises AttributeError: If no namespace goes by that name.



   .. py:method:: __iter__()


   .. py:method:: __len__()


   .. py:method:: entries()


   .. py:method:: register(namespace, name, cls, path = None)

      Registers a class under a namespace.

      :param namespace: Namespace to register under.
      :type namespace: RegistryNamespace
      :param cls: Class to register.
      :type cls: type
      :param path: Category path, used to group the class in the palette of the editor.
      :type path: tuple of str, optional



   .. py:method:: get(namespace, name, default = None)

      Returns the entry registered under a name, across namespaces.

      :param key: Name to look up.
      :type key: str
      :param default: Returned when the name is not registered.
      :type default: Any, optional

      :rtype: RegistryEntry or None



   .. py:method:: exists(namespace, name)


   .. py:property:: is_built
      :type: bool



.. py:class:: SparkRegistry

   Bases: :py:obj:`Registry`


   The registry of the framework.

   Holds one namespace per kind of registered class: components, initializers, payloads,
   interfaces, neurons, configurations and validators. The singleton is `REGISTRY`.


   .. py:attribute:: Components
      :type:  SubRegistry


.. py:data:: REGISTRY

   Registry singleton.

.. py:function:: create_registry_decorator(namespace)

.. py:data:: register_module

   Decorator used to register a new SparkModule.
   Note that module must inherit from spark.nn.Module (spark.core.module.SparkModule)

.. py:data:: register_neuron

   Decorator used to register a new Neuron model.
   Note that module must inherit from spark.nn.Neuron (spark.nn.controllers.neuron.Neuron)

.. py:data:: register_payload

   Decorator used to register a new SparkPayload.
   Note that module must inherit from spark.SparkPayload (spark.core.payloads.SparkPayload)

.. py:data:: register_interface

   Decorator used to register a new Interface.
   Note that module must inherit from spark.nn.interfaces.base.Interface

.. py:data:: register_initializer

   Decorator used to register a new Initializer.
   Note that module must inherit from spark.nn.initializers.base.Initializer

.. py:data:: register_config

   Decorator used to register a new SparkConfig.
   Note that module must inherit from spark.nn.BaseConfig (spark.core.config.SparkConfig)

.. py:data:: register_cfg_validator

   Decorator used to register a new ConfigurationValidator.
   Note that module must inherit from spark.core.config_validation.ConfigurationValidator

.. py:data:: MRO_PATH_ALIAS_MAP

.. py:data:: INITIALIZERS_ALIAS_MAP

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


.. py:function:: register_models_from_payload(payload)

   Registers every model named by a decoded json document.

   Read before the spark decoder runs, so that a file naming models that are not yet
   registered can be decoded.

   :param payload: A decoded json document.
   :type payload: Any

   :returns: Names of the models that were registered.
   :rtype: list of str


.. py:function:: register_neuron_from_config_file(cls_name, path)

   Registers a (Neuron, NeuronConfig) pair built from a .scfg file.

   :param name: Name the pair answers to.
   :type name: str
   :param file_path: File holding the configuration.
   :type file_path: str

   :returns: The registered class.
   :rtype: type of Neuron


