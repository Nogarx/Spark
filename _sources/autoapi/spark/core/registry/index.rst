spark.core.registry
===================

.. py:module:: spark.core.registry


Attributes
----------

.. autoapisummary::

   spark.core.registry.REGISTRY
   spark.core.registry.register_module
   spark.core.registry.register_payload
   spark.core.registry.register_initializer
   spark.core.registry.register_config
   spark.core.registry.register_cfg_validator
   spark.core.registry.MRO_PATH_ALIAS_MAP
   spark.core.registry.INITIALIZERS_ALIAS_MAP


Classes
-------

.. autoapisummary::

   spark.core.registry.RegistryEntry
   spark.core.registry.SubRegistry
   spark.core.registry.Registry


Functions
---------

.. autoapisummary::

   spark.core.registry.create_registry_decorator


Module Contents
---------------

.. py:class:: RegistryEntry

   Structured entry for the registry.


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: class_ref
      :type:  type


   .. py:attribute:: path
      :type:  list[str]


.. py:class:: SubRegistry(registry_base_type)

   Bases: :py:obj:`collections.abc.Mapping`


   Registry for registry_base_type.


   .. py:attribute:: __built__
      :value: False



   .. py:method:: __getitem__(key)


   .. py:method:: __iter__()


   .. py:method:: __len__()


   .. py:method:: items()

      D.items() -> a set-like object providing a view on D's items



   .. py:method:: register(name, cls, path = None)

      Register new registry_base_type.



   .. py:method:: get(name, default = None)

      Safely retrieves a component entry by name.



   .. py:method:: get_by_cls(cls)

      Safely retrieves a component entry by name.



   .. py:property:: is_finalized
      :type: bool



.. py:class:: Registry

   Registry object.


   .. py:attribute:: MODULES


   .. py:attribute:: PAYLOADS


   .. py:attribute:: INITIALIZERS


   .. py:attribute:: CONFIG


   .. py:attribute:: CFG_VALIDATORS


.. py:data:: REGISTRY

   Registry singleton.

.. py:function:: create_registry_decorator(sub_registry, base_class_name, base_class_path, base_class_abr = None)

.. py:data:: register_module

   Decorator used to register a new SparkModule.
   Note that module must inherit from spark.nn.Module (spark.core.module.SparkModule)

.. py:data:: register_payload

   Decorator used to register a new SparkPayload.
   Note that module must inherit from spark.SparkPayload (spark.core.payloads.SparkPayload)

.. py:data:: register_initializer

   Decorator used to register a new Initializer.
   Note that module must inherit from spark.nn.initializers.base.Initializer

.. py:data:: register_config

   Decorator used to register a new SparkConfig.
   Note that module must inherit from spark.nn.BaseConfig (spark.core.config.BaseSparkConfig)

.. py:data:: register_cfg_validator

   Decorator used to register a new ConfigurationValidator.
   Note that module must inherit from spark.core.config_validation.ConfigurationValidator

.. py:data:: MRO_PATH_ALIAS_MAP

.. py:data:: INITIALIZERS_ALIAS_MAP

