spark.core.config
=================

.. py:module:: spark.core.config


Attributes
----------

.. autoapisummary::

   spark.core.config.logger
   spark.core.config.METADATA_TEMPLATE
   spark.core.config.IMMUTABLE_TYPES


Classes
-------

.. autoapisummary::

   spark.core.config.InitializableFieldMetaclass
   spark.core.config.InitializableField
   spark.core.config.SparkMetaConfig
   spark.core.config.BaseSparkConfig
   spark.core.config.SparkConfig


Module Contents
---------------

.. py:data:: logger

.. py:data:: METADATA_TEMPLATE

.. py:data:: IMMUTABLE_TYPES

.. py:class:: InitializableFieldMetaclass

   Bases: :py:obj:`type`


   Metaclass that automatically injects common methods into the class.


.. py:class:: InitializableField(obj)

   Wrapper for fields that allow for Initializers | InitializersConfig to define the init() method.
   The method init() is extensively used through Spark modules to initialize variables either from default
   values or from full fledge initializers.


   .. py:attribute:: __obj__
      :type:  Any


   .. py:method:: __getattr__(name)


   .. py:method:: __setattr__(name, value)


   .. py:method:: __repr__()


   .. py:method:: __str__()


   .. py:method:: init(init_kwargs={}, key = None, shape = None, dtype = None, **kwargs)


.. py:class:: SparkMetaConfig

   Bases: :py:obj:`abc.ABCMeta`


   Metaclass that promotes class attributes to dataclass fields


   .. py:method:: map_common_init_patterns(factory = dc.MISSING)


.. py:class:: BaseSparkConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`abc.ABC`


   Base class for module configuration.


   .. py:attribute:: __config_delimiter__
      :type:  str
      :value: '__'



   .. py:attribute:: __shared_config_delimiter__
      :type:  str
      :value: '_s_'



   .. py:attribute:: __metadata__
      :type:  dict


   .. py:attribute:: __graph_editor_metadata__
      :type:  dict


   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:



   .. py:method:: __eq__(other)


   .. py:method:: merge(partial = {}, __skip_validation__ = False)

      Update config with partial overrides.



   .. py:method:: diff(other)

      Return differences from another config.



   .. py:method:: validate(is_partial = False, errors = None, current_path = ['main'])

      Validates all fields in the configuration class.



   .. py:method:: get_field_errors(field_name)

      Validates all fields in the configuration class.



   .. py:method:: get_metadata()

      Returns all the metadata in the configuration class, indexed by the attribute name.



   .. py:property:: class_ref
      :type: type


      Returns the type of the associated Module/Initializer.

      NOTE: It is recommended to set the __class_ref__ to the name of the associated module/initializer
      when defining custom configuration classes. The automatic class_ref solver is extremely brittle and
      likely to fail in many different custom scenarios.


   .. py:method:: __post_init__()


   .. py:method:: to_dict(is_partial = False)

      Serialize config to dictionary



   .. py:method:: get_kwargs()

      Returns a dictionary with pairs of key, value fields (skips metadata).



   .. py:method:: from_dict(dct)
      :classmethod:


      Create config instance from dictionary.



   .. py:method:: to_file(file_path, is_partial = False)

      Export a config instance from a .scfg file.



   .. py:method:: from_file(file_path, is_partial = False)
      :classmethod:


      Create config instance from a .scfg file.



   .. py:method:: __iter__()

      Custom iterator to simplify SparkConfig inspection across the entire ecosystem.
      This iterator excludes private fields.

      Output:
          field_name: str, field name
          field_value: tp.Any, field value



   .. py:method:: __repr__()


   .. py:method:: inspect(simplified=False)

      Returns a formated string of the datastructure.



   .. py:method:: with_new_seeds(seed=None)

      Utility method to recompute all seed variables within the SparkConfig.
      Useful when creating several populations from the same config.



.. py:class:: SparkConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`BaseSparkConfig`


   Default class for module configuration.


   .. py:attribute:: seed
      :type:  int


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


   .. py:attribute:: dt
      :type:  float


