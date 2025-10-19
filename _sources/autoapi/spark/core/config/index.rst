spark.core.config
=================

.. py:module:: spark.core.config


Attributes
----------

.. autoapisummary::

   spark.core.config.METADATA_TEMPLATE


Classes
-------

.. autoapisummary::

   spark.core.config.SparkMetaConfig
   spark.core.config.BaseSparkConfig
   spark.core.config.SparkConfig


Module Contents
---------------

.. py:data:: METADATA_TEMPLATE

.. py:class:: SparkMetaConfig

   Bases: :py:obj:`abc.ABCMeta`


   Metaclass that promotes class attributes to dataclass fields


.. py:class:: BaseSparkConfig(**kwargs)

   Bases: :py:obj:`abc.ABC`


   Base class for module configuration.


   .. py:attribute:: __config_delimiter__
      :type:  str
      :value: '__'



   .. py:attribute:: __shared_config_delimiter__
      :type:  str
      :value: '_s_'



   .. py:attribute:: __graph_editor_metadata__
      :type:  dict


   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:



   .. py:method:: merge(partial = {})

      Update config with partial overrides.



   .. py:method:: diff(other)

      Return differences from another config.



   .. py:method:: validate()

      Validates all fields in the configuration class.



   .. py:method:: get_field_errors(field_name)

      Validates all fields in the configuration class.



   .. py:method:: get_metadata()

      Returns all the metadata in the configuration class, indexed by the attribute name.



   .. py:property:: class_ref
      :type: type


      Returns the type of the associated Module/Initializer.

      NOTE: It is recommended to set the __class_ref__ to the name of the associated module/initializer
      when defining custom configuration classes. The current class_ref solver is extremely brittle and
      likely to fail in many different custom scenarios.


   .. py:method:: __post_init__()


   .. py:method:: to_dict()

      Serialize config to dictionary



   .. py:method:: from_dict(dct)
      :classmethod:


      Create config instance from dictionary.



   .. py:method:: to_file(file_path)

      Export a config instance from a .scfg file.



   .. py:method:: from_file(file_path)
      :classmethod:


      Create config instance from a .scfg file.



   .. py:method:: __iter__()

      Custom iterator to simplify SparkConfig inspection across the entire ecosystem.
      This iterator excludes private fields.

      Output:
          field_name: str, field name
          field_value: tp.Any, field value



   .. py:method:: get_tree_structure()


.. py:class:: SparkConfig(**kwargs)

   Bases: :py:obj:`BaseSparkConfig`


   Default class for module configuration.


   .. py:attribute:: seed
      :type:  int


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


   .. py:attribute:: dt
      :type:  float


