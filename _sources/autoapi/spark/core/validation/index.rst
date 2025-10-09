spark.core.validation
=====================

.. py:module:: spark.core.validation


Attributes
----------

.. autoapisummary::

   spark.core.validation.DEFAULT_SPARKMODULE_PATH
   spark.core.validation.DEFAULT_PAYLOAD_PATH
   spark.core.validation.DEFAULT_INITIALIZER_PATH
   spark.core.validation.DEFAULT_CONFIG_PATH
   spark.core.validation.DEFAULT_CFG_VALIDATOR_PATH


Functions
---------

.. autoapisummary::

   spark.core.validation.is_shape
   spark.core.validation.is_list_shape
   spark.core.validation.is_dict_of
   spark.core.validation.is_list_of
   spark.core.validation.is_dtype


Module Contents
---------------

.. py:data:: DEFAULT_SPARKMODULE_PATH
   :value: 'spark.core.module.SparkModule'


.. py:data:: DEFAULT_PAYLOAD_PATH
   :value: 'spark.core.payloads.SparkPayload'


.. py:data:: DEFAULT_INITIALIZER_PATH
   :value: 'spark.nn.initializers'


.. py:data:: DEFAULT_CONFIG_PATH
   :value: 'spark.core.config.BaseSparkConfig'


.. py:data:: DEFAULT_CFG_VALIDATOR_PATH
   :value: 'spark.core.config_validation.ConfigurationValidator'


.. py:function:: is_shape(obj)

   Check if an object instance is of 'Shape', i.e., 'tuple[int,...]'.

   :param obj: The instance to check.
   :type obj: tp.Any

   :returns: True if the object is an instance of 'Shape', False otherwise.
   :rtype: bool


.. py:function:: is_list_shape(obj)

   Check if an object instance is of 'list[Shape]', i.e., 'list[tuple[int,...]]'.

   :param obj: The instance to check.
   :type obj: tp.Any

   :returns: True if the object is an instance of 'Shape', False otherwise.
   :rtype: bool


.. py:function:: is_dict_of(obj, value_cls, key_cls = str)

   Check if an object instance is of 'dict[key_cls, value_cls]'.

   :param obj: The instance to check.
   :type obj: tp.Any

   :returns: True if the object is an instance of 'dict[key_cls, value_cls]', False otherwise.
   :rtype: bool


.. py:function:: is_list_of(obj, cls)

   Check if an object instance is of 'list[cls]'.

   :param obj: The instance to check.
   :type obj: tp.Any

   :returns: True if the object is an instance of 'list[cls]', False otherwise.
   :rtype: bool


.. py:function:: is_dtype(obj)

   Check if an object is a 'DTypeLike'.

   :param obj: The instance to check.
   :type obj: tp.Any

   :returns: True if the object is a 'DTypeLike', False otherwise.
   :rtype: bool


