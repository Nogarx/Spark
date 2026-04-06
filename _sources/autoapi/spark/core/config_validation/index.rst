spark.core.config_validation
============================

.. py:module:: spark.core.config_validation


Attributes
----------

.. autoapisummary::

   spark.core.config_validation.DEFAULT_TYPE_PROMOTIONS


Classes
-------

.. autoapisummary::

   spark.core.config_validation.ConfigurationValidator
   spark.core.config_validation.TypeValidator
   spark.core.config_validation.PositiveValidator
   spark.core.config_validation.BinaryValidator
   spark.core.config_validation.ZeroOneValidator


Module Contents
---------------

.. py:data:: DEFAULT_TYPE_PROMOTIONS

.. py:class:: ConfigurationValidator(field)

   Base class for validators for the fields in a SparkConfig.


   .. py:attribute:: field


   .. py:method:: validate(value)
      :abstractmethod:



.. py:class:: TypeValidator(field)

   Bases: :py:obj:`ConfigurationValidator`


   Validates the type of the field against a set of valid_types defined in the metadata.


   .. py:method:: validate(value)


.. py:class:: PositiveValidator(field)

   Bases: :py:obj:`ConfigurationValidator`


   Validates that the value(s) of the attribute are greater than zero.


   .. py:method:: validate(value)


.. py:class:: BinaryValidator(field)

   Bases: :py:obj:`ConfigurationValidator`


   Validates that the value(s) of the attribute are in the set {0,1}.


   .. py:method:: validate(value)


.. py:class:: ZeroOneValidator(field)

   Bases: :py:obj:`ConfigurationValidator`


   Validates that the value(s) of the attribute are in the range [0,1].


   .. py:method:: validate(value)


