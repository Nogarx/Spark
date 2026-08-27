spark.core.config_validation
============================

.. py:module:: spark.core.config_validation


Classes
-------

.. autoapisummary::

   spark.core.config_validation.NoValidation
   spark.core.config_validation.ConfigurationValidator
   spark.core.config_validation.TypeValidator
   spark.core.config_validation.PositiveValidator
   spark.core.config_validation.BinaryValidator
   spark.core.config_validation.ZeroOneValidator


Module Contents
---------------

.. py:class:: NoValidation

   Context manager suspending the field validators.

   For building a configuration out of values that are not valid on their own yet, such as a
   half-finished model in the graph editor.

   .. rubric:: Examples

   >>> with NoValidation():
   ...     config = LeakySomaConfig(potential_tau=None)


   .. py:method:: __enter__()


   .. py:method:: __exit__(*exception)


.. py:class:: ConfigurationValidator(field, valid_types = None)

   Base class for the validators of a configuration field.

   :param field: Field being guarded.
   :type field: dataclasses.Field
   :param valid_types: Types the field accepts. Read from the field metadata when omitted.
   :type valid_types: tuple of type, optional


   .. py:attribute:: field


   .. py:attribute:: valid_types
      :value: None



   .. py:method:: validate(value)
      :abstractmethod:



.. py:class:: TypeValidator(field, valid_types = None)

   Bases: :py:obj:`ConfigurationValidator`


   Checks the value against the types declared by the field.

   The types come from the ``valid_types`` metadata entry, which the metaclass fills in from
   the annotation.


   .. py:method:: validate(value)


.. py:class:: PositiveValidator(field, valid_types = None)

   Bases: :py:obj:`ConfigurationValidator`


   Checks that every entry of the value is greater than zero.


   .. py:method:: validate(value)


.. py:class:: BinaryValidator(field, valid_types = None)

   Bases: :py:obj:`ConfigurationValidator`


   Checks that every entry of the value is 0 or 1.


   .. py:method:: validate(value)


.. py:class:: ZeroOneValidator(field, valid_types = None)

   Bases: :py:obj:`ConfigurationValidator`


   Checks that every entry of the value lies in ``[0, 1]``.


   .. py:method:: validate(value)


