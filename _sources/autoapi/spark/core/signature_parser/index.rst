spark.core.signature_parser
===========================

.. py:module:: spark.core.signature_parser


Functions
---------

.. autoapisummary::

   spark.core.signature_parser.normalize_typehint
   spark.core.signature_parser.is_instance
   spark.core.signature_parser.get_input_specs
   spark.core.signature_parser.get_optional_input_names
   spark.core.signature_parser.get_output_specs
   spark.core.signature_parser.get_property_specs


Module Contents
---------------

.. py:function:: normalize_typehint(t)

   Expands a type hint into a tuple of concrete non-union types.

   :param hint: Type hint to expand.
   :type hint: Any

   :returns: The branches of the hint, with unions and aliases resolved.
   :rtype: tuple of type


.. py:function:: is_instance(value, types)

.. py:function:: get_input_specs(module)

   Reads the input port specifications off a module class.

   :param module: Class to inspect.
   :type module: type of SparkModule

   :returns: One entry per parameter of ``__call__``, excluding ``self``.
   :rtype: dict of str to PortSpecs

   :raises TypeError: If a parameter is annotated with something that is not a `SparkPayload`.


.. py:function:: get_optional_input_names(module)

   Reads the names of the optional input ports off a module class.

   An input is optional when its annotation accepts None. It may be left unconnected, and the
   module falls back to its own default.

   :param module: Class to inspect.
   :type module: type of SparkModule

   :rtype: list of str


.. py:function:: get_output_specs(module)

   Reads the output port specifications off a module class.

   :param module: Class to inspect.
   :type module: type of SparkModule

   :returns: One entry per member of the TypedDict ``__call__`` returns.
   :rtype: dict of str to PortSpecs

   :raises TypeError: If ``__call__`` has no TypedDict return annotation, or if a member is annotated with
       something that is not a `SparkPayload`.


.. py:function:: get_property_specs(module)

   Reads the property port specifications off a module class.

   :param module: Class to inspect.
   :type module: type of SparkModule

   :returns: One entry per `spark_property`.
   :rtype: dict of str to PortSpecs

   :raises SyntaxError: If a property getter has no return annotation, or annotates a union.
   :raises TypeError: If a getter is annotated with something that is not a `SparkPayload`.


