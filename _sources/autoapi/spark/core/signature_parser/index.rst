spark.core.signature_parser
===========================

.. py:module:: spark.core.signature_parser


Functions
---------

.. autoapisummary::

   spark.core.signature_parser.normalize_typehint
   spark.core.signature_parser.is_instance
   spark.core.signature_parser.get_input_specs
   spark.core.signature_parser.get_output_specs


Module Contents
---------------

.. py:function:: normalize_typehint(t)

   Produce a tuple of fully expanded, non-union, non-merged type variants.


.. py:function:: is_instance(value, types)

.. py:function:: get_input_specs(module)

   Returns a dictionary of the SparkModule's input port specifications.


.. py:function:: get_output_specs(module)

   Returns a dictionary of the SparkModule's input port specifications.


