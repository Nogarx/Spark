spark.graph_editor.models.config_types
======================================

.. py:module:: spark.graph_editor.models.config_types


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.config_types.FieldKind


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.config_types.type_tokens
   spark.graph_editor.models.config_types.classify
   spark.graph_editor.models.config_types.is_optional
   spark.graph_editor.models.config_types.accepts_array
   spark.graph_editor.models.config_types.accepts_initializer
   spark.graph_editor.models.config_types.initializer_policy
   spark.graph_editor.models.config_types.type_label
   spark.graph_editor.models.config_types.dtype_key
   spark.graph_editor.models.config_types.values_equal
   spark.graph_editor.models.config_types.coerce


Module Contents
---------------

.. py:class:: FieldKind(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Editor-level classification of a configuration field.


   .. py:attribute:: BOOL


   .. py:attribute:: INT


   .. py:attribute:: FLOAT


   .. py:attribute:: STR


   .. py:attribute:: DTYPE


   .. py:attribute:: INT_TUPLE


   .. py:attribute:: FLOAT_TUPLE


   .. py:attribute:: ARRAY


   .. py:attribute:: MODULE_SPECS


   .. py:attribute:: UNKNOWN


.. py:function:: type_tokens(type_hint = None, valid_types = None)

   Builds the normalized token set describing a configuration field.

   :param type_hint: The dataclass field annotation. May be a string.
   :type type_hint: Any, optional
   :param valid_types: The "valid_types" entry of the field metadata.
   :type valid_types: Any, optional

   :returns: The normalized type tokens.
   :rtype: frozenset of str


.. py:function:: classify(type_hint = None, valid_types = None, tokens = None)

   Maps a configuration field to the editor widget family that can safely edit it.


.. py:function:: is_optional(tokens)

   Returns True if the field explicitly accepts None.


.. py:function:: accepts_array(tokens)

   Returns True if the field accepts a raw array value.


.. py:function:: accepts_initializer(tokens)

   Returns True if the field explicitly accepts an Initializer.


.. py:function:: initializer_policy(tokens, metadata = None)

   Decides whether a field may (and must) be defined through an initializer.

   :param tokens: Normalized type tokens of the field.
   :type tokens: frozenset of str
   :param metadata: Field metadata.
   :type metadata: dict, optional

   :returns: * **allowed** (*bool*) -- True if the field accepts an initializer.
             * **mandatory** (*bool*) -- True if the field can only be defined through an initializer.


.. py:function:: type_label(tokens, type_hint = None)

   Human readable representation of a field type, used for tooltips.


.. py:function:: dtype_key(value)

   Canonical name of a dtype-like value.


.. py:function:: values_equal(first, second)

   Comparison that tolerates arrays, dtypes and objects with an ambiguous __eq__.


.. py:function:: coerce(kind, value)

   Casts a widget value to the python type expected by the configuration field.


