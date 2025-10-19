spark.core.utils
================

.. py:module:: spark.core.utils


Classes
-------

.. autoapisummary::

   spark.core.utils.InheritanceFlags
   spark.core.utils.InheritanceLeaf
   spark.core.utils.InheritanceTree


Functions
---------

.. autoapisummary::

   spark.core.utils.normalize_str
   spark.core.utils.to_human_readable
   spark.core.utils.get_einsum_labels
   spark.core.utils.get_axes_einsum_labels
   spark.core.utils.validate_shape
   spark.core.utils.validate_list_shape
   spark.core.utils.is_shape
   spark.core.utils.is_list_shape
   spark.core.utils.is_dict_of
   spark.core.utils.is_list_of
   spark.core.utils.is_dtype
   spark.core.utils.is_float
   spark.core.utils.ascii_tree


Module Contents
---------------

.. py:function:: normalize_str(s)

   Converts any string into a consistent lowercase_snake_case format.

   :param s: str, string to normalize

   :returns: str, normalized string


.. py:function:: to_human_readable(s, capitalize_all = False)

   Converts a string from various programming cases into a human-readable format.

   Input:
       s: str, string to normalize

   Output:
       str, human readable string


.. py:function:: get_einsum_labels(num_dims, offset = 0)

   Generates labels for a generalized dot product using Einstein notation.

   :param num_dims: int, number of dimensions (labels) to generate
   :param offset: int, initial dimension (label) offset

   :returns: str, a string with num_dims different labels, skipping the first offset characters


.. py:function:: get_axes_einsum_labels(axes, ignore_repeated = False)

   Generates labels for a generalized dot product using Einstein notation.

   :param axes: tuple[int, ...], requested dimensions (labels) to generate

   :returns: str, a string with num_dims different labels, skipping the first offset characters


.. py:function:: validate_shape(obj)

   Verifies that the object is broadcastable to a valid shape (tuple of integers).
   Returns the shape.

   :param obj: tp.Any: the instance to validate

   :returns: list[tuple[int, ...]], the shape


.. py:function:: validate_list_shape(obj)

   Verifies that the object is broadcastable to a valid list ofshape (a list of tuple of integers).
   Returns the list of shapes.

   :param obj: tp.Any: the instance to validate

   :returns: list[tuple[int, ...]], the list of shapes


.. py:function:: is_shape(obj)

   Checks if the obj is broadcastable to a shape.

   :param obj: tp.Any: the instance to check.

   :returns: bool, True if the object is broadcastable to a shape, False otherwise.


.. py:function:: is_list_shape(obj)

   Checks if the obj is broadcastable to a shape.

   :param obj: tp.Any: the instance to check.

   :returns: bool, True if the object is broadcastable to a list of shapes, False otherwise.


.. py:function:: is_dict_of(obj, value_cls, key_cls = str)

   Check if an object instance is of 'dict[key_cls, value_cls]'.

   :param obj: tp.Any: the instance to check.
   :param key_cls: type[tp.Any], the class to compare keys against.
   :param value_cls: type[tp.Any], the class to compare values against.

   :returns: bool, True if the object is an instance of 'dict[key_cls, value_cls]', False otherwise.


.. py:function:: is_list_of(obj, cls)

   Check if an object instance is of 'list[cls]'.

   :param obj: tp.Any, the instance to check.
   :param cls: type[tp.Any], the class to compare values against.

   :returns: bool, True if the object is an instance of 'list[cls]', False otherwise.


.. py:function:: is_dtype(obj)

   Check if an object is a 'DTypeLike'.

   :param obj: The instance to check.
   :type obj: tp.Any

   :returns: bool, True if the object is a 'DTypeLike', False otherwise.


.. py:function:: is_float(obj)

   Check if an object is a 'DTypeLike'.

   :param obj: The instance to check.
   :type obj: tp.Any

   :returns: bool, True if the object is a 'DTypeLike', False otherwise.


.. py:function:: ascii_tree(text)

   Build an ASCII tree from indentation-based text.
   Each level is inferred from leading spaces.


.. py:class:: InheritanceFlags

   Bases: :py:obj:`enum.IntFlag`


   Support for integer-based Flags

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: CAN_INHERIT
      :value: 8



   .. py:attribute:: IS_INHERITING
      :value: 4



   .. py:attribute:: CAN_RECEIVE
      :value: 2



   .. py:attribute:: IS_RECEIVING
      :value: 1



.. py:class:: InheritanceLeaf

   Leaf object for the InheritanceTree data structure.


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: type_string
      :type:  str


   .. py:attribute:: inheritance_childs
      :type:  list[list[str]]


   .. py:attribute:: flags
      :type:  InheritanceFlags
      :value: 0



   .. py:attribute:: break_inheritance
      :type:  bool
      :value: False



   .. py:attribute:: parent
      :type:  InheritanceTree
      :value: None



   .. py:method:: __repr__()


   .. py:method:: to_dict()


   .. py:method:: from_dict(d)
      :classmethod:



   .. py:method:: can_inherit()

      Cheks the leaf node can inherit.



   .. py:method:: is_inheriting()

      Cheks the leaf node is inheriting.



   .. py:method:: can_receive()

      Cheks the leaf node can receive.



   .. py:method:: is_receiving()

      Cheks the leaf node is receiving.



.. py:class:: InheritanceTree(path = [])

   Tree-like data structure to manage the inheritance status of variables in the Spark Graph Editor.

   This data structure is used to link variables with the same names and types for simultaneous updates within the GUI.


   .. py:method:: __repr__()


   .. py:method:: add_leaf(path, type_string = '', inheritance_childs = [], flags = 0, break_inheritance = False, **kwargs)

      Adds a new leaf to the tree.

      Input:
          path: list[str], path to the new leaf node, with the last entry the name of the leaf
          type_string: str, string representation of the types this variable manages
          inheritance_childs: list[list[str]]=[], list of children that can inherit from this variable (Note: do not set by hand)
          flags: InheritanceFlags, 4-bit flags that represent inheritance possibilities (Note: do not set by hand)
          break_inheritance: bool, boolean flag to disconnect this variable from the inheritance dynamics



   .. py:method:: add_branch(path)

      Adds a new branch to the tree.

      Input:
          path: list[str], path to the new branch, with the last entry the name of the branch



   .. py:method:: validate(inheriting_labels = {})

      Validates the flags and the inheritance childs of the tree.



   .. py:method:: get_leaf(path)

      Returns the status of the leaf node.

      Input:
          path: list[str], path to the leaf node, with the last entry the name of the leaf

      :returns: InheritanceLeaf, returns the leaf node instance.



   .. py:method:: to_dict()

      InheritanceTree dict serializer.



   .. py:method:: from_dict(d, path = [])
      :classmethod:


      InheritanceTree dict deserializer.



