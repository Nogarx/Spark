spark.graph_editor.models.inheritance_tree
==========================================

.. py:module:: spark.graph_editor.models.inheritance_tree


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.inheritance_tree.InheritanceFlags
   spark.graph_editor.models.inheritance_tree.InheritanceLeaf
   spark.graph_editor.models.inheritance_tree.InheritanceTree


Module Contents
---------------

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



   .. py:attribute:: type_key
      :type:  frozenset[str]


   .. py:method:: __post_init__()


   .. py:method:: __repr__()


   .. py:method:: to_dict()


   .. py:method:: from_dict(d)
      :classmethod:



   .. py:method:: can_inherit()

      Checks the leaf node can inherit.



   .. py:method:: is_inheriting()

      Checks the leaf node is inheriting.



   .. py:method:: can_receive()

      Checks the leaf node can receive.



   .. py:method:: is_receiving()

      Checks the leaf node is receiving.



   .. py:property:: path
      :type: list[str]


      Returns the path of the leaf node.


.. py:class:: InheritanceTree(path = [])

   Tree-like data structure holding the inheritance status of the variables of a node.

   Links variables of the same name and type, so that they are updated simultaneously.


   .. py:method:: __repr__()


   .. py:method:: add_leaf(path, type_string = '', inheritance_childs = [], flags = 0, break_inheritance = False, **kwargs)

      Adds a new leaf to the tree.

      :param path: Path to the new leaf, with the last entry the name of the leaf.
      :type path: list of str
      :param type_string: String representation of the types the variable accepts.
      :type type_string: str, optional
      :param inheritance_childs: Paths that can inherit from this variable. Computed by validate(), not set by hand.
      :type inheritance_childs: list of list of str, optional
      :param flags: Four bit flags holding the inheritance state. Computed by validate(), not set by hand.
      :type flags: InheritanceFlags, optional
      :param break_inheritance: Disconnect the variable from the inheritance dynamics.
      :type break_inheritance: bool, default False



   .. py:method:: add_branch(path)

      Adds a new branch to the tree.

      :param path: Path to the new branch, with the last entry the name of the branch.
      :type path: list of str



   .. py:method:: invalidate()

      Marks the whole subtree as invalid, forcing a full recomputation on the next validate() call.



   .. py:method:: validate(inheriting_labels = None)

      Validates the flags and the inheritance childs of the tree.

      :param inheriting_labels: {(leaf name, leaf type): is_inheriting} entries contributed by the ancestors of this subtree.
                                Leaves matching an entry are marked as receiving.
      :type inheriting_labels: dict, optional



   .. py:method:: get_leaf(path)

      Returns the leaf addressed by a path.

      :param path: Path to the leaf, with the last entry the name of the leaf.
      :type path: list of str

      :returns: The leaf the path addresses.
      :rtype: InheritanceLeaf

      :raises KeyError: If no leaf sits at the path.



   .. py:method:: get_subtree(path)

      Returns the subtree addressed by a path.

      :param path: Path to the subtree, with the last entry the name of the branch.
      :type path: list of str

      :returns: The branch the path addresses.
      :rtype: InheritanceTree

      :raises KeyError: If no branch sits at the path.



   .. py:method:: to_dict()

      InheritanceTree dict serializer.



   .. py:method:: from_dict(d, path = [])
      :classmethod:


      InheritanceTree dict deserializer.



   .. py:property:: path
      :type: list[str]


      Returns the path of the branch node.


