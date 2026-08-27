spark.graph_editor.models.inspector_model
=========================================

.. py:module:: spark.graph_editor.models.inspector_model


Attributes
----------

.. autoapisummary::

   spark.graph_editor.models.inspector_model.logger


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.inspector_model.ConfigNode
   spark.graph_editor.models.inspector_model.ConfigValueNode
   spark.graph_editor.models.inspector_model.ConfigGroupNode
   spark.graph_editor.models.inspector_model.ConfigListNode


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.inspector_model.collect_field_errors
   spark.graph_editor.models.inspector_model.parse_object_to_state


Module Contents
---------------

.. py:data:: logger

.. py:class:: ConfigNode(name, parent = None)

   Bases: :py:obj:`PySide6.QtCore.QObject`


   Base class for all configuration nodes in the inspector state model.

   Observable tree structure the UI binds to, wrapping the nested dataclasses and ModuleSpecs.


   .. py:attribute:: value_changed


   .. py:attribute:: errors_changed


   .. py:attribute:: inheritance_changed


   .. py:attribute:: initializer_changed


   .. py:attribute:: name


   .. py:attribute:: metadata
      :type:  dict


   .. py:property:: errors
      :type: list[str]



   .. py:property:: is_inherited
      :type: bool



   .. py:property:: is_initializer_active
      :type: bool



   .. py:method:: to_python()
      :abstractmethod:


      Reconstructs the underlying Python object (dataclass, list, or primitive).



.. py:class:: ConfigValueNode(name, value, type_hint, parent = None, field = None)

   Bases: :py:obj:`ConfigNode`


   Represents a primitive or simple value (int, float, str, bool, enum, etc.).


   .. py:attribute:: type_hint


   .. py:attribute:: field
      :value: None



   .. py:property:: value
      :type: Any



   .. py:property:: is_required
      :type: bool


      True if the underlying configuration field defines neither a default nor a default factory.


   .. py:method:: revalidate()

      Recomputes the error list of this node from the value and the field validators.



   .. py:method:: to_python()

      Reconstructs the underlying Python object (dataclass, list, or primitive).



.. py:class:: ConfigGroupNode(name, class_ref, parent = None)

   Bases: :py:obj:`ConfigNode`


   Represents a nested object, typically a SparkConfig, or ModuleSpecs.


   .. py:attribute:: class_ref


   .. py:attribute:: children
      :type:  list[ConfigNode]
      :value: []



   .. py:method:: add_child(child)


   .. py:method:: to_python()

      Reconstructs the underlying Python object (dataclass, list, or primitive).



.. py:class:: ConfigListNode(name, item_type, parent = None)

   Bases: :py:obj:`ConfigNode`


   Represents a list of items, such as list[ModuleSpecs] or list[int].


   .. py:attribute:: item_type


   .. py:attribute:: children
      :type:  list[ConfigNode]
      :value: []



   .. py:method:: add_child(child)


   .. py:method:: to_python()

      Reconstructs the underlying Python object (dataclass, list, or primitive).



.. py:function:: collect_field_errors(name, value, metadata = None, field = None, is_required = False, type_hint = None)

   Runs the validators declared by a configuration field against a value.

   :param name: Field name, used to build the messages.
   :type name: str
   :param value: Current value of the field.
   :type value: Any
   :param metadata: Field metadata. Validators are read from the "validators" entry.
   :type metadata: dict, optional
   :param field: Originating dataclass field. Validators are constructed from it.
   :type field: dataclasses.Field, optional
   :param is_required: True if the field defines neither a default nor a default factory.
   :type is_required: bool, default False
   :param type_hint: Field annotation. Used to detect fields that accept None.
   :type type_hint: Any, optional

   :returns: The collected error messages.
   :rtype: list of str


.. py:function:: parse_object_to_state(name, obj, type_hint = None, metadata = None, parent = None, field = None)

   Parses a Python object (SparkConfig, dataclass, ModuleSpecs, list or primitive) into a ConfigNode tree.


