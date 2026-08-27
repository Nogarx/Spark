spark.graph_editor.models
=========================

.. py:module:: spark.graph_editor.models


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/graph_editor/models/base_model/index
   /autoapi/spark/graph_editor/models/compartment_model/index
   /autoapi/spark/graph_editor/models/config_types/index
   /autoapi/spark/graph_editor/models/controller_profile/index
   /autoapi/spark/graph_editor/models/edge_model/index
   /autoapi/spark/graph_editor/models/graph_export/index
   /autoapi/spark/graph_editor/models/graph_layout/index
   /autoapi/spark/graph_editor/models/graph_model/index
   /autoapi/spark/graph_editor/models/graph_registry/index
   /autoapi/spark/graph_editor/models/inheritance_tree/index
   /autoapi/spark/graph_editor/models/inspector_model/index
   /autoapi/spark/graph_editor/models/model_import/index
   /autoapi/spark/graph_editor/models/model_library/index
   /autoapi/spark/graph_editor/models/node_factory/index
   /autoapi/spark/graph_editor/models/node_model/index
   /autoapi/spark/graph_editor/models/port_model/index
   /autoapi/spark/graph_editor/models/recent_files/index
   /autoapi/spark/graph_editor/models/session_io/index


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.PortModel
   spark.graph_editor.models.CompartmentModel
   spark.graph_editor.models.NodeModel
   spark.graph_editor.models.SourceNodeModel
   spark.graph_editor.models.SinkNodeModel
   spark.graph_editor.models.InterfaceNodeModel
   spark.graph_editor.models.EdgeModel
   spark.graph_editor.models.GraphModel
   spark.graph_editor.models.ConfigNode
   spark.graph_editor.models.ConfigValueNode
   spark.graph_editor.models.ConfigGroupNode
   spark.graph_editor.models.ConfigListNode


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.parse_object_to_state


Package Contents
----------------

.. py:class:: PortModel(name, is_input, port_type, is_optional = False, multi_connection = False, parent=None)

   Bases: :py:obj:`spark.graph_editor.models.base_model.BaseModel`


   Base class for all graph models.


   .. py:attribute:: connected


   .. py:attribute:: disconnected


   .. py:attribute:: id
      :value: ''



   .. py:attribute:: name


   .. py:attribute:: is_input


   .. py:attribute:: is_optional
      :value: False



   .. py:attribute:: multi_connection
      :value: False



   .. py:attribute:: port_type


   .. py:attribute:: node
      :type:  spark.graph_editor.models.node_model.NodeModel | None
      :value: None



   .. py:attribute:: compartment
      :type:  spark.graph_editor.models.compartment_model.CompartmentModel | None
      :value: None



   .. py:attribute:: edges
      :type:  list[spark.graph_editor.models.edge_model.EdgeModel]
      :value: []



   .. py:method:: add_edge(edge)


   .. py:method:: remove_edge(edge)


   .. py:method:: get_connected_nodes()


   .. py:method:: to_dict()


   .. py:method:: from_dict(data)
      :classmethod:



.. py:class:: CompartmentModel(name, parent=None)

   Bases: :py:obj:`spark.graph_editor.models.base_model.BaseModel`


   Base class for all graph models.


   .. py:attribute:: port_added


   .. py:attribute:: port_removed


   .. py:attribute:: id
      :value: ''



   .. py:attribute:: name


   .. py:attribute:: ports
      :type:  list[spark.graph_editor.models.port_model.PortModel]
      :value: []



   .. py:attribute:: node
      :type:  spark.graph_editor.models.node_model.NodeModel | None
      :value: None



   .. py:method:: add_port(port)


   .. py:method:: remove_port(port)


   .. py:method:: get_port(name)


   .. py:method:: to_dict()


   .. py:method:: from_dict(data)
      :classmethod:



.. py:class:: NodeModel(name = None, type_name = 'BaseObject', pos=(0, 0), parent=None)

   Bases: :py:obj:`spark.graph_editor.models.base_model.BaseModel`


   Base class for all graph models.


   .. py:attribute:: position_changed


   .. py:attribute:: name_changed


   .. py:attribute:: type_changed


   .. py:attribute:: selected_changed


   .. py:attribute:: deleted


   .. py:attribute:: id
      :value: ''



   .. py:attribute:: config
      :type:  spark.core.config.SparkConfig | None
      :value: None



   .. py:attribute:: compartments
      :type:  list[spark.graph_editor.models.compartment_model.CompartmentModel]
      :value: []



   .. py:attribute:: call_section


   .. py:attribute:: props_section


   .. py:property:: name
      :type: str



   .. py:property:: type_name
      :type: str



   .. py:property:: pos
      :type: tuple[float, float]



   .. py:property:: is_selected
      :type: bool



   .. py:method:: add_compartment(compartment)


   .. py:method:: get_port_by_name(name, is_input = None)


   .. py:method:: get_all_ports()


   .. py:method:: delete()


   .. py:method:: to_dict()


   .. py:method:: from_dict(data)
      :classmethod:



.. py:class:: SourceNodeModel(name = None, type_name = 'Source Node', pos=(0, 0), parent=None)

   Bases: :py:obj:`NodeModel`


   Node standing for one input of the controller.


   .. py:attribute:: value_port


   .. py:method:: on_port_connected(edge)


.. py:class:: SinkNodeModel(name = None, type_name = 'Sink Node', pos=(0, 0), parent=None)

   Bases: :py:obj:`NodeModel`


   Node standing for one output of the controller.


   .. py:attribute:: value_port


   .. py:method:: on_port_connected(edge)


.. py:class:: InterfaceNodeModel(name = None, type_name = None, pos=(0, 0), parent=None)

   Bases: :py:obj:`NodeModel`


   Node model of an Interface.


   .. py:attribute:: config


.. py:class:: EdgeModel(source_port, target_port, parent=None)

   Bases: :py:obj:`spark.graph_editor.models.base_model.BaseModel`


   Base class for all graph models.


   .. py:attribute:: waypoints_changed


   .. py:attribute:: deleted


   .. py:attribute:: id
      :value: ''



   .. py:attribute:: source_port


   .. py:attribute:: target_port


   .. py:property:: waypoints
      :type: list[tuple]



   .. py:method:: validate_connection(src_port, dst_port)
      :classmethod:


      Returns True if a connection between two ports is allowed.



   .. py:method:: delete()


   .. py:method:: to_dict()


   .. py:method:: from_dict(data, all_ports)
      :classmethod:



.. py:class:: GraphModel(profile = None, parent=None)

   Bases: :py:obj:`spark.graph_editor.models.base_model.BaseModel`


   Base class for all graph models.


   .. py:attribute:: CONTROLLER_ID
      :value: '__controller__'



   .. py:attribute:: node_added


   .. py:attribute:: node_removed


   .. py:attribute:: edge_added


   .. py:attribute:: edge_removed


   .. py:attribute:: graph_cleared


   .. py:attribute:: inheritance_updated


   .. py:attribute:: profile_changed


   .. py:attribute:: config_value_changed


   .. py:attribute:: nodes
      :type:  list[spark.graph_editor.models.node_model.NodeModel]
      :value: []



   .. py:attribute:: edges
      :type:  list[spark.graph_editor.models.edge_model.EdgeModel]
      :value: []



   .. py:attribute:: undo_stack


   .. py:attribute:: inheritance_trees
      :type:  dict[str, spark.graph_editor.models.inheritance_tree.InheritanceTree]


   .. py:property:: profile
      :type: spark.graph_editor.models.controller_profile.ControllerProfile | None


      Controller profile this graph is building.


   .. py:method:: set_profile(profile, force = False)

      Sets the controller profile of the graph.

      The profile determines which modules may be placed and how the graph is exported. It can only
      change while the graph is empty, except when loading a model, where the file dictates it.

      :param profile: The new profile.
      :type profile: ControllerProfile or None
      :param force: Apply the profile regardless of the current content. Reserved for loading.
      :type force: bool, default False

      :returns: True if the profile was applied.
      :rtype: bool



   .. py:property:: controller_config
      :type: Any


      Configuration of the controller the graph describes, created on demand from the profile.


   .. py:method:: adopt_controller_config(config)

      Takes the controller settings of an existing configuration, and only those.

      :param config: Configuration to read the settings from.
      :type config: SparkConfig

      :returns: True if the settings were adopted.
      :rtype: bool

      .. rubric:: Notes

      The modules are dropped. The canvas is the single source of truth for what the controller contains.



   .. py:method:: can_change_profile()

      True if the controller profile may still be changed.



   .. py:method:: rebuild_inheritance_tree(*args)

      Rebuilds the inheritance tree of every node, preserving the currently cascading leaves.



   .. py:method:: get_inheritance_tree(node_id)

      Returns the inheritance tree of a node.



   .. py:method:: get_inheritance_leaf(path)

      Returns the inheritance leaf addressed by a full config path ([node_id, field, ...]).



   .. py:method:: get_inheritance_children(path)

      Returns the full config paths of every field driven by the leaf addressed by "path".



   .. py:method:: is_driven(path)

      Returns True if the field addressed by "path" currently receives its value from an ancestor.



   .. py:method:: get_node_config_value(path)


   .. py:method:: set_node_config_value(path, value, force = False)

      Writes a value into the configuration of a node.

      :param path: Full config path ([node_id, field, ...]).
      :type path: list of str
      :param value: The new value.
      :type value: Any
      :param force: Bypass the inheritance guard. Reserved for cascaded writes.
      :type force: bool, default False



   .. py:method:: update_inherited_value(origin_path, value)

      Propagates a value to every field driven by the leaf that owns "origin_path".



   .. py:method:: toggle_inheritance(path, is_inheriting)

      Enables/disables the cascade of the leaf addressed by "path".



   .. py:method:: add_node(node)


   .. py:method:: remove_node(node)


   .. py:method:: add_edge(edge)


   .. py:method:: remove_edge(edge)


   .. py:method:: get_node_by_id(node_id)


   .. py:method:: clear()


   .. py:method:: to_dict()


   .. py:method:: from_dict(data)


   .. py:method:: is_name_taken(name)


   .. py:method:: get_next_free_name(name)


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



.. py:function:: parse_object_to_state(name, obj, type_hint = None, metadata = None, parent = None, field = None)

   Parses a Python object (SparkConfig, dataclass, ModuleSpecs, list or primitive) into a ConfigNode tree.


