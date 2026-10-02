spark.graph_editor.models.graph_model
=====================================

.. py:module:: spark.graph_editor.models.graph_model


Attributes
----------

.. autoapisummary::

   spark.graph_editor.models.graph_model.logger


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.graph_model.GraphModel


Module Contents
---------------

.. py:data:: logger

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


