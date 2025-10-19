inspector_panel
===============

.. py:module:: inspector_panel


Classes
-------

.. autoapisummary::

   inspector_panel.InspectorPanel
   inspector_panel.QNodeConfig
   inspector_panel.QNodeIO
   inspector_panel.InspectorIdleWidget
   inspector_panel.NodeHeaderWidget
   inspector_panel.NodeNameWidget
   inspector_panel.TreeDisplay


Module Contents
---------------

.. py:class:: InspectorPanel(name = 'Inspector', parent = None, **kwargs)

   Bases: :py:obj:`spark.graph_editor.widgets.dock_panel.QDockPanel`


   Dockable panel to show node configurations.


   .. py:attribute:: broadcast_message


   .. py:method:: on_selection_update(new_selection, previous_selection)

      Event handler for graph selections.



   .. py:method:: set_node(node)

      Sets the node to be inspected. If the node is None, it clears the inspector.

      Input:
          node: The node to inspect, or None to clear.



   .. py:method:: on_error_message(message)


.. py:class:: QNodeConfig(node, parent = None, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Constructs the UI to modify the SparkConfig associated with the a node.


   .. py:attribute:: error_detected


   .. py:attribute:: content


   .. py:method:: addWidget(widget)

      Add a widget to the central content widget's layout.



   .. py:method:: name_update(node, name)


   .. py:method:: on_inheritance_toggle(value, leaf, path)


.. py:class:: QNodeIO(node, parent = None, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Constructs the UI to modify the SparkConfig associated with the a inputs and outpus.


   .. py:attribute:: error_detected


   .. py:attribute:: content


   .. py:method:: addWidget(widget)

      Add a widget to the central content widget's layout.



   .. py:method:: name_update(node, name)


.. py:class:: InspectorIdleWidget(message, parent = None, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Constructs the UI shown when no valid node is selected.


.. py:class:: NodeHeaderWidget(name, node_cls, config_tree = None, parent = None, **kwargs)

   Bases: :py:obj:`spark.graph_editor.widgets.base.SparkQWidget`


   QWidget used for the name of nodes in the SparkGraphEditor's Inspector.


   .. py:attribute:: name_widget


   .. py:attribute:: class_label


   .. py:method:: sizeHint()


   .. py:method:: get_value()

      Returns the widget value.



   .. py:method:: set_value(value)

      Returns the widget value.



.. py:class:: NodeNameWidget(name, parent = None, **kwargs)

   Bases: :py:obj:`spark.graph_editor.widgets.base.SparkQWidget`


   QWidget used for the name of nodes in the SparkGraphEditor's Inspector.


   .. py:method:: sizeHint()


   .. py:method:: get_value()

      Returns the widget value.



   .. py:method:: set_value(value)

      Returns the widget value.



.. py:class:: TreeDisplay(tree, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QPlainTextEdit`


   .. py:method:: sizeHint()


