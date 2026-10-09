spark.graph_editor.commands.graph_commands
==========================================

.. py:module:: spark.graph_editor.commands.graph_commands


Attributes
----------

.. autoapisummary::

   spark.graph_editor.commands.graph_commands.logger


Classes
-------

.. autoapisummary::

   spark.graph_editor.commands.graph_commands.MoveNodeCommand
   spark.graph_editor.commands.graph_commands.AddNodeCommand
   spark.graph_editor.commands.graph_commands.RemoveNodeCommand
   spark.graph_editor.commands.graph_commands.AddEdgeCommand
   spark.graph_editor.commands.graph_commands.RemoveEdgeCommand
   spark.graph_editor.commands.graph_commands.ChangeEdgeWaypointsCommand
   spark.graph_editor.commands.graph_commands.RenameNodeCommand


Module Contents
---------------

.. py:data:: logger

.. py:class:: MoveNodeCommand(node_model, old_pos, new_pos, description = 'Move Node')

   Bases: :py:obj:`PySide6.QtGui.QUndoCommand`


   .. py:attribute:: node_model


   .. py:attribute:: old_pos


   .. py:attribute:: new_pos


   .. py:method:: undo()


   .. py:method:: redo()


.. py:class:: AddNodeCommand(graph_model, node_model, description = 'Add Node')

   Bases: :py:obj:`PySide6.QtGui.QUndoCommand`


   .. py:attribute:: graph_model


   .. py:attribute:: node_model


   .. py:method:: undo()


   .. py:method:: redo()


.. py:class:: RemoveNodeCommand(graph_model, node_model, description = 'Remove Node')

   Bases: :py:obj:`PySide6.QtGui.QUndoCommand`


   .. py:attribute:: graph_model


   .. py:attribute:: node_model


   .. py:attribute:: associated_edges
      :value: []



   .. py:method:: undo()


   .. py:method:: redo()


.. py:class:: AddEdgeCommand(graph_model, edge_model, description = 'Add Edge')

   Bases: :py:obj:`PySide6.QtGui.QUndoCommand`


   .. py:attribute:: graph_model


   .. py:attribute:: edge_model


   .. py:method:: undo()


   .. py:method:: redo()


.. py:class:: RemoveEdgeCommand(graph_model, edge_model, description = 'Remove Edge')

   Bases: :py:obj:`PySide6.QtGui.QUndoCommand`


   .. py:attribute:: graph_model


   .. py:attribute:: edge_model


   .. py:method:: undo()


   .. py:method:: redo()


.. py:class:: ChangeEdgeWaypointsCommand(edge_model, old_waypoints, new_waypoints, description = 'Change Pipe Route')

   Bases: :py:obj:`PySide6.QtGui.QUndoCommand`


   .. py:attribute:: edge_model


   .. py:attribute:: old_waypoints


   .. py:attribute:: new_waypoints


   .. py:method:: undo()


   .. py:method:: redo()


.. py:class:: RenameNodeCommand(node_model, old_name, new_name, description = 'Rename Node')

   Bases: :py:obj:`PySide6.QtGui.QUndoCommand`


   .. py:attribute:: node_model


   .. py:attribute:: old_name


   .. py:attribute:: new_name


   .. py:method:: id()


   .. py:method:: mergeWith(command)


   .. py:method:: undo()


   .. py:method:: redo()


