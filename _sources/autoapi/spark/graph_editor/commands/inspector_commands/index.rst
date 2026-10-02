spark.graph_editor.commands.inspector_commands
==============================================

.. py:module:: spark.graph_editor.commands.inspector_commands


Attributes
----------

.. autoapisummary::

   spark.graph_editor.commands.inspector_commands.logger


Classes
-------

.. autoapisummary::

   spark.graph_editor.commands.inspector_commands.ChangeConfigValueCommand
   spark.graph_editor.commands.inspector_commands.ToggleInheritanceCommand


Module Contents
---------------

.. py:data:: logger

.. py:class:: ChangeConfigValueCommand(graph_model, path, old_value, new_value, node, description = 'Change Config Value')

   Bases: :py:obj:`PySide6.QtGui.QUndoCommand`


   .. py:attribute:: graph_model


   .. py:attribute:: path


   .. py:attribute:: old_value


   .. py:attribute:: new_value


   .. py:attribute:: node


   .. py:method:: id()


   .. py:method:: mergeWith(command)


   .. py:method:: undo()


   .. py:method:: redo()


.. py:class:: ToggleInheritanceCommand(graph_model, path, old_state, new_state, node, description = 'Toggle Inheritance')

   Bases: :py:obj:`PySide6.QtGui.QUndoCommand`


   .. py:attribute:: graph_model


   .. py:attribute:: path


   .. py:attribute:: old_state


   .. py:attribute:: new_state


   .. py:attribute:: node


   .. py:attribute:: old_child_values


   .. py:method:: undo()


   .. py:method:: redo()


