spark.graph_editor.ui.console_panel
===================================

.. py:module:: spark.graph_editor.ui.console_panel


Classes
-------

.. autoapisummary::

   spark.graph_editor.ui.console_panel.MessageLevel
   spark.graph_editor.ui.console_panel.ConsolePanel
   spark.graph_editor.ui.console_panel.Console


Module Contents
---------------

.. py:class:: MessageLevel(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access:

     >>> Color.RED
     <Color.RED: 1>

   - value lookup:

     >>> Color(1)
     <Color.RED: 1>

   - name lookup:

     >>> Color['RED']
     <Color.RED: 1>

   Enumerations can be iterated over, and know how many members they have:

   >>> len(Color)
   3

   >>> list(Color)
   [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

   Methods can be added to enumerations, and members can have their own
   attributes -- see the documentation for details.


   .. py:attribute:: INFO


   .. py:attribute:: SUCCESS


   .. py:attribute:: WARNING


   .. py:attribute:: ERROR


.. py:class:: ConsolePanel(name = 'Console', **kwargs)

   Bases: :py:obj:`spark.graph_editor.widgets.dock_panel.QDockPanel`


   Console panel to show information / errors.


   .. py:attribute:: console


   .. py:method:: clear()


   .. py:method:: publish_message(level, message)


.. py:class:: Console

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Non-interactable console widget.


   .. py:attribute:: content


   .. py:attribute:: vscrollbar


   .. py:method:: add_message(level, text)

      Add a message to the console.



   .. py:method:: info(text)


   .. py:method:: success(text)


   .. py:method:: warning(text)


   .. py:method:: error(text)


   .. py:method:: clear()

      Remove all messages from the console.



   .. py:method:: scrollToBottom(minimum, maximum)


