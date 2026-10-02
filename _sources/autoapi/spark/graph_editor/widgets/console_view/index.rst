spark.graph_editor.widgets.console_view
=======================================

.. py:module:: spark.graph_editor.widgets.console_view


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.console_view.MessageLevel
   spark.graph_editor.widgets.console_view.ConsoleHandler
   spark.graph_editor.widgets.console_view.ConsoleView


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


   .. py:attribute:: DEBUG
      :value: 10



   .. py:attribute:: INFO
      :value: 20



   .. py:attribute:: SUCCESS
      :value: 25



   .. py:attribute:: WARNING
      :value: 30



   .. py:attribute:: ERROR
      :value: 40



.. py:class:: ConsoleHandler(console_view)

   Bases: :py:obj:`logging.Handler`


   Logging handler that routes messages to the ConsoleView.

   Initializes the instance - basically setting the formatter to None
   and the filter list to empty.


   .. py:attribute:: console_view


   .. py:method:: emit(record)

      Do whatever it takes to actually log the specified logging record.

      This version is intended to be implemented by subclasses and so
      raises a NotImplementedError.



.. py:class:: ConsoleView(parent=None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   General console widget connected to the 'spark' logger.


   .. py:attribute:: content


   .. py:attribute:: vscrollbar


   .. py:method:: add_message(level, text)

      Adds a message to the console.



   .. py:method:: clear()

      Removes every message from the console.



   .. py:method:: scrollToBottom(minimum, maximum)


