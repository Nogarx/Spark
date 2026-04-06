spark.graph_editor.widgets.line_edits
=====================================

.. py:module:: spark.graph_editor.widgets.line_edits


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.line_edits.QStrLineEdit
   spark.graph_editor.widgets.line_edits.QString
   spark.graph_editor.widgets.line_edits.QIntLineEdit
   spark.graph_editor.widgets.line_edits.QInt
   spark.graph_editor.widgets.line_edits.QFloatLineEdit
   spark.graph_editor.widgets.line_edits.QFloat


Module Contents
---------------

.. py:class:: QStrLineEdit(initial_value, placeholder = '', **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QLineEdit`


   A simple QLineEdit with a placeholder.


.. py:class:: QString(initial_value, placeholder = '', **kwargs)

   Bases: :py:obj:`spark.graph_editor.widgets.base.QInput`


   Custom QWidget used for string fields in the SparkGraphEditor's Inspector.


   .. py:method:: get_value()

      Returns the widget value.



   .. py:method:: set_value(value)

      Sets the widget value.



.. py:class:: QIntLineEdit(initial_value, bottom_value = None, placeholder = None, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QLineEdit`


   A QLineEdit that only accepts integers.


   .. py:method:: get_value()


.. py:class:: QInt(initial_value = 0, bottom_value = None, placeholder = None, **kwargs)

   Bases: :py:obj:`spark.graph_editor.widgets.base.QInput`


   Custom QWidget used for int fields in the SparkGraphEditor's Inspector.


   .. py:method:: get_value()

      Returns the widget value.



   .. py:method:: set_value(value)

      Sets the widget value.



.. py:class:: QFloatLineEdit(initial_value, bottom_value = None, placeholder = None, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QLineEdit`


   A QLineEdit that only accepts floating-point numbers with flexible precision.


   .. py:method:: get_value()


.. py:class:: QFloat(initial_value = 0, bottom_value = None, placeholder = None, **kwargs)

   Bases: :py:obj:`spark.graph_editor.widgets.base.QInput`


   Custom QWidget used for float fields in the SparkGraphEditor's Inspector.


   .. py:method:: get_value()

      Returns the widget value.



   .. py:method:: set_value(value)

      Sets the widget value.



