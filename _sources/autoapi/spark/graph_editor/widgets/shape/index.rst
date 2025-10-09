spark.graph_editor.widgets.shape
================================

.. py:module:: spark.graph_editor.widgets.shape


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.shape.QPushButtonSquare
   spark.graph_editor.widgets.shape.QShapeEdit
   spark.graph_editor.widgets.shape.QShape


Module Contents
---------------

.. py:class:: QPushButtonSquare(*args, **kwargs)

   Bases: :py:obj:`Qt.QtWidgets.QPushButton`


   A square QPushButton


   .. py:method:: hasHeightForWidth()


   .. py:method:: heightForWidth(width)


.. py:class:: QShapeEdit(initial_shape = None, min_dims = 1, max_dims = 8, max_shape = 1000000000.0, is_static = False, parent = None)

   Bases: :py:obj:`Qt.QtWidgets.QWidget`


   A widget for inputting or displaying a shape with a dynamic number of dimensions.


   .. py:attribute:: editingFinished


   .. py:attribute:: min_dims
      :value: 1



   .. py:attribute:: max_dims
      :value: 8



   .. py:attribute:: max_shape
      :value: 0



   .. py:attribute:: is_static
      :value: False



   .. py:method:: get_shape()


   .. py:method:: set_shape(new_shape)

      Clears existing dimensions and sets a new shape.



.. py:class:: QShape(label, initial_shape = (1, ), min_dims = 1, max_dims = 8, max_shape = 1000000000.0, is_static = False, parent = None)

   Bases: :py:obj:`spark.graph_editor.widgets.base.SparkQWidget`


   Custom QWidget used for integer fields in the SparkGraphEditor's Inspector.


   .. py:method:: get_value()

      Returns the widget value.



