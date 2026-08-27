spark.graph_editor.widgets.attribute_view
=========================================

.. py:module:: spark.graph_editor.widgets.attribute_view


Attributes
----------

.. autoapisummary::

   spark.graph_editor.widgets.attribute_view.logger
   spark.graph_editor.widgets.attribute_view.DEFAULT_DTYPES


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.attribute_view.QDimsEdit
   spark.graph_editor.widgets.attribute_view.QAttrControls
   spark.graph_editor.widgets.attribute_view.QAttribute


Module Contents
---------------

.. py:data:: logger

.. py:data:: DEFAULT_DTYPES

.. py:class:: QDimsEdit(value = None, is_integer = True, minimum = 1, maximum = 1000000000.0, min_dims = _MIN_DIMS, max_dims = _MAX_DIMS, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Editor for variable length numeric tuples (e.g. tuple[int, ...] shapes).

   The number of entries is set through the add/remove buttons. The widget always reports a valid tuple.


   .. py:attribute:: value_changed


   .. py:attribute:: is_integer
      :value: True



   .. py:attribute:: minimum
      :value: 1



   .. py:attribute:: maximum
      :value: 1000000000.0



   .. py:attribute:: min_dims


   .. py:attribute:: max_dims
      :value: 8



   .. py:method:: value()

      Current value as a valid tuple.



   .. py:method:: set_value(value)

      Rebuilds the editor from a value, without emitting change notifications.



   .. py:method:: add_dim()

      Appends a new dimension, seeded with the value of the last one.



   .. py:method:: remove_dim()

      Removes the last dimension. The editor never drops below min_dims.



.. py:class:: QAttrControls(node, config_path = None, graph_model = None, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Holds the toggle buttons for Warning, Initializer, and Inheritance.


   .. py:attribute:: node


   .. py:attribute:: config_path
      :value: None



   .. py:attribute:: graph_model
      :value: None



   .. py:attribute:: warning_btn


   .. py:attribute:: init_btn


   .. py:attribute:: inherit_btn


.. py:class:: QAttribute(node, config_path = None, graph_model = None, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Composite widget containing the input field.


   .. py:attribute:: node


   .. py:attribute:: config_path
      :value: None



   .. py:attribute:: graph_model
      :value: None



   .. py:attribute:: top_row


