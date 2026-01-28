spark.graph_editor.widgets.attributes
=====================================

.. py:module:: spark.graph_editor.widgets.attributes


Attributes
----------

.. autoapisummary::

   spark.graph_editor.widgets.attributes.TYPE_PARSER
   spark.graph_editor.widgets.attributes.WIDGET_PARSER


Classes
-------

.. autoapisummary::

   spark.graph_editor.widgets.attributes.QControlFlags
   spark.graph_editor.widgets.attributes.QAttrControls
   spark.graph_editor.widgets.attributes.QAttribute
   spark.graph_editor.widgets.attributes.QConfigBody
   spark.graph_editor.widgets.attributes.TypeParser
   spark.graph_editor.widgets.attributes.WidgetParserMetadata
   spark.graph_editor.widgets.attributes.WidgetParser


Module Contents
---------------

.. py:class:: QControlFlags

   Bases: :py:obj:`enum.IntFlag`


   Support for integer-based Flags

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: SHOW
      :value: 8



   .. py:attribute:: INTERACTIVE
      :value: 4



   .. py:attribute:: EMPTY_SPACE
      :value: 2



   .. py:attribute:: ACTIVE
      :value: 1



   .. py:attribute:: SHOW_INTERACTIVE
      :value: 12



   .. py:attribute:: SHOW_INTERACTIVE_ACTIVE
      :value: 13



   .. py:attribute:: SHOW_ACTIVE
      :value: 9



   .. py:attribute:: ALL
      :value: 15



.. py:class:: QAttrControls(on_initializer_toggled_callback, on_inheritance_toggled_callback, warning_flags, inheritance_flags, initializer_flags, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   .. py:attribute:: on_update


.. py:class:: QAttribute(attr_label, attr_value, attr_widget, attr_init_mandatory = False, attr_metadata = None, warning_flags = QControlFlags.SHOW_INTERACTIVE, inheritance_flags = QControlFlags.EMPTY_SPACE, initializer_flags = QControlFlags.EMPTY_SPACE, ref_post_callback_stack = None, is_initializer_attr = False, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Base QWidget class for the graph editor attributes.


   .. py:attribute:: on_initializer_toggled


   .. py:attribute:: on_inheritance_toggled


   .. py:attribute:: on_field_update


   .. py:attribute:: on_size_change


   .. py:attribute:: attr_label


   .. py:attribute:: config_widget
      :value: None



   .. py:attribute:: init_config
      :value: None



   .. py:attribute:: main_row


   .. py:attribute:: control_widgets


   .. py:attribute:: secondary_row


   .. py:method:: set_initializer_status(state)

      Sets the error status of widget.



   .. py:method:: set_inheritance_status(state)

      Sets the error status of widget.



   .. py:method:: set_error_status(messages)

      Sets the error status of widget.



   .. py:method:: set_value(value)

      Set the widget value.



   .. py:method:: get_value()

      Get the widget value.



.. py:class:: QConfigBody(config, is_initializer, config_path = None, ref_widgets_map = None, ref_post_callback_stack = None, inheritance_tree = None, on_inheritance_toggle = None, **kwargs)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   QWidget for the body of configuration object.


   .. py:attribute:: on_error_detected


   .. py:attribute:: on_size_change


   .. py:attribute:: on_update


   .. py:attribute:: config


   .. py:method:: get_value()


.. py:class:: TypeParser

   .. py:attribute:: GENERIC_TYPES
      :type:  dict[str, type]


   .. py:attribute:: SPECIAL_TYPES
      :type:  dict[str, type]


   .. py:attribute:: CONTAINER_TYPES
      :type:  re.Pattern


   .. py:method:: typify(raw_types)


.. py:class:: WidgetParserMetadata

   .. py:attribute:: allows_init
      :type:  bool


   .. py:attribute:: requires_init
      :type:  bool


   .. py:attribute:: allows_inheritance
      :type:  bool


   .. py:attribute:: missing
      :type:  bool


   .. py:attribute:: second_option
      :type:  type[PySide6.QtWidgets.QWidget]


.. py:class:: WidgetParser

   .. py:attribute:: WIDGET_TYPE_MAP


   .. py:attribute:: WIDGET_PRIORITY_MAP


   .. py:method:: get_possible_widgets(attr_type)

      Maps an argument type to a corresponding type of possible widgets classes.
      Returns an ordered list of widgets and metadata to format the inspector.



   .. py:method:: get_possible_widgets_raw(raw_types)

      Maps an argument raw type to a corresponding type of possible widgets classes.
      Returns an ordered list of widgets and metadata to format the inspector.



   .. py:method:: get_widget(attr_type)

      Maps an argument type to a corresponding widget class, according to the priority list.
      Returns the best widget fit and metadata to format the inspector.



   .. py:method:: get_widget_raw(raw_types)

      Maps an argument raw type to a corresponding widget class, according to the priority list.
      Returns the best widget fit and metadata to format the inspector.



.. py:data:: TYPE_PARSER

.. py:data:: WIDGET_PARSER

