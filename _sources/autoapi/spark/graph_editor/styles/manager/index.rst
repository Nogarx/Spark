spark.graph_editor.styles.manager
=================================

.. py:module:: spark.graph_editor.styles.manager


Attributes
----------

.. autoapisummary::

   spark.graph_editor.styles.manager.logger
   spark.graph_editor.styles.manager.STYLES


Classes
-------

.. autoapisummary::

   spark.graph_editor.styles.manager.StyleManager


Module Contents
---------------

.. py:data:: logger

.. py:class:: StyleManager

   Bases: :py:obj:`PySide6.QtCore.QObject`


   .. py:attribute:: reloaded


   .. py:method:: init()


   .. py:method:: stylesheet()


   .. py:method:: apply(app)


   .. py:method:: reload(app = None)


   .. py:method:: get_color(category, key)


   .. py:method:: get_port_style(port_type)


   .. py:method:: get_val(*path, default = None)


.. py:data:: STYLES

