spark.graph_editor.styles.resources
===================================

.. py:module:: spark.graph_editor.styles.resources


Attributes
----------

.. autoapisummary::

   spark.graph_editor.styles.resources.BRAIN
   spark.graph_editor.styles.resources.NEURON
   spark.graph_editor.styles.resources.SIMPLE
   spark.graph_editor.styles.resources.COMPLEX
   spark.graph_editor.styles.resources.LINK
   spark.graph_editor.styles.resources.LOCK
   spark.graph_editor.styles.resources.NODE
   spark.graph_editor.styles.resources.DOT


Functions
---------

.. autoapisummary::

   spark.graph_editor.styles.resources.get_pixmap
   spark.graph_editor.styles.resources.empty_pixmap
   spark.graph_editor.styles.resources.get_faded_pixmap
   spark.graph_editor.styles.resources.get_icon
   spark.graph_editor.styles.resources.get_toggle_icon


Module Contents
---------------

.. py:data:: BRAIN
   :value: ':/icons/brain_icon.png'


.. py:data:: NEURON
   :value: ':/icons/neuron_icon.png'


.. py:data:: SIMPLE
   :value: ':/icons/simple_icon.png'


.. py:data:: COMPLEX
   :value: ':/icons/complex_icon.png'


.. py:data:: LINK
   :value: ':/icons/link_icon.png'


.. py:data:: LOCK
   :value: ':/icons/lock_icon.png'


.. py:data:: NODE
   :value: ':/icons/node_icon.png'


.. py:data:: DOT
   :value: ':/icons/dot_icon.png'


.. py:function:: get_pixmap(path, size = None)

   Returns a cached pixmap from the editor resources.

   :param path: Resource path (e.g. ":/icons/brain_icon.png").
   :type path: str
   :param size: Square size the pixmap is scaled to.
   :type size: int, optional

   :returns: The requested pixmap. Empty if the resource does not exist.
   :rtype: QPixmap


.. py:function:: empty_pixmap(size = 16)

   Fully transparent pixmap, used to keep icon slots aligned while empty.


.. py:function:: get_faded_pixmap(path, size = None, opacity = 0.35)

   Returns a dimmed copy of a resource, used to show an action that is available but not active.


.. py:function:: get_icon(path, size = None)

   Returns a QIcon built from the editor resources.


.. py:function:: get_toggle_icon(on_path, off_path, size = None, off_opacity = 1.0)

   Returns a state aware QIcon that follows the checked state of a button.

   :param on_path: Resource shown while checked. None renders nothing.
   :type on_path: str or None
   :param off_path: Resource shown while unchecked. None renders nothing.
   :type off_path: str or None
   :param size: Square size the icon is scaled to.
   :type size: int, optional
   :param off_opacity: Dims the unchecked resource. Use it to show that an action is available without implying that
                       it is active.
   :type off_opacity: float, default 1.0

   :returns: The two state icon.
   :rtype: QIcon


