spark.graph_editor.models.graph_layout
======================================

.. py:module:: spark.graph_editor.models.graph_layout


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.graph_layout.estimate_node_size
   spark.graph_editor.models.graph_layout.layered_layout
   spark.graph_editor.models.graph_layout.bounding_box
   spark.graph_editor.models.graph_layout.offset_below


Module Contents
---------------

.. py:function:: estimate_node_size(node)

   Approximates the rendered size of a node from its ports.


.. py:function:: layered_layout(keys, edges, sizes, h_gap = None, v_gap = None)

   Places a dataflow graph left to right, by dependency depth.

   :param keys: Node identifiers, in a stable order.
   :type keys: list of str
   :param edges: Directed (source, target) dependencies.
   :type edges: list of tuple of str
   :param sizes: Rendered size of every node.
   :type sizes: dict of str to tuple of float
   :param h_gap: Free space between two columns. Read from the style when omitted.
   :type h_gap: float, optional
   :param v_gap: Free space between two nodes of the same column. Read from the style when omitted.
   :type v_gap: float, optional

   :returns: Top-left position of every node.
   :rtype: dict of str to tuple of float


.. py:function:: bounding_box(nodes)

   Bounding box (left, top, right, bottom) of a collection of nodes.


.. py:function:: offset_below(existing, incoming, margin = None)

   Translation that drops a set of new nodes under everything already on the canvas.

   The relative placement of the incoming nodes is preserved, only the block as a whole is moved.


