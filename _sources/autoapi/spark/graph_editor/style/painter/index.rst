spark.graph_editor.style.painter
================================

.. py:module:: spark.graph_editor.style.painter


Attributes
----------

.. autoapisummary::

   spark.graph_editor.style.painter.DEFAULT_PALLETE


Classes
-------

.. autoapisummary::

   spark.graph_editor.style.painter.Pallete


Functions
---------

.. autoapisummary::

   spark.graph_editor.style.painter.make_port_painter
   spark.graph_editor.style.painter.regular_polygon_builder
   spark.graph_editor.style.painter.star_builder


Module Contents
---------------

.. py:function:: make_port_painter(shape_builder)

.. py:function:: regular_polygon_builder(sides)

.. py:function:: star_builder(points, inner_ratio=0.5)

   Builds a star (spark) with `points` spikes.
   inner_ratio: fraction of outer radius for the inner vertices.


.. py:class:: Pallete

   .. py:attribute:: pallete


   .. py:method:: __call__(payload)


.. py:data:: DEFAULT_PALLETE

