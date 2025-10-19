painter
=======

.. py:module:: painter


Attributes
----------

.. autoapisummary::

   painter.DEFAULT_PALLETE


Classes
-------

.. autoapisummary::

   painter.Pallete


Functions
---------

.. autoapisummary::

   painter.make_port_painter
   painter.regular_polygon_builder
   painter.star_builder


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

