spark.graph_editor.style.painter
================================

.. py:module:: spark.graph_editor.style.painter


Attributes
----------

.. autoapisummary::

   spark.graph_editor.style.painter.DEFAULT_COLOR
   spark.graph_editor.style.painter.DEFAULT_HOVER_COLOR
   spark.graph_editor.style.painter.DEFAULT_CONNECTED_COLOR
   spark.graph_editor.style.painter.PROPERTY_COLOR
   spark.graph_editor.style.painter.PROPERTY_HOVER_COLOR
   spark.graph_editor.style.painter.PROPERTY_CONNECTED_COLOR
   spark.graph_editor.style.painter.OPTIONAL_COLOR
   spark.graph_editor.style.painter.OPTIONAL_HOVER_COLOR
   spark.graph_editor.style.painter.OPTIONAL_CONNECTED_COLOR
   spark.graph_editor.style.painter.MISSING_COLOR
   spark.graph_editor.style.painter.NOT_CONNECTED_FILL
   spark.graph_editor.style.painter.DEFAULT_PALETTE


Classes
-------

.. autoapisummary::

   spark.graph_editor.style.painter.PortColorStyle
   spark.graph_editor.style.painter.Palette


Functions
---------

.. autoapisummary::

   spark.graph_editor.style.painter.make_port_painter
   spark.graph_editor.style.painter.regular_polygon_builder
   spark.graph_editor.style.painter.star_builder


Module Contents
---------------

.. py:class:: PortColorStyle(*args, **kwds)

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


   .. py:attribute:: DEFAULT


   .. py:attribute:: OPTIONAL


   .. py:attribute:: PROPERTY


.. py:data:: DEFAULT_COLOR

.. py:data:: DEFAULT_HOVER_COLOR

.. py:data:: DEFAULT_CONNECTED_COLOR

.. py:data:: PROPERTY_COLOR

.. py:data:: PROPERTY_HOVER_COLOR

.. py:data:: PROPERTY_CONNECTED_COLOR

.. py:data:: OPTIONAL_COLOR

.. py:data:: OPTIONAL_HOVER_COLOR

.. py:data:: OPTIONAL_CONNECTED_COLOR

.. py:data:: MISSING_COLOR

.. py:data:: NOT_CONNECTED_FILL

.. py:function:: make_port_painter(shape_builder, color_style = PortColorStyle.DEFAULT)

.. py:function:: regular_polygon_builder(sides)

.. py:function:: star_builder(points, inner_ratio=0.5)

   Builds a star (spark) with `points` spikes.
   inner_ratio: fraction of outer radius for the inner vertices.


.. py:class:: Palette

   .. py:attribute:: builders


   .. py:method:: __call__(payload, color_style = PortColorStyle.DEFAULT)


.. py:data:: DEFAULT_PALETTE

