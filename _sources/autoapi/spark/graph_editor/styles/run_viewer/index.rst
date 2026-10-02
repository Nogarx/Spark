spark.graph_editor.styles.run_viewer
====================================

.. py:module:: spark.graph_editor.styles.run_viewer


Attributes
----------

.. autoapisummary::

   spark.graph_editor.styles.run_viewer.GROUPS
   spark.graph_editor.styles.run_viewer.THEME


Classes
-------

.. autoapisummary::

   spark.graph_editor.styles.run_viewer.RunViewerTheme


Module Contents
---------------

.. py:data:: GROUPS

   Categories of the style read as groups of the theme, by the name of the group. The category
   ``run_viewer`` is read as the theme itself, and ``run_viewer_status`` by the stylesheet only.

.. py:class:: RunViewerTheme

   The style of the run viewer: colours, fonts and sizes of its plots, timeline, badges and
   panels.

   The values of the category ``run_viewer`` of the style are attributes of the theme; those of
   the categories of `GROUPS` are attributes of its groups, such as ``THEME.timeline.row_height``.
   A colour is a QColor named without its ``_color`` suffix (``area_color`` is ``THEME.area``),
   a palette a list of QColor. Read again when the style is reloaded.

   .. attribute:: colormap_pixels

      The colormap of the style over 256 values, as ARGB32 pixels, the lowest first.

      :type: ndarray

   .. attribute:: colorbar_image

      ``colormap_pixels`` as an image one pixel high.

      :type: QImage

   .. attribute:: timeline, badge, layout

      The values of the categories of `GROUPS`.

      :type: SimpleNamespace


   .. py:method:: read()

      Reads the theme from the style.



   .. py:method:: font(size = None, bold = False)

      Returns the font of the run viewer, at ``size`` points or its own size.



   .. py:method:: series_color(index)

      Returns the colour of the series ``index``, the series colours repeated past the last.



   .. py:method:: with_alpha(color, alpha)
      :staticmethod:


      Returns ``color`` with the opacity ``alpha``, from 0 to 255.



.. py:data:: THEME

   The style of the run viewer.

