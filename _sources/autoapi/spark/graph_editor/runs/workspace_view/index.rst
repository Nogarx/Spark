spark.graph_editor.runs.workspace_view
======================================

.. py:module:: spark.graph_editor.runs.workspace_view


Attributes
----------

.. autoapisummary::

   spark.graph_editor.runs.workspace_view.ORDER


Classes
-------

.. autoapisummary::

   spark.graph_editor.runs.workspace_view.RunsTable
   spark.graph_editor.runs.workspace_view.Panel
   spark.graph_editor.runs.workspace_view.Section
   spark.graph_editor.runs.workspace_view.KeyPicker
   spark.graph_editor.runs.workspace_view.WorkspaceView


Functions
---------

.. autoapisummary::

   spark.graph_editor.runs.workspace_view.section_of
   spark.graph_editor.runs.workspace_view.in_section


Module Contents
---------------

.. py:data:: ORDER

   Role of the first column holding the position of a run in the order the runs were created.

.. py:class:: RunsTable(project, selection, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Table of the runs of a project, setting what the workspace and the probe panel draw.

   Every run has an eye, a check box that shows or hides it, and the color it is drawn in.
   Grouped by fields of the runs (Group by), the table is a tree of groups, each with its eye,
   its color and its runs under it. The search field keeps the runs whose name, experiment or
   parameters hold its text. A run is named by the name given to its `Recorder`, next to the
   time it was created; the runs are sorted by it, the newest first, until sorted by another
   column. Columns give the experiment, the steps and status, and the parameters in which the
   runs differ; more are chosen from Columns. Columns are resized by
   dragging the edges of their headers; until it is, the first takes the width the others leave,
   and the last fills what is left after it. A double click on a run
   emits `run_activated` with its path. A right click on a run sets its experiment or its color.

   :param project: The runs.
   :type project: Project
   :param selection: What is shown, set here.
   :type selection: Selection
   :param parent: Parent widget.
   :type parent: QWidget, optional


   .. py:attribute:: run_activated


   .. py:attribute:: FIXED
      :value: ('Run', 'Created', 'Experiment', 'Steps')



   .. py:attribute:: project


   .. py:attribute:: selection


   .. py:attribute:: columns
      :type:  list[str]


   .. py:attribute:: search


   .. py:attribute:: tree


   .. py:method:: rebuild()

      Lists the runs again, grouped as the selection says, keeping what is expanded.



   .. py:method:: eventFilter(watched, event)


   .. py:method:: set_columns(fields)

      Shows the parameters ``fields`` as columns.



   .. py:method:: state()

      Returns the columns and their widths, the search, the groups expanded and the sorting
      of the table, as an `Exploration` keeps them.



   .. py:method:: restore(state)

      Shows the table as ``state`` gives it, without the parameters the runs no longer have.



   .. py:method:: set_experiment(paths, name)

      Sets the experiment of the runs at ``paths`` in the viewer, or none for an empty name.



.. py:function:: section_of(key)

   Returns the section of the workspace of the series ``key``: the prefix of its name, with the
   node of a summary, such as ``'summary · A_excitatory'`` for
   ``'summary/A_excitatory.soma:spikes/active_fraction'``, or ``'logged'`` without a prefix.


.. py:function:: in_section(key)

   Returns the name of the series ``key`` within its section, without the prefix the section
   is named by, such as ``'soma:spikes/active_fraction'`` for
   ``'summary/A_excitatory.soma:spikes/active_fraction'`` and ``'steps'`` for
   ``'episode/steps'``.


.. py:class:: Panel(keys, selection, store, pinned = False, title = None, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   A panel of the workspace: scalar series drawn for every line of a `Selection`.

   Drawn in the space the selection chooses, or else in the natural space of its first series.
   A group is a strong line, the mean of its runs, over a faint band from their lowest to their
   highest value, with its runs as faint lines when the selection draws them. With several
   series, every line of every series has a color of its own. A box drawn with the right button
   zooms the panel; a double click fits it again.

   :param keys: Names of the series.
   :type keys: sequence of str
   :param selection: What is drawn.
   :type selection: Selection
   :param store: Where the series are read.
   :type store: SeriesStore
   :param pinned: Whether the panel is in the Pinned section: it has a button removing it, else one
                  pinning it.
   :type pinned: bool, default False
   :param title: Title of the plot, the names of the series by default.
   :type title: str, optional


   .. py:attribute:: pin_requested


   .. py:attribute:: removed


   .. py:attribute:: keys


   .. py:attribute:: selection


   .. py:attribute:: store


   .. py:attribute:: legend


   .. py:attribute:: plot


   .. py:method:: zoom()

      Returns the ranges the panel is zoomed to, ``x`` and ``y``, or None when it is not.



   .. py:method:: set_zoom(x, y)

      Zooms the panel to the range ``x`` of its axis and ``y`` of its values, each fitted for
      None.



   .. py:method:: space()

      Returns the space the panel is drawn in.



   .. py:method:: draw()

      Draws the series of the panel for every line of the selection.



.. py:class:: Section(name, keys, workspace, opened, pinned = False, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   A section of the workspace: panels of series sharing a prefix, open or collapsed.

   Its panels are made, and their series read, when it is first open. The search of the
   workspace shows the panels whose series match, and hides a section without any.


   .. py:attribute:: pin_requested


   .. py:attribute:: removed


   .. py:attribute:: name


   .. py:attribute:: keys


   .. py:attribute:: workspace


   .. py:attribute:: pinned
      :value: False



   .. py:attribute:: panels
      :type:  dict[str, Panel]


   .. py:attribute:: header


   .. py:property:: opened
      :type: bool



   .. py:method:: set_open(opened)

      Opens the section, making its panels, or collapses it.



   .. py:method:: arrange()

      Places the panels matching the search, in rows of the columns of the workspace.



   .. py:method:: set_filter(needle)

      Shows the panels whose series hold ``needle``, and the section when it has any.



   .. py:method:: draw()


   .. py:method:: remove(key)


.. py:class:: KeyPicker(counts, runs, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QDialog`


   Dialog choosing scalar series by name, with the number of runs that have each.


   .. py:method:: keys()

      Returns the names checked.



.. py:class:: WorkspaceView(project, selection, store, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   The workspace: a panel for every scalar series of the runs of a project, in sections named by
   the prefix of the series, drawn for every run or group shown.

   The settings bar sets, for every panel, the x axis (the natural space of each series, steps,
   wall time, or an integer tag), the smoothing, whether the runs of groups are drawn faintly,
   and the panels per row. The search field shows the panels whose series match. Panels added by
   hand (Add panel), of one or more series, and panels pinned from a section are in the Pinned
   section, on top. Sections of more than `OPEN` panels start collapsed.

   :param project: The runs.
   :type project: Project
   :param selection: What is drawn.
   :type selection: Selection
   :param store: Where the series are read.
   :type store: SeriesStore
   :param parent: Parent widget.
   :type parent: QWidget, optional


   .. py:attribute:: OPEN
      :value: 12


      Largest number of panels of a section open from the start.


   .. py:attribute:: project


   .. py:attribute:: selection


   .. py:attribute:: store


   .. py:attribute:: columns
      :value: 2



   .. py:attribute:: sections
      :type:  dict[str, Section]


   .. py:attribute:: search


   .. py:attribute:: pinned


   .. py:method:: keys()

      Returns the names of the scalar series of the runs of the project.



   .. py:method:: rebuild()

      Makes the sections again from the series of the runs, keeping which are open.



   .. py:method:: set_columns(columns)

      Places ``columns`` panels per row.



   .. py:method:: set_search(text)

      Shows the panels whose series hold ``text``.



   .. py:method:: refresh(redraw = True)

      Makes the sections again when the runs hold series not listed yet, else draws every
      panel again with ``redraw``.



   .. py:method:: redraw()

      Draws every panel made again, as after rows are written.



   .. py:method:: state()

      Returns the panels per row, the search, the sections open, the pinned panels and the
      zooms of the workspace, as an `Exploration` keeps them.



   .. py:method:: restore(state)

      Shows the workspace as ``state`` gives it.



   .. py:method:: add_panel(keys)

      Adds a panel of the series ``keys`` to the Pinned section.



   .. py:method:: pin(panel)

      Pins a copy of ``panel`` on top.



