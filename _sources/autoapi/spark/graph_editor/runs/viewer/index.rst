spark.graph_editor.runs.viewer
==============================

.. py:module:: spark.graph_editor.runs.viewer


Attributes
----------

.. autoapisummary::

   spark.graph_editor.runs.viewer.PREFERENCES


Classes
-------

.. autoapisummary::

   spark.graph_editor.runs.viewer.ActivityBadge
   spark.graph_editor.runs.viewer.ProbePanel
   spark.graph_editor.runs.viewer.InputsPanel
   spark.graph_editor.runs.viewer.RunPanel
   spark.graph_editor.runs.viewer.RunViewerWindow
   spark.graph_editor.runs.viewer.SparkRunViewer


Module Contents
---------------

.. py:data:: PREFERENCES
   :value: ('Canvas', 'Nodes', 'Ports & Payloads', 'Edges', 'Run Viewer', 'Window')


   Sections of the preferences of the editor that change the viewer.

.. py:class:: ActivityBadge(node)

   Bases: :py:obj:`PySide6.QtWidgets.QGraphicsItem`


   Badge showing the firing rate of a node at the cursor.

   Drawn above the top right corner of the node, at the same size at any zoom. Its colour goes
   from the low to the high colour of ``THEME.badge`` as the rate rises to its full scale, in Hz.


   .. py:attribute:: value
      :type:  float | None
      :value: None



   .. py:method:: set_value(hz)

      Sets the rate shown, in Hz, or hides the badge for None.



   .. py:method:: boundingRect()


   .. py:method:: paint(painter, option, widget=None)


.. py:class:: ProbePanel(data, selection = None, store = None, inputs = None, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Panel of plots of what was recorded for one node of the graph.

   Each set of measurements recording the node has a tab. Within a tab, plots are grouped by
   module and titled by port or attribute and reduction. Without a node, the panel plots the
   scalars logged with `Recorder.log`, in tabs by the prefix of their names. ``inputs``, when
   given, is the last tab. The tab chosen last is shown again for the next node that has a tab
   of that name.

   Scalar series are drawn in their natural space, the tag they were written per, or in the
   space ``selection`` chooses, over the steps of `view`; the ``mean``, ``std``, ``min`` and
   ``max`` of a summary share one plot. The cursor, a step, is drawn at the tag value it falls
   in, and a click moves it to the first step of the value clicked. The runs and groups
   ``selection`` shows besides this run are drawn with it, each in its color, and listed under
   the name of the node. A box drawn with the right button in a scalar plot asks for its steps
   with `view_requested`, for every plot.
   Values recorded once per group, such as histograms, active fractions per unit, snapshots
   and changes, show the last written group starting at or before the cursor. Traces and
   rasters show the consecutive recorded steps around the cursor.


   .. py:attribute:: cursor_moved


   .. py:attribute:: view_requested


   .. py:attribute:: LINES
      :value: 16


      Largest number of units of a trace drawn as lines. Wider traces are drawn as images.


   .. py:attribute:: SPREAD
      :value: ('mean', 'std', 'min', 'max')


      the mean as a line, the band of one standard
      deviation around it, and the band from the lowest to the highest value, fainter. Runs
      compared draw their means.

      :type: Reductions of a summary drawn in one plot


   .. py:attribute:: data


   .. py:attribute:: selection
      :value: None



   .. py:attribute:: store
      :value: None



   .. py:attribute:: node
      :type:  str | None
      :value: None



   .. py:attribute:: cursor
      :value: 0



   .. py:attribute:: view
      :type:  tuple[float, float] | None
      :value: None



   .. py:attribute:: tabs


   .. py:method:: release()

      Stops following the lines shown, before the panel is dropped.



   .. py:method:: plots()

      Returns the plots shown, in the order of the tabs and within them.



   .. py:method:: show_node(node)

      Shows the plots of ``node``, or of the logged scalars for None.



   .. py:method:: refresh()

      Draws the scalar series again with the rows read since, and the runs compared.

      Without a node, logged scalars that appeared since are added.



   .. py:method:: set_view(start, end)

      Shows the steps from ``start`` to ``end`` in the scalar plots, each in its space.



   .. py:method:: set_cursor(t)

      Moves the cursor of every plot to step ``t``, in the space of each, and updates the
      values shown at it.



.. py:class:: InputsPanel(data, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Panel of what the model received at the cursor.

   Shows the recorded inputs of the model and the raw streams of every set of measurements, as
   last recorded at or before the cursor, with the step they were recorded at. Single values
   are listed together; vectors are drawn as bars, and a raw stream viewed as an image as an
   image, following the views of its measurements.


   .. py:attribute:: data


   .. py:method:: set_cursor(t)

      Shows the inputs and raw frames at step ``t``.



.. py:class:: RunPanel(data, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QWidget`


   Panel of the status, step, environment and parameters of the run, of its warnings and
   errors, and of its measurements.

   While the run is written, each set of measurements can be recorded for a number of steps.
   The request is made with `Run.record`, and its status is shown under the row. Clicking the
   step of a warning or an error emits `cursor_requested` with that step.


   .. py:attribute:: cursor_requested


   .. py:attribute:: WARNINGS
      :value: 50


      Largest number of warnings and errors listed. The others are counted.


   .. py:attribute:: data


   .. py:attribute:: record_rows
      :type:  dict[str, _RecordRow]


   .. py:method:: refresh()

      Shows the status and step of the run and the status of the last request of each row.

      The rows can record only while the run is written.



.. py:class:: RunViewerWindow(path, refresh_every = 1000, parent = None)

   Bases: :py:obj:`PySide6.QtWidgets.QMainWindow`


   Window comparing the runs of a directory, and showing one of them over the graph of its
   model.

   Opened on a directory, the window shows its Workspace: a panel for every scalar series of its
   runs, drawn for every run or group the Runs table shows, each in the space it was recorded
   in. Opened on a run, it shows that run, compared with nothing until other runs are shown. A
   double click on a run of the table shows it on the Graph tab, the Details and Probes panels
   and the timeline. Runs added to the directory are listed as they appear.

   The exploration of the directory, what the window shows and how, is kept as it changes, in a
   file of the viewer rather than within the directory, and resumed when the directory is opened
   again. File > Save Exploration As keeps a copy elsewhere, which File > Open Exploration
   resumes.

   The graph shows at the cursor the firing rate of every node whose spikes are summarized.
   Selecting a node shows what was recorded for it. The timeline moves the cursor through the
   run, and its zoom sets the steps the scalar plots show. The toolbar plays the run forward
   and moves the cursor to the previous or next recorded span. A run still being written is
   read again periodically. Its measurements can be recorded from the window, and Follow keeps
   the cursor at the last step. The Probes panel shows what was recorded for the node selected,
   and what the model received, in tabs. The Probes panel and the timeline belong to the Graph
   tab: both are collapsed while the Workspace is shown, and open again with the Graph tab
   unless closed there. The Runs and Details panels span the height of the window; the timeline lies
   under the graph and the Probes panel. On the Graph tab, the Probes panel can take all but a
   sliver of the graph.

   The runs and groups the Runs table shows besides the run shown are drawn with it in the scalar
   plots of the Probes panel. The menu bar opens other directories and runs, the preferences of
   the look of the viewer, and the panels closed.

   Keys: Left and Right move the cursor by 1% of the steps the timeline shows, one step with
   Shift. Home and End go to the first and the last step, PageUp and PageDown to the start of
   the previous and next recorded span. Space plays and pauses.

   :param path: Directory of a run, or of runs.
   :type path: str or path-like
   :param refresh_every: Milliseconds between two reads of the run. Once a read finds a change while the run is
                         not being written, the interval is five times as long.
   :type refresh_every: int, default 1000
   :param parent: Parent widget.
   :type parent: QWidget, optional

   .. attribute:: project

      The runs of the directory.

      :type: Project

   .. attribute:: exploration

      Where the state of the window is kept for the directory.

      :type: Exploration

   .. attribute:: selection

      The runs shown, their groups and how they are drawn.

      :type: Selection

   .. attribute:: store

      The series read from the runs.

      :type: SeriesStore

   .. attribute:: data

      What the window read from the run shown.

      :type: RunData

   .. attribute:: cursor

      Step at the cursor.

      :type: int

   .. attribute:: badges

      Badge of every node of the graph, by name.

      :type: dict of str to ActivityBadge

   .. attribute:: refresh_every

      Milliseconds between two reads of a run being written.

      :type: int

   .. rubric:: Notes

   When the model of the run cannot be loaded, the window shows the error in place of the
   graph. Custom neurons must be registered before the run is opened, as for the editor.
   Closing the window drops what it read from the run.

   .. seealso::

      :py:obj:`SparkRunViewer`
          Opens runs in windows of their own from a script or a notebook.

      :py:obj:`RunData`
          What the viewer reads from a run.

      :py:obj:`spark.recording.Run`
          A run written by a `Recorder`, read back.


   .. py:attribute:: project


   .. py:attribute:: selection


   .. py:attribute:: store


   .. py:attribute:: data


   .. py:attribute:: cursor
      :value: 0



   .. py:attribute:: badges
      :type:  dict[str, ActivityBadge]


   .. py:attribute:: center


   .. py:attribute:: workspace


   .. py:attribute:: refresh_every
      :value: 1000



   .. py:attribute:: exploration


   .. py:method:: state()

      Returns the state of the exploration: the directory, the run shown, its node selected,
      the cursor and the view of the timeline, the tab shown, the runs shown and how, the
      table, the workspace and the layout of the panels.



   .. py:method:: resume(state, shown = True)

      Resumes the exploration ``state``, with the run it shows when ``shown``.

      What no longer applies, such as a run deleted since, is left out. A state that cannot be
      resumed whole is reported in the status bar.



   .. py:method:: keep_exploration()

      Writes the state of the window to its exploration when it changed since last written.

      A failure is shown in the status bar.



   .. py:method:: save_exploration(path)

      Writes the state of the window to the file ``path``, which cannot be within a directory
      of runs.

      Returns whether it was written; a refusal or a failure is reported in a message box.



   .. py:method:: open_exploration(path)

      Resumes the exploration saved in the file ``path``: in this window when it explores the
      same directory, else in a window of its own.

      Returns the window, or None when the file cannot be read or its directory opened, which
      is reported in a message box.



   .. py:method:: show_run(path)

      Shows the run at ``path`` on the Graph tab, the Details and Probes panels and the timeline.

      A run that cannot be read is reported in a message box.



   .. py:method:: open_run(path)

      Opens the run, or the directory of runs, at ``path`` in a window of its own.

      Returns the window, or None when nothing at ``path`` can be opened, which is reported in
      a message box.



   .. py:method:: quit(checked = False)

      Closes every window of the run viewer.

      The application goes on when it holds other windows, such as the editor's, or runs in a
      notebook.



   .. py:method:: open_preferences(checked = False)

      Opens the preferences of the editor, limited to what changes the viewer.

      Applying them redraws every window of the viewer.



   .. py:method:: restyle()

      Draws the window again with the style: the plots, the timeline and the graph.

      Called when the style is reloaded, as by the preferences, once `THEME` has read it.



   .. py:method:: set_cursor(step)

      Moves the cursor to a step and updates the timeline, the plots and the badges.

      The status bar shows the step and the error of the last read that failed.

      :param step: Step, clipped to the steps of the run.
      :type step: int



   .. py:method:: refresh()

      Reads what the run added since the last refresh and redraws what changed.

      Called by a timer. A read that fails is shown in the status bar and tried again on the
      next refresh. With Follow checked, the cursor moves to the last step.



   .. py:method:: showEvent(event)


   .. py:method:: closeEvent(event)


.. py:class:: SparkRunViewer

   Opens directories of runs, and runs, in windows of their own from a script or a notebook.

   Available as ``spark.RunViewer``. In a notebook, the Qt event loop runs within the kernel
   and `open` returns at once. In a script, `open` runs the event loop until the windows are
   closed.

   .. attribute:: app

      The application, created when none exists.

      :type: QApplication

   .. attribute:: windows

      Windows opened, those closed dropped at the next `open`.

      :type: list of RunViewerWindow

   .. seealso::

      :py:obj:`RunViewerWindow`
          The window of a directory of runs.

      :py:obj:`SparkGraphEditor`
          The graph editor, opened in the same way.

   .. rubric:: Examples

   >>> viewer = spark.RunViewer()
   >>> viewer.open('runs')                                         # every run, compared
   >>> viewer.open('runs/20260923-091240_cartpole_4784ed')        # one run


   .. py:attribute:: app


   .. py:attribute:: windows
      :type:  list[RunViewerWindow]
      :value: []



   .. py:method:: open(path)

      Opens the directory of runs, or the run, at ``path`` in a new window.

      Outside a notebook, blocks until the windows are closed.

      :param path: Directory of runs, opened on its workspace, or of a run.
      :type path: str or path-like

      :returns: The window.
      :rtype: RunViewerWindow

      :raises FileNotFoundError: When ``path`` is not a run and holds none.
      :raises ValueError: When the run was written with another version of the index.



