spark.graph_editor.runs.workspace
=================================

.. py:module:: spark.graph_editor.runs.workspace


Attributes
----------

.. autoapisummary::

   spark.graph_editor.runs.workspace.NATURAL
   spark.graph_editor.runs.workspace.BUCKETS
   spark.graph_editor.runs.workspace.SUFFIX


Classes
-------

.. autoapisummary::

   spark.graph_editor.runs.workspace.Project
   spark.graph_editor.runs.workspace.Spaces
   spark.graph_editor.runs.workspace.SeriesStore
   spark.graph_editor.runs.workspace.Band
   spark.graph_editor.runs.workspace.Line
   spark.graph_editor.runs.workspace.Selection
   spark.graph_editor.runs.workspace.Exploration


Functions
---------

.. autoapisummary::

   spark.graph_editor.runs.workspace.identity
   spark.graph_editor.runs.workspace.run_names
   spark.graph_editor.runs.workspace.axis_label
   spark.graph_editor.runs.workspace.smooth
   spark.graph_editor.runs.workspace.aggregate
   spark.graph_editor.runs.workspace.explorations_path
   spark.graph_editor.runs.workspace.within_runs


Module Contents
---------------

.. py:data:: NATURAL
   :value: 'natural'


   The space chosen to draw every series in the space it was written per.

.. py:data:: BUCKETS
   :value: 500


   Buckets of the x axis in which the runs of a group are averaged, in steps and wall time.

.. py:data:: SUFFIX
   :value: '.exploration.json'


   Ending of the files of explorations.

.. py:function:: identity(run)

   Returns the name given to the `Recorder` of a run and the time the run was created.

   Read from ``run.json``, else from the name of its directory, `_DIRECTORY`. The time is the
   local time of the host that created the run, or None when it cannot be read.


.. py:function:: run_names(runs)

   Returns the name every run is shown by: the name given to its `Recorder` and the minute it
   was created, as ``'cartpole · 10-01 21:23'``.

   Runs that would share a name are told apart by the second they were created, then by the
   order they were created in, as ``'cartpole · 10-01 21:23:05 (2)'``.

   :param runs: The runs, oldest first.
   :type runs: sequence of Run

   :returns: The names, by path of run.
   :rtype: dict of str to str


.. py:class:: Project(root, parent = None)

   Bases: :py:obj:`PySide6.QtCore.QObject`


   The runs of a directory, looked for again as new ones appear.

   The experiment of a run is the one given to its `Recorder`, unless set in the viewer with
   `set_experiment`, which writes nothing to the run. Fields describe the runs for searching
   and grouping them: ``experiment``, ``name``, and every key of their parameters.

   :param root: Directory of the runs.
   :type root: str or path-like
   :param parent: Parent object.
   :type parent: QObject, optional

   .. attribute:: root

      Directory of the runs.

      :type: pathlib.Path

   .. attribute:: runs

      The runs, oldest first.

      :type: list of Run

   .. attribute:: overrides

      Experiments set in the viewer, by path of run. None takes a run out of its experiment.

      :type: dict of str to str or None


   .. py:attribute:: changed


   .. py:attribute:: root


   .. py:attribute:: runs
      :type:  list[spark.recording.run.Run]
      :value: []



   .. py:attribute:: overrides
      :type:  dict[str, str | None]


   .. py:method:: refresh()

      Opens the runs added to the directory since, and reads again those being written.

      Returns whether a run was added. Emits `changed` then. A directory that cannot be listed
      adds none.



   .. py:method:: name(run)

      Returns the name ``run`` is shown by, as `run_names` gives it.



   .. py:method:: run(path)

      Returns the run at ``path``, or None when the directory holds none there.



   .. py:method:: experiment(run)

      Returns the experiment of ``run``: the one set in the viewer, else the recorded one.



   .. py:method:: set_experiment(run, name)

      Sets the experiment of ``run`` in the viewer, or takes it out of any for None.



   .. py:method:: state()

      Returns the experiments set in the viewer, by name of run, as an `Exploration` keeps them.



   .. py:method:: restore(state)

      Sets the experiments of ``state`` to the runs of the directory with those names.



   .. py:method:: fields()

      Returns the fields of the runs: ``experiment``, ``name``, then the keys of their
      parameters in sorted order.



   .. py:method:: value(run, field)

      Returns the value of a field of ``run``, None when it has none.



.. py:function:: axis_label(space)

   Returns the name of the horizontal axis of ``space``: ``'step'``, ``'wall time (s)'``, or
   the name of the tag.


.. py:class:: Spaces(run)

   The spaces the series of a run are drawn in, and the mapping of their rows between them.

   A space is ``'step'``, ``'wall'``, or the name of an integer tag of the run, such as
   ``'episode'``. A row at a step holds, in a tag space, the value the tag held at that step,
   and in wall time, the seconds since the first row of the run. The natural space of a series
   is the tag it was written per (`Run.tag_of`), or steps.

   :param run: The run.
   :type run: Run


   .. py:attribute:: run


   .. py:method:: natural(key)

      Returns the space ``key`` was written per: the name of its tag, or ``'step'``.



   .. py:method:: tags()

      Returns the names of the tags of the run set to integers, in sorted order.



   .. py:method:: timeline(tag)

      Returns the steps at which ``tag`` took a new integer value, and the values, ordered by
      step. A value set again as it was is not a new one.

      Read once; call `forget` for a run still being written.



   .. py:method:: forget()

      Drops what was read, to read it again, as for a run being written.



   .. py:method:: to_space(steps, space)

      Returns the position of rows at ``steps`` in ``space``, NaN for rows it holds no value
      at, such as rows before a tag is first set.



   .. py:method:: to_steps(x, space)

      Returns the steps ``[first, end)`` holding the position ``x`` of ``space``, or None.

      A value of a tag holds from the first step it is set at to the step it next changes;
      the last value, to the last step of the run. A step, or a wall time, holds one step.



.. py:class:: SeriesStore

   Scalar series read from runs, by run, name and space, kept until read again.

   A series is read as its envelope past `POINTS` rows. Series of a run being written are read
   again at most every `REREAD` seconds. A reader set with `set_reader` reads the series of a
   run instead, such as those the viewer keeps up to date for the run it shows. In a tag
   space, the rows holding the same value of the tag are averaged into one.


   .. py:attribute:: REREAD
      :value: 30.0


      Seconds after which the series of a run being written are read again.


   .. py:method:: set_reader(run, reader)

      Reads the series of ``run`` with ``reader``, called with the name of a series, or as
      by default for None.



   .. py:method:: spaces(run)

      Returns the spaces of ``run``.



   .. py:method:: keys(run)

      Returns the names of the scalar series of ``run``, or none when they cannot be read.



   .. py:method:: rows(run, key)

      Returns the steps and values of a series of ``run``, empty when it has none.



   .. py:method:: series(run, key, space = NATURAL)

      Returns a series of ``run`` in ``space``, its natural space by default, as positions
      and values ordered by position.

      Rows without a position in the space are left out. In a tag space, the rows holding
      the same value are averaged into one.



   .. py:method:: forget(run)

      Drops what was read of ``run``, read again when next asked for.



   .. py:method:: release()

      Drops every series read.



.. py:function:: smooth(y, weight)

   Returns ``y`` smoothed by an exponential moving average of weight ``weight``, from 0 (none)
   to below 1, corrected for its start as by wandb. NaN are kept and skipped.


.. py:class:: Band

   Bases: :py:obj:`NamedTuple`


   The series of the runs of a group taken together, at positions ``x``: their mean, lowest
   and highest value, and the number of runs with a value there.


   .. py:attribute:: x
      :type:  numpy.ndarray


   .. py:attribute:: mean
      :type:  numpy.ndarray


   .. py:attribute:: low
      :type:  numpy.ndarray


   .. py:attribute:: high
      :type:  numpy.ndarray


   .. py:attribute:: count
      :type:  numpy.ndarray


.. py:function:: aggregate(series, buckets = BUCKETS)

   Returns the series of several runs taken together.

   With ``buckets`` None, as in a tag space, the runs line up on their positions: every
   position of any run, each run counting where it has a value. Otherwise, the values of each
   run are averaged within each of ``buckets`` equal buckets spanning the runs, positioned at
   their middles. The mean, lowest and highest value at a position are taken over the runs with
   a value there.


.. py:class:: Line

   Bases: :py:obj:`NamedTuple`


   What is drawn for a visible run, or for a group of visible runs.


   .. py:attribute:: label
      :type:  str


   .. py:attribute:: color
      :type:  PySide6.QtGui.QColor


   .. py:attribute:: runs
      :type:  list[spark.recording.run.Run]


   .. py:attribute:: group
      :type:  bool


.. py:class:: Selection(project, parent = None)

   Bases: :py:obj:`PySide6.QtCore.QObject`


   The runs of a project shown, their colors and their groups.

   A run is visible or not. Grouped by fields of the project, such as ``experiment``, the runs
   sharing their values form a group, drawn as one line: the mean of its visible runs over the
   band from their lowest to their highest value. A run of no group, when a field has no value
   for it, is drawn on its own. Colors are given to runs in the order of the project, and to
   groups in the order they are met. Emits `changed` when what is drawn changes.

   :param project: The runs.
   :type project: Project
   :param parent: Parent object.
   :type parent: QObject, optional

   .. attribute:: visible

      Paths of the runs shown.

      :type: set of str

   .. attribute:: group_by

      Fields grouped by, none for no groups.

      :type: tuple of str

   .. attribute:: members

      Whether the runs of a group are drawn too, faintly.

      :type: bool

   .. attribute:: smoothing

      Weight of the exponential moving average of every series, from 0 to 0.99.

      :type: float

   .. attribute:: space

      Space every series is drawn in: `NATURAL`, ``'step'``, ``'wall'`` or a tag.

      :type: str


   .. py:attribute:: changed


   .. py:attribute:: NEWEST
      :value: 10


      every run up to this number, else the newest ones.

      :type: Runs shown by default


   .. py:attribute:: project


   .. py:attribute:: visible
      :type:  set[str]


   .. py:attribute:: group_by
      :type:  tuple[str, ...]
      :value: ('experiment',)



   .. py:attribute:: members
      :value: False



   .. py:attribute:: smoothing
      :value: 0.0



   .. py:attribute:: space
      :value: 'natural'



   .. py:method:: set_visible(paths, visible)

      Shows or hides the runs at ``paths``.



   .. py:method:: show_only(paths)

      Shows the runs at ``paths``, and hides the others.



   .. py:method:: show_newest(count = NEWEST)

      Shows the ``count`` newest runs, and hides the others.



   .. py:method:: set_group_by(fields)

      Groups the runs by ``fields``, or not for none.



   .. py:method:: set_color(key, color)

      Sets the color of a run, by path, or of a group, by label.



   .. py:method:: set_settings(*, members = None, smoothing = None, space = None)

      Sets how the lines are drawn; the values not given are kept.



   .. py:method:: state()

      Returns what is shown and how, with runs by name, as an `Exploration` keeps it.



   .. py:method:: restore(state)

      Shows what ``state`` shows, as it shows it. Runs added to the directory since are shown;
      runs gone are left out.



   .. py:method:: color(key)

      Returns the color of a run, by path, or of a group, by label: the run colours of `THEME`
      in order, repeated past the last.



   .. py:method:: label(run)

      Returns the label of a run: its experiment and the name it is shown by (`Project.name`).



   .. py:method:: groups()

      Returns the groups of every run of the project, visible or not, as their labels and
      runs in the order of the project. A run of no group is a group of its own, labelled by
      an empty string.



   .. py:method:: lines()

      Returns what is drawn: a line per group with a visible run, and per visible run of no
      group, in the order of the project.



   .. py:method:: space_of(store, run, key)

      Returns the space a series of ``run`` is drawn in: its natural one, or the one chosen.



   .. py:method:: draw(store, line, key, space = None)

      Returns the space of a line for the series ``key``, its band, and the series of each of
      its runs, smoothed.

      A run's band is its series, as its mean, lowest and highest value. A group's runs line
      up in a tag space, and are averaged in `BUCKETS` buckets otherwise. The runs of a line
      are drawn in ``space``, or else in the space of its first run with the series.



.. py:function:: explorations_path()

   Returns the folder of the explorations the viewer keeps, next to the model library of the
   editor.


.. py:function:: within_runs(path)

   Returns whether the file ``path`` is within a run, or within a directory holding runs.


.. py:class:: Exploration(path)

   The state of the exploration of a directory of runs, kept in a file of its own.

   The viewer keeps one per directory in `explorations_path`, and saves others where asked, but
   never within a directory of runs: what the viewer writes cannot take the place of what a
   recorder wrote. The state is a dictionary of JSON values, written whole.

   :param path: The file.
   :type path: str or path-like

   .. attribute:: path

      The file.

      :type: pathlib.Path


   .. py:attribute:: VERSION
      :value: 1


      Version of the state written, read back only by the same version.


   .. py:attribute:: path


   .. py:method:: of(root)
      :classmethod:


      Returns the exploration the viewer keeps for the directory of runs ``root``.



   .. py:method:: read()

      Returns the state written, or None when there is none, of another version, or unreadable.



   .. py:method:: write(state)

      Writes ``state``, in place of what the file held.

      :raises ValueError: When the file is within a run, or within a directory holding runs.
      :raises OSError: When the file cannot be written.



