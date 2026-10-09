spark.recording.records
=======================

.. py:module:: spark.recording.records


Attributes
----------

.. autoapisummary::

   spark.recording.records.KINDS


Classes
-------

.. autoapisummary::

   spark.recording.records.Rows
   spark.recording.records.Reductions
   spark.recording.records.Record


Functions
---------

.. autoapisummary::

   spark.recording.records.split_key


Module Contents
---------------

.. py:data:: KINDS
   :value: ('traces', 'rasters', 'raw', 'summaries', 'deltas', 'snapshots')


   Data types holded by a `Record`.

.. py:function:: split_key(name)

   Splits the name of an array of a window file into its address, mode and part.

   ``'<address>@<mode>#<part>'`` gives ``(address, mode, part)``, the part being ``'t'`` for
   the steps of rows, or the name of a reduction. ``'raw:<name>#t'`` gives
   ``(name, 'raw', 't')``. A name without a part gives a part of None, and the other names,
   such as ``'span_t0'``, give ``(name, None, None)``.


.. py:class:: Rows

   Bases: :py:obj:`NamedTuple`


   The rows of a trace, a raster or a raw stream within a record, and their steps.

   Unpacks as ``t, values``.

   .. attribute:: t

      Step of each row, counted from the start of the record.

      :type: ndarray

   .. attribute:: values

      One row per step of ``t``. Rasters hold one bool per unit.

      :type: ndarray


   .. py:attribute:: t
      :type:  numpy.ndarray


   .. py:attribute:: values
      :type:  numpy.ndarray


   .. py:method:: __repr__()


.. py:class:: Reductions(values)

   Bases: :py:obj:`Mapping`\ [\ :py:obj:`str`\ , :py:obj:`Any`\ ]


   The reductions of a summary or a delta over one group, by name.

   Read as attributes (``summary.active_fraction``) or as keys (``summary['active_fraction']``).


   .. py:method:: __getitem__(name)


   .. py:method:: __getattr__(name)


   .. py:method:: __iter__()


   .. py:method:: __len__()


   .. py:method:: __dir__()


   .. py:method:: keys()

      D.keys() -> a set-like object providing a view on D's keys



   .. py:method:: values()

      D.values() -> an object providing a view on D's values



   .. py:method:: items()

      D.items() -> a set-like object providing a view on D's items



   .. py:method:: __repr__()


.. py:class:: Record(key, t0, t, *, traces = None, rasters = None, raw = None, summaries = None, deltas = None, snapshots = None)

   Object holding a collection of measurements recorded over one group of steps (`Measurements.group`).

   The data is held by type, addressed by the type of the probe:

   * `traces`, `rasters` and `raw` (by stream name): `Rows`, the rows and their steps.
   * `summaries` and `deltas`: `Reductions`, the reductions of the group by name.
   * `snapshots`: the value at the end of the group, an array.

   .. attribute:: key

      For a group by tag, the value of the tag: ``(value, n)`` the ``n``-th time the value
      comes back, None for the steps before the tag is first set. For a group of ``n`` steps,
      its number ``g``, of the steps ``[g * n, (g + 1) * n)``. Without a group, the first step
      of the stretch.

      :type: object

   .. attribute:: t0

      Step of the run at which the group starts, or the stretch.

      :type: int

   .. attribute:: t

      Steps recorded, counted from `t0`. A group recorded from its middle starts past 0.

      :type: ndarray

   .. attribute:: traces, rasters, raw



      :type: dict of str to Rows

   .. attribute:: summaries, deltas



      :type: dict of str to Reductions

   .. attribute:: snapshots



      :type: dict of str to ndarray

   .. rubric:: Notes

   Printed, or shown by a notebook, a record lists what it holds.

   .. seealso::

      :py:obj:`Run.read`
          The records of a set of measurements.

   .. rubric:: Examples

   >>> episode = run.read('episode')[20]
   >>> t, potential = episode.traces['first_pool.soma.potential']
   >>> episode.summaries['first_pool.soma:spikes'].active_fraction


   .. py:attribute:: key


   .. py:attribute:: t0


   .. py:attribute:: t


   .. py:attribute:: traces


   .. py:attribute:: rasters


   .. py:attribute:: raw


   .. py:attribute:: summaries


   .. py:attribute:: deltas


   .. py:attribute:: snapshots


   .. py:method:: from_arrays(key, t0, t, arrays)
      :classmethod:


      Builds a record from arrays named as in a window file.

      ``<address>@trace`` and ``<address>@raster`` with their steps in ``<...>#t``,
      ``raw:<name>`` with ``raw:<name>#t``, ``<address>@summary#<reduction>``,
      ``<address>@delta#<reduction>`` and ``<address>@snapshot``. The steps are counted from
      ``t0``.



   .. py:property:: steps
      :type: int


      Number of steps recorded.


   .. py:method:: __repr__()


   .. py:method:: __str__()


