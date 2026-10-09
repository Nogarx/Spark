spark.recording.probe
=====================

.. py:module:: spark.recording.probe


Attributes
----------

.. autoapisummary::

   spark.recording.probe.CALL
   spark.recording.probe.STEP_PROBES
   spark.recording.probe.BOUNDARY_PROBES
   spark.recording.probe.GROUPED_PROBES


Classes
-------

.. autoapisummary::

   spark.recording.probe.ProbeMode
   spark.recording.probe.SummaryReduction
   spark.recording.probe.DeltaReduction
   spark.recording.probe.Probe
   spark.recording.probe.SummaryProbe
   spark.recording.probe.TraceProbe
   spark.recording.probe.RasterProbe
   spark.recording.probe.SnapshotProbe
   spark.recording.probe.DeltaProbe


Functions
---------

.. autoapisummary::

   spark.recording.probe.address
   spark.recording.probe.probe_addresses
   spark.recording.probe.expand
   spark.recording.probe.validate
   spark.recording.probe.resolve
   spark.recording.probe.get_recorded_array
   spark.recording.probe.check_units


Module Contents
---------------

.. py:data:: CALL
   :value: '__call__'


   Path segment naming the inputs of a controller.

.. py:class:: ProbeMode

   Bases: :py:obj:`enum.StrEnum`


   How a probe records its value.

   Each probe is associated with a specific `Probe.mode`.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: SUMMARY
      :value: 'summary'


      reductions of the value over the units and over each group of steps.

      :type: `SummaryProbe`


   .. py:attribute:: TRACE
      :value: 'trace'


      the value on every step.

      :type: `TraceProbe`


   .. py:attribute:: RASTER
      :value: 'raster'


      whether each unit is active, on every step.

      :type: `RasterProbe`


   .. py:attribute:: SNAPSHOT
      :value: 'snapshot'


      the value of an attribute at the end of each group of steps.

      :type: `SnapshotProbe`


   .. py:attribute:: DELTA
      :value: 'delta'


      reductions of the change of an attribute over each group of steps.

      :type: `DeltaProbe`


.. py:class:: SummaryReduction

   Bases: :py:obj:`enum.StrEnum`


   `SummaryProbe` requested operations (reductions) on the Probes.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: MEAN
      :value: 'mean'


      Mean over the units and the steps.


   .. py:attribute:: STD
      :value: 'std'


      Standard deviation over the units and the steps.


   .. py:attribute:: MIN
      :value: 'min'


      Minimum over the units and the steps.


   .. py:attribute:: MAX
      :value: 'max'


      Maximum over the units and the steps.


   .. py:attribute:: ACTIVE_FRACTION
      :value: 'active_fraction'


      Fraction of the units active, averaged over the steps. For spikes, the firing rate per step.


   .. py:attribute:: ACTIVE_FRACTION_PER_UNIT
      :value: 'active_fraction_per_unit'


      Fraction of the steps on which each unit is active. One value per unit.


   .. py:attribute:: INACTIVE_UNIT_FRACTION
      :value: 'inactive_unit_fraction'


      Fraction of the units inactive in the group.


   .. py:attribute:: HIST
      :value: 'hist'


      Counts of the values of the units and the steps in ``bins`` equal bins over ``range``.


   .. py:property:: scalar
      :type: bool


      Whether the reduction gives one number per group of steps. ``ACTIVE_FRACTION_PER_UNIT``
      and ``HIST`` give an array. A `Recorder` also writes the scalar reductions to the
      scalars of the run.


.. py:class:: DeltaReduction

   Bases: :py:obj:`enum.StrEnum`


   `DeltaProbe` requested operations (reductions) on the Probes. Unlike `SummaryProbe`, `DeltaProbe`
   compute differences of an attribute over each group of steps.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: FULL
      :value: 'full'


      The change of every unit.


   .. py:attribute:: NORM
      :value: 'norm'


      Euclidean norm of the change.


   .. py:attribute:: MEAN_ABS
      :value: 'mean_abs'


      Mean absolute change over the units.


   .. py:property:: scalar
      :type: bool


      Whether the reduction gives one number per group of steps. ``FULL`` gives an array. A
      `Recorder` also writes the scalar reductions to the scalars of the run.


.. py:class:: Probe

   Bases: :py:obj:`abc.ABC`


   Specification for a measurement, denoted by a port address and the measurement operator:
   `SummaryProbe`, `TraceProbe`, `RasterProbe`, `SnapshotProbe` or `DeltaProbe`.

   :param address: Address of the value. ``path`` is the dotted chain of module names from the root
                   controller.

                   * ``path:port``: an output port of a module.
                   * ``path.__call__:port``: an input port of a controller, ``__call__:port`` for the root.
                   * ``path.name``: an attribute of a module.

                   A pattern, as ``*_excitatory.soma:spikes`` or ``**.soma.potential``, stands for a probe
                   of every address it matches, with the same fields (see `spark.core.addresses`). It is
                   matched when the probes are checked against a built model: by `validate`, and by a
                   `Recorder` given the model.
   :type address: str

   .. attribute:: mode

      Mode of the class of the probe.

      :type: ProbeMode

   .. attribute:: kind

      ``'port'`` or ``'attribute'``.

      :type: str

   .. attribute:: path

      Path of the module producing the port or holding the attribute.

      :type: tuple of str

   .. attribute:: name

      Port or attribute name.

      :type: str

   .. attribute:: key

      Name of the entry of the probe in the records, ``address@mode``.

      :type: str

   :raises ValueError: When the address is malformed, or a field is invalid.

   .. rubric:: Notes

   Probes are frozen and hashable. Probes of the same class with equal fields compare and hash
   equal. A tuple of probes can be a static argument of a jitted function.

   Ports are read as the module produces them. Attributes are read at the end of each step,
   after every module of the step ran.

   .. seealso::

      :py:obj:`Measurements`
          A named set of probes recorded together.

      :py:obj:`get_probe_targets`
          Lists the values of a controller that a probe can address.

      :py:obj:`validate`
          Checks probes against a controller.

      :py:obj:`Recorder`
          Records the measurements asked for and writes them to a run.


   .. py:attribute:: address
      :type:  str


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: kind
      :type:  str


   .. py:attribute:: path
      :type:  tuple[str, ...]


   .. py:attribute:: name
      :type:  str


   .. py:method:: __post_init__()


   .. py:method:: __eq__(other)


   .. py:method:: __hash__()


   .. py:method:: __reduce__()


   .. py:property:: key
      :type: str



   .. py:property:: per_step
      :type: bool


      Whether the probe is read on every step (`STEP_PROBES`).


   .. py:method:: to_dict()

      Returns the mode and the fields of the probe as JSON types.

      :returns: ``mode``, and the keyword arguments of the class, with lists in place of tuples.
      :rtype: dict



   .. py:method:: from_dict(data)
      :classmethod:


      Rebuilds a probe from `to_dict`.

      :param data: Mode and fields of a probe, as `to_dict` gives them.
      :type data: dict

      :returns: A probe of the class of the mode.
      :rtype: Probe

      :raises ValueError: When the mode is unknown, or is not the mode of the class called.



.. py:class:: SummaryProbe

   Bases: :py:obj:`Probe`


   Reductions of a value over the units and over each group of steps.

   :param address: Address of the value, as in `Probe`.
   :type address: str
   :param reduce: Reductions, as `SummaryReduction` members or their names.

                  * ``MEAN``, ``STD``, ``MIN``, ``MAX``: over the units and the steps of the group.
                  * ``ACTIVE_FRACTION``: fraction of the units active, averaged over the steps.
                  * ``ACTIVE_FRACTION_PER_UNIT``: fraction of the steps on which each unit is active.
                  * ``INACTIVE_UNIT_FRACTION``: fraction of the units active on none of the steps.
                  * ``HIST``: counts in ``bins`` equal bins over ``range``.
   :type reduce: SummaryReduction or str or sequence of them, default ('mean', 'std', 'min', 'max')
   :param bins: Number of bins of ``HIST``. Ignored without it.
   :type bins: int, default 32
   :param range: Lower and upper edges of ``HIST``, finite as float32. Required by ``HIST`` and ignored
                 without it. Values outside the range, and NaN, are not counted.
   :type range: tuple of float, optional
   :param group: How the steps are split into groups, one record per group. Set by the `Measurements`
                 holding the probe; see `Measurements` for how a group is recorded.

                 * A number of steps ``n``: groups of ``n`` steps on the steps of the run, ``[0, n)``,
                   ``[n, 2n)``, and so on.
                 * The name of a tag, such as ``'episode'``: a new group each time the tag takes a
                   different value (`Recorder.tag`).
   :type group: int or str, optional

   .. rubric:: Notes

   A unit is an entry of the value. A unit is active on a step when its value is nonzero.

   Means, spreads and fractions are computed in float32 and counts in int32, whatever the dtype
   of the value. Histogram counts are int64. ``MIN`` and ``MAX`` keep integer dtypes, and are
   float32 for floating values and uint8 for bool.

   A NaN in a value makes ``MEAN``, ``STD``, ``MIN`` and ``MAX`` NaN, and counts as active for
   the fractions. An infinity makes ``MIN`` or ``MAX`` infinite, and ``MEAN`` and ``STD`` NaN.
   A float32 value near 1e19 or beyond can overflow ``STD``.

   Summaries keep device memory in proportion to the value and to the groups a call
   touches. A histogram is counted after the call from the values of every step, or step by
   step when those values take more than `SETTINGS.histogram_rows_limit` bytes.

   Histogram edges are float32, as in ``numpy.histogram`` of float32 values. XLA on the CPU
   reads float32 subnormals (below about 1.2e-38 in magnitude) as zero in fractions and
   histograms. GPUs do not read them as zero.

   .. rubric:: Examples

   >>> SummaryProbe('first_pool:out_spikes', reduce=(
   ...     SummaryReduction.ACTIVE_FRACTION, SummaryReduction.INACTIVE_UNIT_FRACTION,
   ... ))
   >>> SummaryProbe('first_pool.soma.potential', reduce=SummaryReduction.HIST,
   ...              range=(-80.0, 40.0))


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: reduce
      :type:  tuple[SummaryReduction | str, ...]
      :value: ('mean', 'std', 'min', 'max')



   .. py:attribute:: bins
      :type:  int
      :value: 32



   .. py:attribute:: range
      :type:  tuple[float, float] | None
      :value: None



   .. py:attribute:: group
      :type:  int | str | None
      :value: None



.. py:class:: TraceProbe

   Bases: :py:obj:`Probe`


   The value on every step.

   :param address: Address of the value, as in `Probe`.
   :type address: str
   :param units: Flat indices of the units kept. All units when omitted.
   :type units: sequence of int, optional
   :param stride: Keeps the steps ``t`` of the run with ``t % stride == 0``.
   :type stride: int, default 1

   .. rubric:: Notes

   A unit is an entry of the value. A trace keeps the dtype of the value. A trace of an
   attribute holds the value the next step starts from.

   A trace keeps every step of the call it records. With a stride, it keeps only the steps of
   the stride when every step would take more than `SETTINGS.spaced_rows_limit`
   bytes.

   .. rubric:: Examples

   >>> TraceProbe('first_pool.soma.potential', units=range(64))
   >>> TraceProbe('__call__:signal', stride=10)


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: units
      :type:  tuple[int, ...] | None
      :value: None



   .. py:attribute:: stride
      :type:  int
      :value: 1



.. py:class:: RasterProbe

   Bases: :py:obj:`Probe`


   Whether each unit is active, on every step.

   :param address: Address of the value, as in `Probe`.
   :type address: str
   :param units: Flat indices of the units kept. All units when omitted.
   :type units: sequence of int, optional
   :param stride: Keeps the steps ``t`` of the run with ``t % stride == 0``.
   :type stride: int, default 1

   .. rubric:: Notes

   A unit is an entry of the value. A unit is active on a step when its value is nonzero. A
   NaN counts as active. A float16 or bfloat16 value is rounded to its dtype before the test.

   A raster is moved and stored as bits, and read as bool. It keeps every step of the call it
   records. With a stride, it keeps only the steps of the stride when every step would take
   more than `SETTINGS.spaced_rows_limit` bytes.

   XLA on the CPU reads float32 subnormals (below about 1.2e-38 in magnitude) as zero. GPUs do
   not read them as zero.

   .. rubric:: Examples

   >>> RasterProbe('first_pool:out_spikes')
   >>> RasterProbe('first_pool:out_spikes', units=range(256))


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: units
      :type:  tuple[int, ...] | None
      :value: None



   .. py:attribute:: stride
      :type:  int
      :value: 1



.. py:class:: SnapshotProbe

   Bases: :py:obj:`Probe`


   The value of an attribute at the end of each group of steps.

   :param address: Address of an attribute, as in `Probe`.
   :type address: str
   :param units: Flat indices of the units kept. All units when omitted.
   :type units: sequence of int, optional
   :param group: How the steps are split into groups, one record per group, as for `SummaryProbe`: a
                 number of steps or the name of a tag. Set by the `Measurements` holding the probe.
   :type group: int or str, optional

   .. rubric:: Notes

   The value is read after the last step of each group. Until the group ends, a snapshot with a
   group keeps the value after each call on the device, and, for a call crossing the end of a
   group, the value at that end.

   .. rubric:: Examples

   >>> SnapshotProbe('first_pool.synapses.kernel')


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: units
      :type:  tuple[int, ...] | None
      :value: None



   .. py:attribute:: group
      :type:  int | str | None
      :value: None



.. py:class:: DeltaProbe

   Bases: :py:obj:`Probe`


   Reductions of the change of an attribute over each group of steps.

   :param address: Address of an attribute, as in `Probe`.
   :type address: str
   :param reduce: Reductions, as `DeltaReduction` members or their names.

                  * ``FULL``: the change of every unit.
                  * ``NORM``: the Euclidean norm of the change.
                  * ``MEAN_ABS``: the mean absolute change over the units.
   :type reduce: DeltaReduction or str or sequence of them, default ('norm',)
   :param group: How the steps are split into groups, one record per group, as for `SummaryProbe`: a
                 number of steps or the name of a tag. Set by the `Measurements` holding the probe.
   :type group: int or str, optional

   .. rubric:: Notes

   The change is the value after the last step of the group minus the value before its first
   step recorded, computed in float32. Until the group ends, a delta with a group keeps the
   value before and after each call on the device, and, for a call crossing the end of a group,
   the value at that end.

   .. rubric:: Examples

   >>> DeltaProbe('first_pool.synapses.kernel',
   ...            reduce=(DeltaReduction.NORM, DeltaReduction.MEAN_ABS))


   .. py:attribute:: mode
      :type:  ClassVar[ProbeMode]


   .. py:attribute:: reduce
      :type:  tuple[DeltaReduction | str, ...]
      :value: ('norm',)



   .. py:attribute:: group
      :type:  int | str | None
      :value: None



.. py:function:: address(path, name, kind)

   Returns the address of the port or attribute ``name`` of the module at ``path``.

   :param path: Module path. Ends with `CALL` for the inputs of a controller.
   :type path: sequence of str
   :param name: Port or attribute name.
   :type name: str
   :param kind: ``'port'`` or ``'attribute'``.
   :type kind: str

   :returns: ``path:name`` for a port, ``path.name`` for an attribute.
   :rtype: str


.. py:function:: probe_addresses(controller)

   Returns the address of every value of a built controller that a probe can read.

   :param controller: A built controller.
   :type controller: Controller

   :returns: The inputs of every controller (``path.__call__:port``), the output ports of every module
             of a controller (``path:port``), and every attribute holding an array (``path.name``).
   :rtype: tuple of str

   .. seealso::

      :py:obj:`get_probe_targets`
          Lists the same values, with their shapes and dtypes, from example inputs.


.. py:function:: expand(controller, probes)

   Returns the probes with every pattern replaced by a probe of each address it matches.

   :param controller: A built controller.
   :type controller: Controller
   :param probes: Probes, whose addresses may be patterns (see `spark.core.addresses`).
   :type probes: iterable of Probe

   :returns: The probes in the order given, each pattern replaced by probes with the same fields, one for
             each address it matches, in the order of `probe_addresses`.
   :rtype: tuple of Probe

   :raises ValueError: When a pattern matches nothing.


.. py:function:: validate(controller, probes)

   Checks that every probe is a valid probe (targets an existing variable within the controller).

   The address of each probe must name a module, port or attribute of the controller. ``units`` must
   be within the size of the value, and a `SummaryProbe` or a `DeltaProbe` must have entries to reduce.
   An attribute must hold an array. A pattern is checked as the probes of the addresses it matches,
   and must match one at least.

   :param controller: A built controller, called at least once.
   :type controller: Controller
   :param probes: Probes to check, one per key.
   :type probes: iterable of Probe

   :raises TypeError: When ``controller`` is not a controller, or a probe is not a `Probe`.
   :raises ValueError: When the controller is not built, two probes share a key, or a probe does not fit the controller.
       The message lists the names available where the address failed.

   .. rubric:: Notes

   The sizes of controller inputs and of the ports of nested controllers are not known before
   the controller is traced, and are not checked. `Runner` traces every probe of its recorder before
   its first call.

   .. seealso::

      :py:obj:`get_probe_targets`
          Lists the values of a controller that a probe can address.


.. py:function:: resolve(controller, path, address = None)

   Retrieves the module pointed by ``path``, within ``controller``.

   :param controller: Object the path starts from, such as a controller.
   :type controller: object
   :param path: Attribute names, followed one after the other.
   :type path: sequence of str
   :param address: Address named in the error message. Defaults to the path.
   :type address: str, optional

   :returns: The object at the end of the path.
   :rtype: object

   :raises ValueError: When a segment is not found. The message lists the names available at that level.


.. py:function:: get_recorded_array(value)

   Returns the array a probe reduces.

   :param value: Value read from the controller.
   :type value: SparkPayload or Variable or array

   :rtype: array

   :raises TypeError: When the value holds no array.


.. py:function:: check_units(size, units, address)

   Checks that ``units`` are indices within a value of ``size`` entries.

   :param size: Number of entries of the value.
   :type size: int
   :param units: Flat indices, or None for all units.
   :type units: tuple of int or None
   :param address: Address named in the error message.
   :type address: str

   :raises ValueError: When an index of ``units`` is not below ``size``.


.. py:data:: STEP_PROBES

   Probes read on every step.

.. py:data:: BOUNDARY_PROBES

   Probes read before and after the steps of each group, not on every step. Attributes only.

.. py:data:: GROUPED_PROBES

   Probes giving one record per group of steps.

