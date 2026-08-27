spark.nn.interfaces.input.topological
=====================================

.. py:module:: spark.nn.interfaces.input.topological


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.input.topological.TopologicalSpikerConfig
   spark.nn.interfaces.input.topological.TopologicalPoissonSpikerConfig
   spark.nn.interfaces.input.topological.TopologicalPoissonSpiker
   spark.nn.interfaces.input.topological.TopologicalLinearSpikerConfig
   spark.nn.interfaces.input.topological.TopologicalLinearSpiker


Module Contents
---------------

.. py:class:: TopologicalSpikerConfig

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterfaceConfig`


   Base configuration for topological spikers.

   :param glue: Per-dimension flag marking whether the two ends of that dimension are identified. A
                glued dimension is encoded on a circle, so its two extremes excite the same units.
   :type glue: jax.Array, default 0
   :param mins: Lower end of the input range, per dimension or as a single value.
   :type mins: jax.Array, default 0
   :param maxs: Upper end of the input range, per dimension or as a single value.
   :type maxs: jax.Array, default 1
   :param resolution: Number of units each input dimension is spread over.
   :type resolution: int, default 64
   :param sigma: Width of the activity bump, as the standard deviation of a Gaussian in the target
                 space.
   :type sigma: float, default 1/32


   .. py:attribute:: glue
      :type:  int | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: mins
      :type:  int | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: maxs
      :type:  int | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: resolution
      :type:  int


   .. py:attribute:: sigma
      :type:  float


.. py:class:: TopologicalPoissonSpikerConfig

   Bases: :py:obj:`TopologicalSpikerConfig`, :py:obj:`spark.nn.interfaces.input.poisson.PoissonSpikerConfig`


   Configuration for `TopologicalPoissonSpiker`.


.. py:class:: TopologicalPoissonSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Place-coded stochastic encoding of a continuous signal.

   Each input dimension is spread over ``resolution`` units laid out along that dimension.
   A value excites the units near its position under a Gaussian bump of width ``sigma``, and
   those units then fire as independent Poisson processes. Nearby values excite overlapping
   populations, which a per-unit rate code does not give.

   Setting ``glue`` for a dimension identifies its two ends, so that dimension is encoded on
   a circle. (e.g. an angular signal should reconciliate units placed at 0 and 2π positions)

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: TopologicalPoissonSpikerConfig

   :Input Ports: **signal** (*FloatArray*) -- Value to encode, expected in ``[mins, maxs]``.

   :Output Ports: **spikes** (*SpikeArray*) -- Place-coded spike trains, of shape ``signal.shape + (resolution,)``.

   .. seealso::

      :py:obj:`PoissonSpiker`
          One unit per input, without the place code.

      :py:obj:`TopologicalLinearSpiker`
          The same place code, deterministically encoded.


   .. py:attribute:: config
      :type:  TopologicalPoissonSpikerConfig


   .. py:attribute:: resolution


   .. py:attribute:: max_freq


   .. py:attribute:: sigma


   .. py:method:: build(signal)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: __call__(signal)

      Encodes the signal as spikes.

      :param signal: Value to encode, expected in ``[mins, maxs]``.
      :type signal: FloatArray

      :returns: Dictionary with one entry, ``spikes``, of shape ``signal.shape + (resolution,)``.
      :rtype: InputInterfaceOutput



.. py:class:: TopologicalLinearSpikerConfig

   Bases: :py:obj:`TopologicalSpikerConfig`, :py:obj:`spark.nn.interfaces.input.linear.LinearSpikerConfig`


   Configuration for `TopologicalLinearSpiker`.

   Union of `TopologicalSpikerConfig` and `LinearSpikerConfig`. It declares no field of its
   own.


.. py:class:: TopologicalLinearSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Place-coded deterministic encoding of a continuous signal.

   Each input dimension is spread over ``resolution`` units laid out along that dimension.
   A value excites the units near its position under a Gaussian bump of width ``sigma``, and
   those units then fire as independent Poisson processes. Nearby values excite overlapping
   populations, which a per-unit rate code does not give.

   Setting ``glue`` for a dimension identifies its two ends, so that dimension is encoded on
   a circle. (e.g. an angular signal should reconciliate units placed at 0 and 2π positions)

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: TopologicalLinearSpikerConfig, optional

   :Input Ports: **signal** (*FloatArray*) -- Value to encode, expected in ``[mins, maxs]``.

   :Output Ports: **spikes** (*SpikeArray*) -- Place-coded spike trains, of shape ``signal.shape + (resolution,)``.

   .. seealso::

      :py:obj:`TopologicalPoissonSpiker`
          The same place code, stochastically encoded.

      :py:obj:`LinearSpiker`
          One unit per input, without the place code.


   .. py:attribute:: config
      :type:  TopologicalLinearSpikerConfig


   .. py:attribute:: resolution


   .. py:attribute:: tau


   .. py:attribute:: cd


   .. py:attribute:: max_freq


   .. py:attribute:: sigma


   .. py:method:: build(signal)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: __call__(signal)

      Encodes the signal as spikes.

      :param signal: Value to encode, expected in ``[mins, maxs]``.
      :type signal: FloatArray

      :returns: Dictionary with one entry, ``spikes``, of shape ``signal.shape + (resolution,)``.
      :rtype: InputInterfaceOutput



