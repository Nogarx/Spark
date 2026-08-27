spark.nn.interfaces.output.exponential
======================================

.. py:module:: spark.nn.interfaces.output.exponential


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.output.exponential.ExponentialIntegratorConfig
   spark.nn.interfaces.output.exponential.ExponentialIntegrator


Module Contents
---------------

.. py:class:: ExponentialIntegratorConfig

   Bases: :py:obj:`spark.nn.interfaces.output.base.OutputInterfaceConfig`


   Configuration for `ExponentialIntegrator`.

   :param num_outputs: Number of output signals. The input units are split equally among them.
   :type num_outputs: int
   :param saturation_freq: Population firing rate at which an output reaches 1.0, in Hz.
   :type saturation_freq: float, default 50.0
   :param tau: Decay constant of the integrator, in ms.
   :type tau: float, default 5.0
   :param output_map: Index of the output each input unit contributes to. Must have one entry per input
                      unit. Assigned automatically when None.
   :type output_map: jax.Array or None, default None
   :param shuffle: Draw the automatic assignment at random rather than in order. Only read when
                   ``output_map`` is None.
   :type shuffle: bool, default True
   :param smooth_trace: Use a rise-and-decay trace instead of a single exponential, which removes the step
                        each spike would otherwise leave in the output.
   :type smooth_trace: bool, default True


   .. py:attribute:: num_outputs
      :type:  int


   .. py:attribute:: saturation_freq
      :type:  float


   .. py:attribute:: tau
      :type:  float


   .. py:attribute:: output_map
      :type:  jax.Array | None


   .. py:attribute:: shuffle
      :type:  bool


   .. py:attribute:: smooth_trace
      :type:  bool


.. py:class:: ExponentialIntegrator(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.output.base.OutputInterface`


   Decodes spikes into a continuous signal.

   The input units are split into ``num_outputs`` groups. The spikes of each group are
   counted per step and passed through an exponential trace, so each output tracks the firing
   rate of its group.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ExponentialIntegratorConfig, optional

   :Input Ports: **spikes** (*SpikeArray*) -- Spikes of the pool being read out.

   :Output Ports: **signal** (*FloatArray*) -- One value per group, of shape ``(num_outputs,)``. Reads 1.0 at ``saturation_freq``.

   .. rubric:: Notes

   The trace is scaled so that a group firing at ``saturation_freq`` reads 1.0. Rates above
   it are not clipped and read above 1.0.

   Assignment is by ``output_map`` when given. Otherwise the units are dealt out evenly,
   shuffled when ``shuffle`` is set, which spreads each output over the pool rather than
   over one contiguous block of it.


   .. py:attribute:: config
      :type:  ExponentialIntegratorConfig


   .. py:attribute:: num_outputs


   .. py:attribute:: saturation_freq


   .. py:attribute:: tau


   .. py:attribute:: shuffle


   .. py:attribute:: smooth_trace


   .. py:method:: build(spikes)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: __call__(spikes)

      Counts the spikes of each group and integrates them.

      :param spikes: Spikes of the pool being read out.
      :type spikes: SpikeArray

      :returns: Dictionary with one entry, ``signal``, of shape ``(num_outputs,)``.
      :rtype: OutputInterfaceOutput



