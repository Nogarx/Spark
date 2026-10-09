spark.nn.interfaces.output.tanh
===============================

.. py:module:: spark.nn.interfaces.output.tanh


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.output.tanh.TanhIntegratorConfig
   spark.nn.interfaces.output.tanh.TanhIntegrator


Module Contents
---------------

.. py:class:: TanhIntegratorConfig

   Bases: :py:obj:`spark.nn.interfaces.output.base.OutputInterfaceConfig`


   Configuration for `TanhIntegrator`.

   :param saturation_freq: Population firing rate at which an output reaches 1.0, in Hz.
   :type saturation_freq: float, default 50.0
   :param tau: Decay constant of the integrator, in ms.
   :type tau: float, default 5.0
   :param smooth_trace: Use a rise-and-decay trace instead of a single exponential, which removes the step
                        each spike would otherwise leave in the output.
   :type smooth_trace: bool, default True


   .. py:attribute:: saturation_freq
      :type:  float


   .. py:attribute:: tau
      :type:  float


   .. py:attribute:: smooth
      :type:  bool


   .. py:attribute:: smooth_tau
      :type:  float


.. py:class:: TanhIntegrator(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.output.base.OutputInterface`


   Decodes spikes into a continuous signal.

   The input units are split into two groups. The spikes of each group are
   counted per step and passed through an exponential trace, so each output tracks the firing
   rate of its group.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: TanhIntegratorConfig, optional

   :Input Ports: **spikes** (*SpikeArray*) -- Spikes of the pool being read out.

   :Output Ports: **signal** (*FloatArray*) -- One value per group, of shape ``(num_outputs,)``. Reads 1.0 at ``saturation_freq``.

   .. rubric:: Notes

   The trace is scaled so that a group firing at ``saturation_freq`` reads 1.0. Rates above
   it are not clipped and read above 1.0.


   .. py:attribute:: config
      :type:  TanhIntegratorConfig


   .. py:attribute:: saturation_freq


   .. py:attribute:: tau


   .. py:attribute:: smooth


   .. py:attribute:: smooth_tau


   .. py:method:: build(x_spikes, y_spikes)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: __call__(x_spikes, y_spikes)

      Counts the spikes of each group and integrates them.

      :param spikes: Spikes of the pool being read out.
      :type spikes: SpikeArray

      :returns: Dictionary with one entry, ``signal``, of shape ``(num_outputs,)``.
      :rtype: OutputInterfaceOutput



