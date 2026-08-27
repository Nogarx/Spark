spark.nn.interfaces.input.linear
================================

.. py:module:: spark.nn.interfaces.input.linear


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.input.linear.LinearSpikerConfig
   spark.nn.interfaces.input.linear.LinearSpiker


Module Contents
---------------

.. py:class:: LinearSpikerConfig

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterfaceConfig`


   Configuration for `LinearSpiker`.

   :param tau: Decay constant of the internal potential, in ms.
   :type tau: float, default 100.0
   :param cd: Refractory period, in ms. Bounds the firing rate at ``1000 / cd`` Hz.
   :type cd: float, default 2.0
   :param max_freq: Firing rate reached by an input of 1.0, in Hz.
   :type max_freq: float, default 100.0


   .. py:attribute:: tau
      :type:  float


   .. py:attribute:: cd
      :type:  float


   .. py:attribute:: max_freq
      :type:  float


.. py:class:: LinearSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Deterministic rate encoding of a continuous signal.

   Each unit integrates its input into a potential and fires when the potential reaches a
   threshold, after which it resets and waits out a refractory period. The same input always
   produces the same spike train, which makes a run repeatable without fixing a seed.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: LinearSpikerConfig, optional

   :Input Ports: **signal** (*FloatArray*) -- Value to encode, expected in ``[0, 1]``.

   :Output Ports: **spikes** (*SpikeArray*) -- One spike train per input, of the same shape as ``signal``.

   .. rubric:: Notes

   The input is gated off during the refractory period, so ``cd`` bounds the firing rate at
   ``1000 / cd`` Hz regardless of ``max_freq``.

   .. seealso::

      :py:obj:`PoissonSpiker`
          Stochastic encoding of the same signal.


   .. py:attribute:: config
      :type:  LinearSpikerConfig


   .. py:attribute:: tau


   .. py:attribute:: cd


   .. py:attribute:: max_freq


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

      :param signal: Value to encode, expected in ``[0, 1]``.
      :type signal: FloatArray

      :returns: Dictionary with one entry, ``spikes``, of the same shape as ``signal``.
      :rtype: InputInterfaceOutput



