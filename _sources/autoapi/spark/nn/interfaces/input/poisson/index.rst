spark.nn.interfaces.input.poisson
=================================

.. py:module:: spark.nn.interfaces.input.poisson


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.input.poisson.PoissonSpikerConfig
   spark.nn.interfaces.input.poisson.PoissonSpiker


Module Contents
---------------

.. py:class:: PoissonSpikerConfig

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterfaceConfig`


   Configuration for `PoissonSpiker`.

   :param max_freq: Firing rate reached by an input of 1.0, in Hz.
   :type max_freq: float, default 100.0


   .. py:attribute:: max_freq
      :type:  float


.. py:class:: PoissonSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Stochastic rate encoding of a continuous signal.

   Each unit emits a spike on a step with a probability proportional to its input, drawn
   independently every step. Repeated presentations of the same input give different spike
   trains with the same expected rate.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: PoissonSpikerConfig, optional

   :Input Ports: **signal** (*FloatArray*) -- Value to encode, expected in ``[0, 1]``.

   :Output Ports: **spikes** (*SpikeArray*) -- One spike train per input, of the same shape as ``signal``.

   .. rubric:: Notes

   .. math::
       P(s_i = 1) = \frac{f_{\max} \Delta t}{1000} x_i

   with ``dt`` in ms. Inputs above 1.0 saturate at one spike per step, and the encoding is
   only linear while :math:`f_{\max} \Delta t / 1000 \le 1`.

   .. seealso::

      :py:obj:`LinearSpiker`
          Deterministic encoding of the same signal.


   .. py:attribute:: config
      :type:  PoissonSpikerConfig


   .. py:attribute:: max_freq


   .. py:method:: build(signal)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: __call__(signal)

      Encodes the signal as spikes.

      :param signal: Value to encode, expected in ``[0, 1]``.
      :type signal: FloatArray

      :returns: Dictionary with one entry, ``spikes``, of the same shape as ``signal``.
      :rtype: InputInterfaceOutput



