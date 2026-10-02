spark.nn.interfaces.control.trace
=================================

.. py:module:: spark.nn.interfaces.control.trace


Attributes
----------

.. autoapisummary::

   spark.nn.interfaces.control.trace.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.control.trace.SignalTraceOutput
   spark.nn.interfaces.control.trace.SignalTraceConfig
   spark.nn.interfaces.control.trace.SignalTrace
   spark.nn.interfaces.control.trace.SignalAccumulatorConfig
   spark.nn.interfaces.control.trace.SignalAccumulator
   spark.nn.interfaces.control.trace.SignalAverageConfig
   spark.nn.interfaces.control.trace.SignalAverage


Module Contents
---------------

.. py:class:: SignalTraceOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a signal trace.

   .. attribute:: output

      The trace after this step, plus whatever arrived on the ``trace`` port.

      :type: FloatArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: output
      :type:  spark.core.payloads.FloatArray


.. py:class:: SignalTraceConfig

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterfaceConfig`


   Base configuration for signal traces.

   :param tau: Decay constant of the trace, in ms.
   :type tau: float or jax.Array, default 10.0
   :param base: Value the trace decays towards.
   :type base: float or jax.Array, default 0.0


   .. py:attribute:: tau
      :type:  float | jax.Array


   .. py:attribute:: base
      :type:  float | jax.Array


.. py:data:: ConfigT

.. py:class:: SignalTrace(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for exponentially decaying traces of a signal.

   A signal that arrives at one rate and is read at another is held here rather than by the
   caller: the value is delivered as it happens and the trace is what the rest of the graph
   reads. This keeps the decay constant inside the model, where ``dt`` is known.

   Subclasses differ in the gain applied to the incoming signal, and therefore in which
   quantity is preserved when ``dt`` changes:

   * `SignalAccumulator` preserves the area of a signal delivered once.
   * `SignalAverage` preserves the level of a signal delivered on every step.

   Neither preserves both. Picking the other one rescales the trace rather than raising.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: SignalTraceConfig

   :Input Ports: * **signal** (*FloatArray*) -- Value delivered on this step.
                 * **trace** (*FloatArray, optional*) -- Trace arriving from an upstream `SignalTrace`, added to this one.

   :Output Ports: **output** (*FloatArray*) -- The trace after this step, plus whatever arrived on ``trace``.

   .. rubric:: Notes

   Traces chain. Feeding the output of one into the ``trace`` port of the next adds them, so
   a port carrying signals of both kinds takes one trace of each.

   .. seealso::

      :py:obj:`SignalAccumulator`
          Area-preserving trace.

      :py:obj:`SignalAverage`
          Level-preserving trace.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: build(signal, trace = None)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(signal, trace = None)

      Advances the trace with the incoming signal.

      :param signal: Value delivered on this step.
      :type signal: FloatArray
      :param trace: Trace arriving from an upstream `SignalTrace`, added to this one.
      :type trace: FloatArray, optional

      :returns: Dictionary with one entry, ``output``, the trace after this step.
      :rtype: SignalTraceOutput



.. py:class:: SignalAccumulatorConfig

   Bases: :py:obj:`SignalTraceConfig`


   Configuration for `SignalAccumulator`.

   :param tau: Decay constant of the trace, in ms.
   :type tau: float or jax.Array, default 10.0
   :param base: Value the trace decays towards.
   :type base: float or jax.Array, default 0.0


.. py:class:: SignalAccumulator(config = None, **kwargs)

   Bases: :py:obj:`SignalTrace`


   Area-preserving trace of a signal.

   The signal enters the trace unscaled, so a one-off delivery leaves a jump of its full
   magnitude that then decays. Use this for events: a reward, a cue, anything delivered on
   the step it happens and on no other.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: SignalAccumulatorConfig

   :Input Ports: * **signal** (*FloatArray*) -- Value delivered on this step.
                 * **trace** (*FloatArray, optional*) -- Trace arriving from an upstream `SignalTrace`, added to this one.

   :Output Ports: **output** (*FloatArray*) -- The trace after this step, plus whatever arrived on ``trace``.

   .. rubric:: Notes

   With :math:`\lambda = 1 - \exp(-\Delta t / \tau)`,

   .. math::
       T \leftarrow T + \lambda (T_{\mathrm{base}} - T) + x

   A signal delivered on every step accumulates to :math:`x / \lambda`, which grows as
   ``dt`` shrinks. Use `SignalAverage` for that case.

   .. seealso::

      :py:obj:`SignalAverage`
          Level-preserving trace.


   .. py:attribute:: config
      :type:  SignalAccumulatorConfig


.. py:class:: SignalAverageConfig

   Bases: :py:obj:`SignalTraceConfig`


   Configuration for `SignalAverage`.

   :param tau: Decay constant of the trace, in ms.
   :type tau: float or jax.Array, default 10.0
   :param base: Value the trace decays towards.
   :type base: float or jax.Array, default 0.0


.. py:class:: SignalAverage(config = None, **kwargs)

   Bases: :py:obj:`SignalTrace`


   Level-preserving trace of a signal.

   The signal is scaled by the decay rate as it enters the trace, so a constant input settles
   at that constant rather than at a multiple of it. Use this for signals present on every
   step: a firing rate, a sensor reading, a running error.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: SignalAverageConfig, optional

   :Input Ports: * **signal** (*FloatArray*) -- Value delivered on this step.
                 * **trace** (*FloatArray, optional*) -- Trace arriving from an upstream `SignalTrace`, added to this one.

   :Output Ports: **output** (*FloatArray*) -- The trace after this step, plus whatever arrived on ``trace``.

   .. rubric:: Notes

   With :math:`\lambda = 1 - \exp(-\Delta t / \tau)`,

   .. math::
       T \leftarrow T + \lambda (T_{\mathrm{base}} - T) + \lambda x

   which is the exponential moving average of :math:`x`. A one-off delivery leaves a jump of
   :math:`\lambda x`, which shrinks with ``dt``. Use `SignalAccumulator` for that case.

   .. seealso::

      :py:obj:`SignalAccumulator`
          Area-preserving trace.


   .. py:attribute:: config
      :type:  SignalAverageConfig


