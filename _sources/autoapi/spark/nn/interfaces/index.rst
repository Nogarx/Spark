spark.nn.interfaces
===================

.. py:module:: spark.nn.interfaces


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/interfaces/base/index
   /autoapi/spark/nn/interfaces/control/index
   /autoapi/spark/nn/interfaces/input/index
   /autoapi/spark/nn/interfaces/output/index


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.Interface
   spark.nn.interfaces.InterfaceConfig
   spark.nn.interfaces.ControlInterface
   spark.nn.interfaces.ControlInterfaceConfig
   spark.nn.interfaces.ControlInterfaceOutput
   spark.nn.interfaces.Concat
   spark.nn.interfaces.ConcatConfig
   spark.nn.interfaces.ConcatReshape
   spark.nn.interfaces.ConcatReshapeConfig
   spark.nn.interfaces.Sampler
   spark.nn.interfaces.SamplerConfig
   spark.nn.interfaces.SignalTrace
   spark.nn.interfaces.SignalTraceConfig
   spark.nn.interfaces.SignalTraceOutput
   spark.nn.interfaces.SignalAccumulator
   spark.nn.interfaces.SignalAccumulatorConfig
   spark.nn.interfaces.SignalAverage
   spark.nn.interfaces.SignalAverageConfig
   spark.nn.interfaces.InputInterface
   spark.nn.interfaces.InputInterfaceConfig
   spark.nn.interfaces.InputInterfaceOutput
   spark.nn.interfaces.PoissonSpiker
   spark.nn.interfaces.PoissonSpikerConfig
   spark.nn.interfaces.LinearSpiker
   spark.nn.interfaces.LinearSpikerConfig
   spark.nn.interfaces.TopologicalPoissonSpiker
   spark.nn.interfaces.TopologicalPoissonSpikerConfig
   spark.nn.interfaces.TopologicalLinearSpiker
   spark.nn.interfaces.TopologicalLinearSpikerConfig
   spark.nn.interfaces.OutputInterface
   spark.nn.interfaces.OutputInterfaceConfig
   spark.nn.interfaces.OutputInterfaceOutput
   spark.nn.interfaces.ExponentialIntegrator
   spark.nn.interfaces.ExponentialIntegratorConfig


Package Contents
----------------

.. py:class:: Interface(config = None, **kwargs)

   Bases: :py:obj:`spark.core.module.SparkModule`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for interfaces.

   An interface sits between a network and something that is not a network. Unlike a
   `Component`, it holds no neuronal state: it converts, routes or summarizes payloads.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: InterfaceConfig

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Declared by the concrete interface through the signature of its ``__call__``.

   :Output Ports: **\*\*outputs** (*SparkPayload*) -- Declared by the concrete interface through the TypedDict its ``__call__`` returns.

   .. seealso::

      :py:obj:`InputInterface`
          Turns an external signal into spikes.

      :py:obj:`OutputInterface`
          Turns spikes into a continuous signal.

      :py:obj:`ControlInterface`
          Routes and combines payloads inside a network.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Runs the interface operation.

      :param \*args: Inputs, as declared by the concrete interface.
      :type \*args: SparkPayload
      :param \*\*kwargs: Inputs, as declared by the concrete interface.

      :returns: Dictionary of output ports, as declared by the concrete interface.
      :rtype: InterfaceOutput



.. py:class:: InterfaceConfig

   Bases: :py:obj:`spark.core.config.DefaultSparkConfig`


   Base configuration for interfaces.


.. py:class:: ControlInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for control interfaces.

   A control interface moves and transforms payloads around a graph rather than modelling anything.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ControlInterfaceConfig, optional

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Named by the graph that wires them, not by the signature.

   :Output Ports: **output** (*SparkPayload*) -- Result of the operation. Its type follows the inputs.

   .. seealso::

      :py:obj:`Concat`
          Joins several inputs of one type along an axis.

      :py:obj:`Sampler`
          Draws a subset of one input.

      :py:obj:`SignalTrace`
          Exponentially decaying trace of an input.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Runs the control operation.

      :param \*args: Inputs, named by the graph rather than by the signature.
      :type \*args: SparkPayload
      :param \*\*kwargs: Inputs, named by the graph rather than by the signature.

      :returns: Dictionary with one entry, ``output``.
      :rtype: ControlInterfaceOutput



.. py:class:: ControlInterfaceConfig

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Base configuration for control interfaces.


.. py:class:: ControlInterfaceOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a control interface.

   .. attribute:: output

      The result of the operation. Its type follows the inputs.

      :type: SparkPayload

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: output
      :type:  spark.core.payloads.SparkPayload


.. py:class:: Concat(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Joins several inputs into one flat payload.

   Every input is flattened and concatenated in the order the ports were declared. All inputs
   must carry the same payload type, which is also the type of the result.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ConcatConfig, optional

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Any number of inputs, all of the same payload type. Named by the graph.

   :Output Ports: **output** (*SparkPayload*) -- The joined inputs, one dimensional and of the same payload type as the inputs.

   .. seealso::

      :py:obj:`ConcatReshape`
          The same join, followed by a reshape.


   .. py:attribute:: config
      :type:  ConcatConfig


   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: __call__(**inputs)

      Flattens and concatenates every input.

      :param \*\*inputs: Any number of inputs, all of the same payload type.
      :type \*\*inputs: SparkPayload

      :returns: Dictionary with one entry, ``output``, one dimensional and of the payload type of the
                inputs.
      :rtype: ControlInterfaceOutput



.. py:class:: ConcatConfig

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterfaceConfig`


   Configuration for `Concat`.


.. py:class:: ConcatReshape(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Joins several inputs into one payload of a given shape.

   `Concat` followed by a reshape to ``reshape``, which is what lets several one dimensional
   sources feed a module that expects a pool of a particular shape.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ConcatReshapeConfig

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Any number of inputs, all of the same payload type. Named by the graph.

   :Output Ports: **output** (*SparkPayload*) -- The joined inputs, of shape ``reshape`` and of the same payload type as the inputs.

   .. seealso::

      :py:obj:`Concat`
          The join, without the reshape.


   .. py:attribute:: config
      :type:  ConcatReshapeConfig


   .. py:attribute:: reshape
      :value: ()



   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: __call__(**inputs)

      Flattens, concatenates and reshapes every input.

      :param \*\*inputs: Any number of inputs, all of the same payload type.
      :type \*\*inputs: SparkPayload

      :returns: Dictionary with one entry, ``output``, of shape ``reshape`` and of the payload type of
                the inputs.
      :rtype: ControlInterfaceOutput



.. py:class:: ConcatReshapeConfig

   Bases: :py:obj:`ConcatConfig`


   Configuration for `ConcatReshape`.

   :param reshape: Shape of the result. Its size must match the total size of the inputs.
   :type reshape: tuple of int


   .. py:attribute:: reshape
      :type:  tuple[int, ...]


.. py:class:: Sampler(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Draws a fixed set of entries from its inputs.

   The inputs are flattened, concatenated and indexed by a set of indices drawn once at build
   time and held for the lifetime of the module. The same entries are read on every step, so
   this is a fixed projection rather than a fresh sample per step.

   Indices are drawn with replacement, so ``sample_size`` may exceed the size of the input
   and an entry may appear more than once.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: SamplerConfig

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Any number of inputs, all of the same payload type. Named by the graph.

   :Output Ports: **output** (*SparkPayload*) -- The drawn entries, of shape ``sample_size`` and of the same payload type as the inputs.

   :Properties: **indices** (*jax.Array*) -- Flat indices drawn at build time and read on every step. Read only.


   .. py:attribute:: config
      :type:  SamplerConfig


   .. py:attribute:: sample_size


   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:property:: indices
      :type: jax.Array



   .. py:method:: __call__(**inputs)

      Reads the drawn entries out of the inputs.

      :param \*\*inputs: Any number of inputs, all of the same payload type.
      :type \*\*inputs: SparkPayload

      :returns: Dictionary with one entry, ``output``, of shape ``sample_size`` and of the payload
                type of the inputs.
      :rtype: ControlInterfaceOutput



.. py:class:: SamplerConfig

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterfaceConfig`


   Configuration for `Sampler`.

   :param sample_size: Shape of the result. May be larger than the input, in which case entries are drawn
                       more than once.
   :type sample_size: tuple of int


   .. py:attribute:: sample_size
      :type:  int


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


.. py:class:: SignalTraceOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a signal trace.

   .. attribute:: output

      The trace after this step, plus whatever arrived on the ``trace`` port.

      :type: FloatArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: output
      :type:  spark.core.payloads.FloatArray


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


.. py:class:: SignalAccumulatorConfig

   Bases: :py:obj:`SignalTraceConfig`


   Configuration for `SignalAccumulator`.

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


.. py:class:: SignalAverageConfig

   Bases: :py:obj:`SignalTraceConfig`


   Configuration for `SignalAverage`.

   :param tau: Decay constant of the trace, in ms.
   :type tau: float or jax.Array, default 10.0
   :param base: Value the trace decays towards.
   :type base: float or jax.Array, default 0.0


.. py:class:: InputInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for input interfaces.

   An input interface encodes a continuous signal as spikes, which is what lets a network
   read data that did not come from a network.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: InputInterfaceConfig

   :Input Ports: **signal** (*FloatArray*) -- Value to encode.

   :Output Ports: **spikes** (*SpikeArray*) -- The encoded signal.

   .. seealso::

      :py:obj:`PoissonSpiker`
          Stochastic rate encoding.

      :py:obj:`LinearSpiker`
          Deterministic rate encoding.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Encodes the signal as spikes.

      :param \*args: Inputs, as declared by the concrete interface.
      :type \*args: SparkPayload
      :param \*\*kwargs: Inputs, as declared by the concrete interface.

      :returns: Dictionary with one entry, ``spikes``.
      :rtype: InputInterfaceOutput



.. py:class:: InputInterfaceConfig

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Base configuration for input interfaces.


.. py:class:: InputInterfaceOutput

   Bases: :py:obj:`TypedDict`


   Output ports of an input interface.

   .. attribute:: spikes

      The encoded signal.

      :type: SpikeArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: spikes
      :type:  spark.core.payloads.SpikeArray


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



.. py:class:: PoissonSpikerConfig

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterfaceConfig`


   Configuration for `PoissonSpiker`.

   :param max_freq: Firing rate reached by an input of 1.0, in Hz.
   :type max_freq: float, default 100.0


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



.. py:class:: TopologicalPoissonSpikerConfig

   Bases: :py:obj:`TopologicalSpikerConfig`, :py:obj:`spark.nn.interfaces.input.poisson.PoissonSpikerConfig`


   Configuration for `TopologicalPoissonSpiker`.


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



.. py:class:: TopologicalLinearSpikerConfig

   Bases: :py:obj:`TopologicalSpikerConfig`, :py:obj:`spark.nn.interfaces.input.linear.LinearSpikerConfig`


   Configuration for `TopologicalLinearSpiker`.

   Union of `TopologicalSpikerConfig` and `LinearSpikerConfig`. It declares no field of its
   own.


.. py:class:: OutputInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for output interfaces.

   An output interface decodes spikes into a continuous signal, which is what lets something
   that is not a network read what a network produced.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: OutputInterfaceConfig, optional

   :Input Ports: **spikes** (*SpikeArray*) -- Spikes to decode.

   :Output Ports: **signal** (*FloatArray*) -- The decoded signal.

   .. seealso::

      :py:obj:`ExponentialIntegrator`
          Exponential filter of the incoming spikes.


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Decodes the incoming spikes into a signal.

      :param \*args: Inputs, as declared by the concrete interface.
      :type \*args: SpikeArray
      :param \*\*kwargs: Inputs, as declared by the concrete interface.

      :returns: Dictionary with one entry, ``signal``.
      :rtype: dict of str to SparkPayload



.. py:class:: OutputInterfaceConfig

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Base configuration for output interfaces.


.. py:class:: OutputInterfaceOutput

   Bases: :py:obj:`TypedDict`


   Output ports of an output interface.

   .. attribute:: signal

      The decoded signal.

      :type: FloatArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: signal
      :type:  spark.core.payloads.FloatArray


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


