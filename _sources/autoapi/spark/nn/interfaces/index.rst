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


   Abstract Interface model.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Computes the control flow operation.



.. py:class:: InterfaceConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.core.config.SparkConfig`


   Abstract Interface model configuration class.


.. py:class:: ControlInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract ControlInterface model.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Control operation.



.. py:class:: ControlInterfaceConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Abstract ControlInterface model configuration class.


.. py:class:: ControlInterfaceOutput

   Bases: :py:obj:`TypedDict`


   ControlInterface model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: output
      :type:  spark.core.payloads.SparkPayload


.. py:class:: Concat(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Combines several streams of inputs of the same type into a single stream.

   Init:
       num_inputs: int
       payload_type: type[SparkPayload]

   Input:
       input: type[SparkPayload]

   Output:
       output: type[SparkPayload]


   .. py:attribute:: config
      :type:  ConcatConfig


   .. py:attribute:: num_inputs


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(inputs)

      Merge all input streams into a single data output stream.



.. py:class:: ConcatConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterfaceConfig`


   Concat configuration class.


   .. py:attribute:: num_inputs
      :type:  int


.. py:class:: ConcatReshape(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Combines several streams of inputs of the same type into a single stream.

   Init:
       num_inputs: int
       reshape: tuple[int, ...]
       payload_type: type[SparkPayload]

   Input:
       input: type[SparkPayload]

   Output:
       output: type[SparkPayload]


   .. py:attribute:: config
      :type:  ConcatReshapeConfig


   .. py:attribute:: reshape


   .. py:attribute:: num_inputs


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(inputs)

      Merge all input streams into a single data output stream. Output stream is reshape to match the pre-specified shape.



.. py:class:: ConcatReshapeConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`ConcatConfig`


   ConcatReshape configuration class.


   .. py:attribute:: reshape
      :type:  tuple[int, Ellipsis]


.. py:class:: Sampler(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Sample a single input streams of inputs of the same type into a single stream.
   Indices are selected randomly and remain fixed.

   Init:
       sample_size: int

   Input:
       input: type[SparkPayload]

   Output:
       output: type[SparkPayload]


   .. py:attribute:: config
      :type:  SamplerConfig


   .. py:attribute:: sample_size


   .. py:method:: build(input_specs)

      Build method.



   .. py:property:: indices
      :type: jax.Array



   .. py:method:: __call__(inputs)

      Sub/Super-sample the input stream to get the pre-specified number of samples.



.. py:class:: SamplerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterfaceConfig`


   Sampler configuration class.


   .. py:attribute:: sample_size
      :type:  int


.. py:class:: InputInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract input interface model.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Transform the input signal into an Spike signal.



.. py:class:: InputInterfaceConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Abstract InputInterface model configuration class.


.. py:class:: InputInterfaceOutput

   Bases: :py:obj:`TypedDict`


   InputInterface model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: spikes
      :type:  spark.core.payloads.SpikeArray


.. py:class:: PoissonSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Transforms a continuous signal to a spiking signal.
   This transformation assumes a very simple poisson neuron model without any type of adaptation or plasticity.

   Init:
       max_freq: float [Hz]

   Input:
       signal: FloatArray

   Output:
       spikes: SpikeArray


   .. py:attribute:: config
      :type:  PoissonSpikerConfig


   .. py:attribute:: max_freq


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(signal)

      Input interface operation.

      Input:
          A FloatArray of values in the range [0,1].
      Output:
          A SpikeArray of the same shape as the input.



.. py:class:: PoissonSpikerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterfaceConfig`


   PoissonSpiker model configuration class.


   .. py:attribute:: max_freq
      :type:  float


.. py:class:: LinearSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Transforms a continuous signal to a spiking signal.
   This transformation assumes a very simple linear neuron model without any type of adaptation or plasticity.
   Units have a fixed refractory period and at maximum input signal will fire up to some fixed frequency.

   Init:
       tau: float [ms]
       cd: float [ms]
       max_freq: float [Hz]

   Input:
       signal: FloatArray

   Output:
       spikes: SpikeArray


   .. py:attribute:: config
      :type:  LinearSpikerConfig


   .. py:attribute:: tau


   .. py:attribute:: cd


   .. py:attribute:: max_freq


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: __call__(signal)

      Input interface operation.

      Input:
          A FloatArray of values in the range [0,1].
      Output:
          A SpikeArray of the same shape as the input.



.. py:class:: LinearSpikerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterfaceConfig`


   LinearSpiker model configuration class.


   .. py:attribute:: tau
      :type:  float


   .. py:attribute:: cd
      :type:  float


   .. py:attribute:: max_freq
      :type:  float


.. py:class:: TopologicalPoissonSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Transforms a continuous signal to a spiking signal.
   This transformation maps a vector (a point in a hypercube) into a simple manifold with/without its borders glued.
   This transformation assumes a very simple poisson neuron model without any type of adaptation or plasticity.

   Init:
       glue: jax.Array
       mins: jax.Array
       maxs: jax.Array
       resolution: int
       max_freq: float [Hz]
       sigma: float

   Input:
       signal: FloatArray

   Output:
       spikes: SpikeArray


   .. py:attribute:: config
      :type:  TopologicalPoissonSpikerConfig


   .. py:attribute:: resolution


   .. py:attribute:: max_freq


   .. py:attribute:: sigma


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(signal)

      Input interface operation.

      Input: A FloatArray of values in the range [mins, maxs].
      Output: A SpikeArray of the same shape as the input.



.. py:class:: TopologicalPoissonSpikerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`TopologicalSpikerConfig`, :py:obj:`spark.nn.interfaces.input.poisson.PoissonSpikerConfig`


   TopologicalPoissonSpiker configuration class.


.. py:class:: TopologicalLinearSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Transforms a continuous signal to a spiking signal.
   This transformation maps a vector (a point in a hypercube) into a simple manifold with/without its borders glued.
   This transformation assumes a very simple linear neuron model without any type of adaptation or plasticity.

   Init:
       glue: jax.Array
       mins: jax.Array
       maxs: jax.Array
       resolution: int
       tau: float [ms]
       cd: float [ms]
       max_freq: float [Hz]
       sigma: float

   Input:
       signal: FloatArray

   Output:
       spikes: SpikeArray


   .. py:attribute:: config
      :type:  TopologicalLinearSpikerConfig


   .. py:attribute:: resolution


   .. py:attribute:: tau


   .. py:attribute:: cd


   .. py:attribute:: max_freq


   .. py:attribute:: sigma


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: __call__(signal)

      Input interface operation.

      Input: A FloatArray of values in the range [mins, maxs].
      Output: A SpikeArray of the same shape as the input.



.. py:class:: TopologicalLinearSpikerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`TopologicalSpikerConfig`, :py:obj:`spark.nn.interfaces.input.linear.LinearSpikerConfig`


   TopologicalLinearSpiker configuration class.


.. py:class:: OutputInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract OutputInterface model.


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Transform incomming spikes into a output signal.



.. py:class:: OutputInterfaceConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Abstract OutputInterface model configuration class.


.. py:class:: OutputInterfaceOutput

   Bases: :py:obj:`TypedDict`


   OutputInterface model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: signal
      :type:  spark.core.payloads.FloatArray


.. py:class:: ExponentialIntegrator(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.output.base.OutputInterface`


   Transforms a discrete spike signal to a continuous signal.
   This transformation assumes a very simple integration model model without any type of adaptation or plasticity.
   Spikes are grouped into k non-overlaping clusters and every neuron contributes the same amount to the ouput.

   Init:
       num_outputs: int
       saturation_freq: float [Hz]
       tau: float [ms]
       shuffle: bool
       smooth_trace: bool

   Input:
       spikes: SpikeArray

   Output:
       signal: FloatArray


   .. py:attribute:: config
      :type:  ExponentialIntegratorConfig


   .. py:attribute:: num_outputs


   .. py:attribute:: saturation_freq


   .. py:attribute:: tau


   .. py:attribute:: shuffle


   .. py:attribute:: smooth_trace


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: __call__(spikes)

      Transform incomming spikes into a output signal.



.. py:class:: ExponentialIntegratorConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.output.base.OutputInterfaceConfig`


   ExponentialIntegrator configuration class.


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


