spark.nn.components.synapses
============================

.. py:module:: spark.nn.components.synapses


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/synapses/base/index
   /autoapi/spark/nn/components/synapses/linear/index
   /autoapi/spark/nn/components/synapses/traced/index


Classes
-------

.. autoapisummary::

   spark.nn.components.synapses.Synapses
   spark.nn.components.synapses.SynanpsesOutput
   spark.nn.components.synapses.LinearSynapses
   spark.nn.components.synapses.LinearSynapsesConfig
   spark.nn.components.synapses.TracedSynapses
   spark.nn.components.synapses.TracedSynapsesConfig
   spark.nn.components.synapses.RDTracedSynapses
   spark.nn.components.synapses.RDTracedSynapsesConfig
   spark.nn.components.synapses.RFSTracedSynapses
   spark.nn.components.synapses.RFSTracedSynapsesConfig


Package Contents
----------------

.. py:class:: Synapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for synapse models.

   A synapse model turns presynaptic spikes into postsynaptic current. Subclasses provide
   `_dot`, which is the whole of the step.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: SynanpsesConfig, optional

   :Input Ports: **spikes** (*SpikeArray*) -- Presynaptic spikes.

   :Output Ports: **currents** (*CurrentArray*) -- Current delivered to each postsynaptic unit.

   :Properties: **kernel** (*FloatArray*) -- Synaptic weights, in pA. Writable, which is how a plasticity rule updates them.

   .. rubric:: Notes

   Kernel entries are in pA. The framework runs in half precision by default, and nA-scale
   weights lose too much of the mantissa to be summed reliably.

   The weights are exposed as the writable ``kernel`` property, which is what lets a
   plasticity rule read them and write them back.

   .. seealso::

      :py:obj:`LinearSynapses`
          Weighted sum of the incoming spikes.

      :py:obj:`TracedSynapses`
          Weighted sum filtered by a single exponential.


   .. py:method:: kernel()


   .. py:method:: get_kernel()
      :abstractmethod:



   .. py:method:: set_kernel(new_kernel)
      :abstractmethod:



   .. py:method:: __call__(spikes)

      Converts presynaptic spikes into postsynaptic current.

      :param spikes: Presynaptic spikes.
      :type spikes: SpikeArray

      :returns: Dictionary with one entry, ``currents``, the current delivered to each postsynaptic
                unit.
      :rtype: SynanpsesOutput



.. py:class:: SynanpsesOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a synapse model.

   .. attribute:: currents

      Current delivered to each postsynaptic unit.

      :type: CurrentArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: currents
      :type:  spark.core.payloads.CurrentArray


.. py:class:: LinearSynapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.base.Synapses`


   Linear synapse.

   The postsynaptic current is the weighted sum of the incoming spikes, with no extra dynamics,
   equivalent to delta increments in current.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: LinearSynapsesConfig, optional

   :Input Ports: **spikes** (*SpikeArray*) -- Presynaptic spikes.

   :Output Ports: **currents** (*CurrentArray*) -- Current delivered to each postsynaptic unit, of shape ``units``.

   :Properties: **kernel** (*FloatArray*) -- Synaptic weights, in pA, of shape ``units + input_shape``. Writable, which is how a
                plasticity rule updates them.

   .. rubric:: Notes

   .. math::
       I_i = \sum_j W_{ij} s_j

   The kernel is built with shape ``units + input_shape`` and the sum runs over the
   presynaptic axes. Spikes marked asynchronous, as produced by `N2NDelays`, already carry
   one entry per (postsynaptic, presynaptic) pair; the postsynaptic axes are then matched
   elementwise and only the presynaptic axes are summed.

   .. rubric:: References

   .. [1] W. Gerstner, W. M. Kistler, R. Naud and L. Paninski, "Neuronal Dynamics: From
          Single Neurons to Networks and Models of Cognition", Chapter 1.3, Integrate-And-Fire
          Models. https://neuronaldynamics.epfl.ch/online/Ch1.S3.html

   .. seealso::

      :py:obj:`TracedSynapses`
          This model with an exponential postsynaptic current.


   .. py:attribute:: config
      :type:  LinearSynapsesConfig


   .. py:method:: build(spikes)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: get_kernel()


   .. py:method:: get_flat_kernel()


   .. py:method:: set_kernel(new_kernel)


.. py:class:: LinearSynapsesConfig

   Bases: :py:obj:`spark.nn.components.synapses.base.SynanpsesConfig`


   Configuration for `LinearSynapses`.

   :param units: Shape of the postsynaptic pool.
   :type units: tuple of int
   :param kernel: Synaptic weights, in pA. The kernel is built with shape ``units + input_shape``.
   :type kernel: jax.Array or Initializer, default SparseUniformInitializerConfig()


   .. py:attribute:: units
      :type:  tuple[int, ...]


   .. py:attribute:: kernel
      :type:  jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: TracedSynapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapses`


   Linear synapse with an exponential postsynaptic current.

   The weighted spikes are passed through a single exponential trace, so one spike
   contributes a current that decays over ``tau`` rather than over a single step.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: TracedSynapsesConfig, optional

   :Input Ports: **spikes** (*SpikeArray*) -- Presynaptic spikes.

   :Output Ports: **currents** (*CurrentArray*) -- Current delivered to each postsynaptic unit, of shape ``units``.

   :Properties: **kernel** (*FloatArray*) -- Synaptic weights, in pA, of shape ``units + input_shape``. Writable, which is how a
                plasticity rule updates them.

   .. rubric:: Notes

   With :math:`\lambda = 1 - \exp(-\Delta t / \tau)` the trace is

   .. math::
       T \leftarrow T + \lambda (T_{\mathrm{base}} - T) + c \, W s

   and the current is the sum of :math:`T` over the presynaptic axes.

   When ``tau``, ``scale`` and ``base`` are uniform along the summed axes, the trace is
   applied to the already summed current instead of to each connection. That holds one state
   entry per postsynaptic unit rather than one per weight, and gives the same result.

   .. seealso::

      :py:obj:`LinearSynapses`
          Weighted sum without a postsynaptic current.

      :py:obj:`RDTracedSynapses`
          Separate rise and decay constants.


   .. py:attribute:: config
      :type:  TracedSynapsesConfig


   .. py:attribute:: current_tracer
      :type:  spark.core.tracers.Tracer


   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



.. py:class:: TracedSynapsesConfig

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapsesConfig`


   Configuration for `TracedSynapses`.

   :param units: Shape of the postsynaptic pool.
   :type units: tuple of int
   :param kernel: Synaptic weights, in pA.
   :type kernel: jax.Array or Initializer, default SparseUniformInitializerConfig()
   :param tau: Decay constant of the postsynaptic current, in ms.
   :type tau: float or jax.Array or Initializer, default 5.0
   :param scale: Factor applied to the weighted spikes entering the trace.
   :type scale: float or jax.Array, default 1.0
   :param base: Value the trace decays to.
   :type base: float or jax.Array, default 0.0


   .. py:attribute:: tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale
      :type:  float | jax.Array


   .. py:attribute:: base
      :type:  float | jax.Array


.. py:class:: RDTracedSynapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapses`


   Linear synapse with a rise-and-decay postsynaptic current.

   The weighted spikes are passed through the difference of two exponentials, which gives a
   current that rises over ``tau_rise`` and falls over ``tau_decay`` instead of jumping on
   the step a spike arrives.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: RDTracedSynapsesConfig, optional

   :Input Ports: **spikes** (*SpikeArray*) -- Presynaptic spikes.

   :Output Ports: **currents** (*CurrentArray*) -- Current delivered to each postsynaptic unit, of shape ``units``.

   :Properties: **kernel** (*FloatArray*) -- Synaptic weights, in pA, of shape ``units + input_shape``. Writable, which is how a
                plasticity rule updates them.

   .. rubric:: Notes

   The trace is the decay exponential minus the rise exponential. The rise constant is
   coupled to the decay constant as

   .. math::
       \tau_r' = \frac{\tau_r \tau_d}{\tau_r + \tau_d}

   which is what keeps the peak at the intended height as the two constants approach each
   other.

   The contraction described in `TracedSynapses` applies here as well.

   .. seealso::

      :py:obj:`TracedSynapses`
          Single exponential.

      :py:obj:`RFSTracedSynapses`
          Rise with a fast and a slow decay.


   .. py:attribute:: config
      :type:  RDTracedSynapsesConfig


   .. py:attribute:: current_tracer
      :type:  spark.core.tracers.RDTracer


   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



.. py:class:: RDTracedSynapsesConfig

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapsesConfig`


   Configuration for `RDTracedSynapses`.

   :param units: Shape of the postsynaptic pool.
   :type units: tuple of int
   :param kernel: Synaptic weights, in pA.
   :type kernel: jax.Array or Initializer, default SparseUniformInitializerConfig()
   :param tau_rise: Rise constant of the postsynaptic current, in ms.
   :type tau_rise: float or jax.Array or Initializer, default 1.0
   :param scale_rise: Factor applied to the weighted spikes entering the rise trace.
   :type scale_rise: float or jax.Array, default 1.0
   :param base_rise: Value the rise trace decays to.
   :type base_rise: float or jax.Array, default 0.0
   :param tau_decay: Decay constant of the postsynaptic current, in ms.
   :type tau_decay: float or jax.Array or Initializer, default 5.0
   :param scale_decay: Factor applied to the weighted spikes entering the decay trace.
   :type scale_decay: float or jax.Array, default 1.0
   :param base_decay: Value the decay trace decays to.
   :type base_decay: float or jax.Array, default 0.0


   .. py:attribute:: tau_rise
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale_rise
      :type:  float | jax.Array


   .. py:attribute:: base_rise
      :type:  float | jax.Array


   .. py:attribute:: tau_decay
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale_decay
      :type:  float | jax.Array


   .. py:attribute:: base_decay
      :type:  float | jax.Array


.. py:class:: RFSTracedSynapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapses`


   Linear synapse with a two-component postsynaptic current.

   A blend of two rise-and-decay traces that share a rise constant and differ in their decay
   constants. One spike then leaves both a fast transient and a slow tail, which a single
   decay constant cannot produce.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: RFSTracedSynapsesConfig, optional

   :Input Ports: **spikes** (*SpikeArray*) -- Presynaptic spikes.

   :Output Ports: **currents** (*CurrentArray*) -- Current delivered to each postsynaptic unit, of shape ``units``.

   :Properties: **kernel** (*FloatArray*) -- Synaptic weights, in pA, of shape ``units + input_shape``. Writable, which is how a
                plasticity rule updates them.

   .. rubric:: Notes

   .. math::
       T = \alpha T_{\mathrm{fast}} + (1 - \alpha) T_{\mathrm{slow}}

   where each component is a `RDTracedSynapses` trace built on the shared ``tau_rise``. The
   contraction described in `TracedSynapses` applies here as well.

   .. seealso::

      :py:obj:`RDTracedSynapses`
          One rise and one decay constant.


   .. py:attribute:: config
      :type:  RFSTracedSynapsesConfig


   .. py:attribute:: current_tracer
      :type:  spark.core.tracers.RDTracer


   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: reset()

      Resets component state.



.. py:class:: RFSTracedSynapsesConfig

   Bases: :py:obj:`spark.nn.components.synapses.linear.LinearSynapsesConfig`


   Configuration for `RFSTracedSynapses`.

   :param units: Shape of the postsynaptic pool.
   :type units: tuple of int
   :param kernel: Synaptic weights, in pA.
   :type kernel: jax.Array or Initializer, default SparseUniformInitializerConfig()
   :param alpha: Weight of the fast component. The slow component takes ``1 - alpha``. Must lie in
                 ``[0, 1]``.
   :type alpha: float or jax.Array, default 0.8
   :param tau_rise: Rise constant shared by both components, in ms.
   :type tau_rise: float or jax.Array or Initializer, default 1.0
   :param scale_rise: Factor applied to the weighted spikes entering the rise traces.
   :type scale_rise: float or jax.Array, default 1.0
   :param base_rise: Value the rise traces decay to.
   :type base_rise: float or jax.Array, default 0.0
   :param tau_fast_decay: Decay constant of the fast component, in ms.
   :type tau_fast_decay: float or jax.Array or Initializer, default 5.0
   :param scale_fast_decay: Factor applied to the weighted spikes entering the fast decay trace.
   :type scale_fast_decay: float or jax.Array, default 1.0
   :param base_fast_decay: Value the fast decay trace decays to.
   :type base_fast_decay: float or jax.Array, default 0.0
   :param tau_slow_decay: Decay constant of the slow component, in ms.
   :type tau_slow_decay: float or jax.Array or Initializer, default 50.0
   :param scale_slow_decay: Factor applied to the weighted spikes entering the slow decay trace.
   :type scale_slow_decay: float or jax.Array, default 1.0
   :param base_slow_decay: Value the slow decay trace decays to.
   :type base_slow_decay: float or jax.Array, default 0.0


   .. py:attribute:: alpha
      :type:  float | jax.Array


   .. py:attribute:: tau_rise
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale_rise
      :type:  float | jax.Array


   .. py:attribute:: base_rise
      :type:  float | jax.Array


   .. py:attribute:: tau_fast_decay
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale_fast_decay
      :type:  float | jax.Array


   .. py:attribute:: base_fast_decay
      :type:  float | jax.Array


   .. py:attribute:: tau_slow_decay
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: scale_slow_decay
      :type:  float | jax.Array


   .. py:attribute:: base_slow_decay
      :type:  float | jax.Array


