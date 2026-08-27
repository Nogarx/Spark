spark.core.tracers
==================

.. py:module:: spark.core.tracers


Classes
-------

.. autoapisummary::

   spark.core.tracers.BaseTracer
   spark.core.tracers.Tracer
   spark.core.tracers.RDTracer
   spark.core.tracers.RFSTracer


Functions
---------

.. autoapisummary::

   spark.core.tracers.contract_tracer_args


Module Contents
---------------

.. py:function:: contract_tracer_args(axes, shape, **values)

   Reduces the arguments of a tracer over the axes its output is summed on.

   A tracer whose arguments are constant along the summed axes gives the same result when it
   is applied to the summed value instead of to each entry, which holds far less state.

   :param axes: Axes the trace is summed over.
   :type axes: tuple of int
   :param shape: Shape of the trace before the sum.
   :type shape: tuple of int
   :param \*\*values: Arguments of the tracer.
   :type \*\*values: Any

   :returns: * **reduced** (*dict of str to Any*) -- The arguments, contracted when possible.
             * **contracted** (*bool*) -- True when every argument was contracted, so the caller may build the smaller tracer.


.. py:class:: BaseTracer(shape, seed = None, dtype = jnp.float16, dt = 1.0, **kwargs)

   Bases: :py:obj:`spark.core.backend.Module`, :py:obj:`abc.ABC`


   Base class for exponential traces.

   A tracer holds a value that decays towards a base between calls and is driven by whatever
   is passed in. Subclasses provide `_update`, which advances the trace one step.

   :param shape: Shape of the trace.
   :type shape: tuple of int
   :param dt: Integration step, in ms.
   :type dt: float, default 1.0
   :param dtype: Dtype of the trace.
   :type dtype: DTypeLike, optional


   .. py:attribute:: shape


   .. py:attribute:: rng


   .. py:method:: reset()
      :abstractmethod:



   .. py:method:: masked_reset(mask)
      :abstractmethod:



   .. py:property:: value
      :type: jax.Array

      :abstractmethod:



   .. py:method:: __call__(x, **kwargs)


.. py:class:: Tracer(shape, tau, scale = 1, base = 0, **kwargs)

   Bases: :py:obj:`BaseTracer`


   Single exponential trace.

   :param shape: Shape of the trace.
   :type shape: tuple of int
   :param tau: Decay constant, in ms.
   :type tau: jax.Array or float
   :param scale: Factor applied to the incoming value.
   :type scale: jax.Array or float, default 1
   :param base: Value the trace decays towards.
   :type base: jax.Array or float, default 0

   .. rubric:: Notes

   With :math:`\lambda = 1 - \exp(-\Delta t / \tau)`,

   .. math::
           T \leftarrow T + \lambda (T_{\mathrm{base}} - T) + c \, x

   .. seealso::

      :py:obj:`RDTracer`
          Difference of two exponentials.


   .. py:attribute:: scale


   .. py:attribute:: base


   .. py:attribute:: decay_rate


   .. py:attribute:: trace


   .. py:method:: reset()


   .. py:method:: masked_reset(mask)


   .. py:property:: value
      :type: jax.Array



.. py:class:: RDTracer(shape, tau_rise, tau_decay, scale_rise = 1, scale_decay = 1, base_rise = 0, base_decay = 0, **kwargs)

   Bases: :py:obj:`BaseTracer`


   Rise-and-decay trace.

   The difference of two single exponentials, which rises over ``tau_rise`` and falls over
   ``tau_decay`` instead of jumping on the step a value arrives.

   :param shape: Shape of the trace.
   :type shape: tuple of int
   :param tau_rise: Rise constant, in ms.
   :type tau_rise: jax.Array or float
   :param tau_decay: Decay constant, in ms.
   :type tau_decay: jax.Array or float
   :param scale_rise: Factors applied to the incoming value in each component.
   :type scale_rise: jax.Array or float, default 1
   :param scale_decay: Factors applied to the incoming value in each component.
   :type scale_decay: jax.Array or float, default 1
   :param base_rise: Values each component decays towards.
   :type base_rise: jax.Array or float, default 0
   :param base_decay: Values each component decays towards.
   :type base_decay: jax.Array or float, default 0

   .. rubric:: Notes

   The rise constant is coupled to the decay constant as

   .. math::
           \tau_r' = \frac{\tau_r \tau_d}{\tau_r + \tau_d}

   which keeps the peak at the intended height as the two constants approach each other. The
   trace is the decay component minus the rise component.

   .. seealso::

      :py:obj:`Tracer`
          Single exponential.

      :py:obj:`RFSTracer`
          Rise with a fast and a slow decay.


   .. py:attribute:: tracer_rise


   .. py:attribute:: tracer_decay


   .. py:method:: reset()


   .. py:method:: masked_reset(mask)


   .. py:property:: value
      :type: jax.Array



.. py:class:: RFSTracer(shape, alpha, tau_rise, tau_fast_decay, tau_slow_decay, scale_rise = 1, scale_fast_decay = 1, scale_slow_decay = 1, base_rise = 0, base_fast_decay = 0, base_slow_decay = 0, **kwargs)

   Bases: :py:obj:`BaseTracer`


   Rise, fast decay and slow decay trace.

   A blend of two `RDTracer` traces that share a rise constant and differ in their decay
   constants, so one input leaves both a fast transient and a slow tail.

   :param shape: Shape of the trace.
   :type shape: tuple of int
   :param alpha: Weight of the fast component. The slow component takes ``1 - alpha``.
   :type alpha: jax.Array or float
   :param tau_rise: Rise constant shared by both components, in ms.
   :type tau_rise: jax.Array or float
   :param tau_fast_decay: Decay constants of the two components, in ms.
   :type tau_fast_decay: jax.Array or float
   :param tau_slow_decay: Decay constants of the two components, in ms.
   :type tau_slow_decay: jax.Array or float
   :param scale_rise: Factors applied to the incoming value.
   :type scale_rise: jax.Array or float, default 1
   :param scale_fast_decay: Factors applied to the incoming value.
   :type scale_fast_decay: jax.Array or float, default 1
   :param scale_slow_decay: Factors applied to the incoming value.
   :type scale_slow_decay: jax.Array or float, default 1
   :param base_rise: Values each component decays towards.
   :type base_rise: jax.Array or float, default 0
   :param base_fast_decay: Values each component decays towards.
   :type base_fast_decay: jax.Array or float, default 0
   :param base_slow_decay: Values each component decays towards.
   :type base_slow_decay: jax.Array or float, default 0

   .. rubric:: Notes

   .. math::
           T = \alpha T_{\mathrm{fast}} + (1 - \alpha) T_{\mathrm{slow}}

   .. seealso::

      :py:obj:`RDTracer`
          One rise and one decay constant.


   .. py:attribute:: tracer_rise_fast


   .. py:attribute:: tracer_rise_slow


   .. py:attribute:: alpha


   .. py:method:: reset()


   .. py:method:: masked_reset(mask)


   .. py:property:: value
      :type: jax.Array



