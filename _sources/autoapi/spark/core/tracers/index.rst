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
   spark.core.tracers.RUTracer


Module Contents
---------------

.. py:class:: BaseTracer(shape, seed = None, dtype = jnp.float16, dt = 1.0, **kwargs)

   Bases: :py:obj:`flax.nnx.Module`, :py:obj:`abc.ABC`


   Base Tracer class


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


   Multipurpose exponential tracer.


   .. py:attribute:: scale


   .. py:attribute:: base


   .. py:attribute:: decay


   .. py:attribute:: trace


   .. py:method:: reset()


   .. py:method:: masked_reset(mask)


   .. py:property:: value
      :type: jax.Array



.. py:class:: RDTracer(shape, tau_rise, tau_decay, scale_rise = 1, scale_decay = 1, base_rise = 0, base_decay = 0, **kwargs)

   Bases: :py:obj:`BaseTracer`


   Rise-Decay Tracer.

   Multipurpose double exponential tracer.


   .. py:attribute:: tracer_rise


   .. py:attribute:: tracer_decay


   .. py:method:: reset()


   .. py:method:: masked_reset(mask)


   .. py:property:: value
      :type: jax.Array



.. py:class:: RFSTracer(shape, alpha, tau_rise, tau_fast_decay, tau_slow_decay, scale_rise = 1, scale_fast_decay = 1, scale_slow_decay = 1, base_rise = 0, base_fast_decay = 0, base_slow_decay = 0, **kwargs)

   Bases: :py:obj:`BaseTracer`


   Rise-Fast-Slow Tracer

   Multipurpose triple exponential tracer.


   .. py:attribute:: tracer_rise_fast


   .. py:attribute:: tracer_rise_slow


   .. py:attribute:: alpha


   .. py:method:: reset()


   .. py:method:: masked_reset(mask)


   .. py:property:: value
      :type: jax.Array



.. py:class:: RUTracer(shape, r_tau, u_tau, u_scale, **kwargs)

   Bases: :py:obj:`BaseTracer`


   Resource-Usage tracer for STP (Short Term Plasticity).


   .. py:attribute:: r_tracer


   .. py:attribute:: u_tracer


   .. py:method:: reset()


   .. py:method:: masked_reset(mask)


   .. py:property:: value
      :type: jax.Array



