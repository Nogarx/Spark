spark.core.tracers
==================

.. py:module:: spark.core.tracers


Classes
-------

.. autoapisummary::

   spark.core.tracers.BaseTracer
   spark.core.tracers.Tracer
   spark.core.tracers.DoubleTracer
   spark.core.tracers.SaturableTracer
   spark.core.tracers.SaturableDoubleTracer
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



.. py:class:: DoubleTracer(shape, tau_1, tau_2, scale_1 = 1, scale_2 = 1, base_1 = 0, base_2 = 0, *args, **kwargs)

   Bases: :py:obj:`BaseTracer`


   Multipurpose double exponential tracer.


   .. py:attribute:: scale_1


   .. py:attribute:: scale_2


   .. py:attribute:: base_1


   .. py:attribute:: base_2


   .. py:attribute:: decay_1


   .. py:attribute:: decay_2


   .. py:attribute:: trace_1


   .. py:attribute:: trace_2


   .. py:method:: reset()


   .. py:method:: masked_reset(mask)


   .. py:property:: value
      :type: jax.Array



.. py:class:: SaturableTracer(shape, tau, scale = 1, base = 0, **kwargs)

   Bases: :py:obj:`Tracer`


   Multipurpose saturable exponential tracer.


.. py:class:: SaturableDoubleTracer(shape, tau_1, tau_2, scale_1 = 1, scale_2 = 1, base_1 = 0, base_2 = 0, *args, **kwargs)

   Bases: :py:obj:`DoubleTracer`


   Multipurpose saturable double exponential tracer.


.. py:class:: RUTracer(shape, R_tau, U_tau, scale_U, **kwargs)

   Bases: :py:obj:`BaseTracer`


   Resource-Usage tracer for STP (Short Term Plasticity).


   .. py:attribute:: scale_U


   .. py:attribute:: decay_R


   .. py:attribute:: decay_U


   .. py:attribute:: trace_R


   .. py:attribute:: trace_U


   .. py:method:: reset()


   .. py:method:: masked_reset(mask)


   .. py:property:: value
      :type: jax.Array



