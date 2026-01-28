spark.nn.initializers
=====================

.. py:module:: spark.nn.initializers


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/initializers/base/index
   /autoapi/spark/nn/initializers/common/index


Classes
-------

.. autoapisummary::

   spark.nn.initializers.Initializer
   spark.nn.initializers.InitializerConfig
   spark.nn.initializers.ConstantInitializer
   spark.nn.initializers.ConstantInitializerConfig
   spark.nn.initializers.UniformInitializer
   spark.nn.initializers.UniformInitializerConfig
   spark.nn.initializers.SparseUniformInitializer
   spark.nn.initializers.SparseUniformInitializerConfig
   spark.nn.initializers.NormalizedSparseUniformInitializer
   spark.nn.initializers.NormalizedSparseUniformInitializerConfig


Package Contents
----------------

.. py:class:: Initializer(*, config = None, **kwargs)

   Bases: :py:obj:`abc.ABC`


   Base (abstract) class for all Spark initializers.


   .. py:attribute:: config
      :type:  InitializerConfig


   .. py:attribute:: default_config
      :type:  type[ConfigT]


   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:



   .. py:method:: get_config_spec()
      :classmethod:


      Returns the default configuration class associated with this module.



   .. py:method:: __call__(key, shape)
      :abstractmethod:



.. py:class:: InitializerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.core.config.BaseSparkConfig`, :py:obj:`abc.ABC`


   Base initializers configuration class.


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


   .. py:attribute:: scale
      :type:  int | float


   .. py:attribute:: min_value
      :type:  int | float | None


   .. py:attribute:: max_value
      :type:  int | float | None


.. py:class:: ConstantInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.Initializer`


   Initializer that returns real uniformly-distributed random arrays.

   Init:
       scale: numeric, value for the output array (default = 1).

   Input:
       key: jax.Array, key for the random generator (jax.random.key).
       shape: tuple[int, ...],shaoe fir the output array.


   .. py:attribute:: config
      :type:  ConstantInitializerConfig


   .. py:method:: __call__(key, shape)


.. py:class:: ConstantInitializerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.InitializerConfig`


   ConstantInitializer configuration class.


   .. py:attribute:: __class_ref__
      :type:  str
      :value: 'ConstantInitializer'



.. py:class:: UniformInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.Initializer`


   Initializer that returns real uniformly-distributed random arrays.

   Init:
       scale: numeric, multiplicative factor for the output array (default = 1).
       min_value: numeric, minimum value for the output array (default = None).
       max_value: numeric, maximum value for the output array (default = None).

   Input:
       key: jax.Array, key for the random generator (jax.random.key).
       shape: tuple[int, ...],shaoe fir the output array.


   .. py:attribute:: config
      :type:  UniformInitializerConfig


   .. py:method:: __call__(key, shape)


.. py:class:: UniformInitializerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.InitializerConfig`


   UniformInitializer configuration class.


   .. py:attribute:: __class_ref__
      :type:  str
      :value: 'UniformInitializer'



.. py:class:: SparseUniformInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`UniformInitializer`


   Initializer that returns a real sparse uniformly-distributed random arrays.

   Note that the output will contain zero values even if min_value > 0.

   Init:
       scale: numeric, multiplicative factor for the output array (default = 1).
       min_value: numeric, minimum value for the output array (default = None).
       max_value: numeric, maximum value for the output array (default = None).
       density: float, expected ratio of non-zero entries (default = 0.2).

   Input:
       key: jax.Array, key for the random generator (jax.random.key).
       shape: tuple[int, ...],shaoe fir the output array.


   .. py:attribute:: config
      :type:  SparseUniformInitializerConfig


   .. py:method:: __call__(key, shape)


.. py:class:: SparseUniformInitializerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`UniformInitializerConfig`


   SparseUniformInitializer configuration class.


   .. py:attribute:: __class_ref__
      :type:  str
      :value: 'SparseUniformInitializer'



   .. py:attribute:: density
      :type:  float


.. py:class:: NormalizedSparseUniformInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`SparseUniformInitializer`


   Initializer that returns a real sparse uniformly-distributed random arrays.
   This is a variation of the SparseUniformInitializer that normalizes the array, which may be useful to prevent quiescent neurons.
   Entries in the array are normalized by contracting the array over to the norm_axes and rescaled back to [min_value, max_value].

   Normalization example
   array -> ijk;
   norm_axes -> (i,k)
   contraction = 'ijk->ik'
   sum(norm_array[i,:,k]) = 1

   Note that the output will contain zero values even if min_value > 0.

   Init:
       scale: numeric, multiplicative factor for the output array (default = 1).
       min_value: numeric, minimum value for the output array (default = None).
       max_value: numeric, maximum value for the output array (default = None).
       density: float, expected ratio of non-zero entries (default = 0.2).
       norm_axes: tuple[int, ...], axes used for normalization (default = (0,)):

   Input:
       key: jax.Array, key for the random generator (jax.random.key).
       shape: tuple[int, ...], shape for the output array.

   Output:
       jax.Array[dtype]


   .. py:attribute:: config
      :type:  NormalizedSparseUniformInitializerConfig


   .. py:method:: __call__(key, shape)


.. py:class:: NormalizedSparseUniformInitializerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`SparseUniformInitializerConfig`


   NormalizedSparseUniformInitializer configuration class.


   .. py:attribute:: __class_ref__
      :type:  str
      :value: 'NormalizedSparseUniformInitializer'



   .. py:attribute:: norm_axes
      :type:  tuple[int, Ellipsis] | None


