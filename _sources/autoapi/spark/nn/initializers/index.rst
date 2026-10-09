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
   spark.nn.initializers.MaskedInitializer
   spark.nn.initializers.ConstantInitializer
   spark.nn.initializers.ConstantInitializerConfig
   spark.nn.initializers.UniformInitializer
   spark.nn.initializers.UniformInitializerConfig
   spark.nn.initializers.SparseUniformInitializer
   spark.nn.initializers.SparseUniformInitializerConfig
   spark.nn.initializers.NormalizedSparseUniformInitializer
   spark.nn.initializers.NormalizedSparseUniformInitializerConfig
   spark.nn.initializers.NormalInitializer
   spark.nn.initializers.NormalInitializerConfig
   spark.nn.initializers.SparseNormalInitializer
   spark.nn.initializers.SparseNormalInitializerConfig
   spark.nn.initializers.LogNormalInitializer
   spark.nn.initializers.LogNormalInitializerConfig
   spark.nn.initializers.SparseLogNormalInitializer
   spark.nn.initializers.SparseLogNormalInitializerConfig


Package Contents
----------------

.. py:class:: Initializer(*, config = None, **kwargs)

   Bases: :py:obj:`abc.ABC`


   Base class for initializers.

   An initializer produces the array a parameter starts from. Passing one in place of a value
   lets a configuration describe a whole pool without holding the array: the array is drawn
   at build time, once the shape is known.

   A subclass must declare its configuration through the ``config`` annotation, which is also
   what it falls back to when constructed without one.

   :param config: Initializer configuration. Its fields may also be given as keyword arguments.
   :type config: InitializerConfig, optional

   .. seealso::

      :py:obj:`ConstantInitializer`
          Every entry the same value.

      :py:obj:`UniformInitializer`
          Entries drawn uniformly.

      :py:obj:`SparseUniformInitializer`
          Uniform entries with a fraction zeroed.


   .. py:attribute:: config
      :type:  InitializerConfig


   .. py:attribute:: default_config
      :type:  type[ConfigT]


   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:



   .. py:method:: get_config_spec()
      :classmethod:


      Returns the default configuration class associated with this module.



   .. py:method:: __call__(key, shape, **kwargs)
      :abstractmethod:


      Draws the array.

      :param key: PRNG key.
      :type key: jax.Array
      :param shape: Shape of the array to draw.
      :type shape: tuple of int
      :param \*\*kwargs: Extra arguments accepted by the concrete initializer.

      :returns: The drawn array, cast to ``dtype``.
      :rtype: jax.Array



.. py:class:: InitializerConfig

   Bases: :py:obj:`spark.core.config.SparkConfig`, :py:obj:`abc.ABC`


   Base configuration for initializers.

   :param dtype: Dtype of the produced array.
   :type dtype: DTypeLike, default jnp.float16
   :param scale: Factor applied to the produced array.
   :type scale: int or float, default 1
   :param min_value: Lower bound. Applied by clipping in the initializers of this package.
   :type min_value: int or float or None, default None
   :param max_value: Upper bound. Applied by clipping in the initializers of this package.
   :type max_value: int or float or None, default None


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


   .. py:attribute:: scale
      :type:  int | float


   .. py:attribute:: min_value
      :type:  int | float | None


   .. py:attribute:: max_value
      :type:  int | float | None


.. py:class:: MaskedInitializer

   Bases: :py:obj:`abc.ABC`


   Base class for initializers that take a mask.

   Called with a mask alongside the key and the shape, which is what lets an initializer draw
   different values for different groups of entries.


   .. py:method:: __call__(mask, key, shape)
      :abstractmethod:


      Draws the array under a mask.

      :param mask: Selects which entries are drawn together.
      :type mask: jax.Array
      :param key: PRNG key.
      :type key: jax.Array
      :param shape: Shape of the array to draw.
      :type shape: tuple of int

      :returns: The drawn array.
      :rtype: jax.Array



.. py:class:: ConstantInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.Initializer`


   Fills the array with one value: ``scale``.

   :param config: Initializer configuration. Its fields may also be given as keyword arguments.
   :type config: ConstantInitializerConfig, optional


   .. py:attribute:: config
      :type:  ConstantInitializerConfig


   .. py:method:: __call__(key, shape)

      Returns an array filled with ``scale``.

      :param key: PRNG key. Unused, and accepted only to match the initializer signature.
      :type key: jax.Array
      :param shape: Shape of the array.
      :type shape: tuple of int

      :returns: The array, cast to ``dtype``.
      :rtype: jax.Array



.. py:class:: ConstantInitializerConfig

   Bases: :py:obj:`spark.nn.initializers.base.InitializerConfig`


   Configuration for `ConstantInitializer`.

   :param dtype: Dtype of the produced array.
   :type dtype: DTypeLike, default jnp.float16
   :param scale: The value every entry takes.
   :type scale: int or float, default 1


   .. py:attribute:: __class_ref__
      :type:  ClassVar[str]
      :value: 'ConstantInitializer'



.. py:class:: UniformInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.Initializer`


   Draws every entry uniformly from ``[0, scale)``.

   :param config: Initializer configuration. Its fields may also be given as keyword arguments.
   :type config: UniformInitializerConfig, optional

   .. rubric:: Notes

   ``min_value`` and ``max_value`` clip the result, so they narrow the range rather than
   shift it.

   .. seealso::

      :py:obj:`SparseUniformInitializer`
          The same draw with a fraction of the entries zeroed.


   .. py:attribute:: config
      :type:  UniformInitializerConfig


   .. py:method:: __call__(key, shape)

      Draws every entry uniformly from ``[0, scale)``.

      :param key: PRNG key.
      :type key: jax.Array
      :param shape: Shape of the array to draw.
      :type shape: tuple of int

      :returns: The drawn array, clipped to ``[min_value, max_value]`` and cast to ``dtype``.
      :rtype: jax.Array



.. py:class:: UniformInitializerConfig

   Bases: :py:obj:`spark.nn.initializers.base.InitializerConfig`


   Configuration for `UniformInitializer`.

   :param dtype: Dtype of the produced array.
   :type dtype: DTypeLike, default jnp.float16
   :param scale: Upper end of the range drawn from.
   :type scale: int or float, default 1
   :param min_value: Lower clip applied after drawing.
   :type min_value: int or float or None, default None
   :param max_value: Upper clip applied after drawing.
   :type max_value: int or float or None, default None


   .. py:attribute:: __class_ref__
      :type:  ClassVar[str]
      :value: 'UniformInitializer'



.. py:class:: SparseUniformInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`UniformInitializer`


   Draws every entry uniformly, then zeroes a fraction of them.

   Each entry is kept with probability ``density`` and set to zero otherwise, so the number
   of non-zero entries varies between draws around its expected value.

   :param config: Initializer configuration. Its fields may also be given as keyword arguments.
   :type config: SparseUniformInitializerConfig, optional

   .. rubric:: Notes

   The zeroing is applied before the clip, so a ``min_value`` above zero fills the zeroed
   entries back in.

   .. seealso::

      :py:obj:`NormalizedSparseUniformInitializer`
          The same draw, normalized along chosen axes.


   .. py:attribute:: config
      :type:  SparseUniformInitializerConfig


   .. py:method:: __call__(key, shape)

      Draws every entry uniformly, then zeroes a fraction of them.

      :param key: PRNG key. Split once, for the values and for the zeroing mask.
      :type key: jax.Array
      :param shape: Shape of the array to draw.
      :type shape: tuple of int

      :returns: The drawn array, with each entry kept with probability ``density``, clipped to
                ``[min_value, max_value]`` and cast to ``dtype``.
      :rtype: jax.Array



.. py:class:: SparseUniformInitializerConfig

   Bases: :py:obj:`UniformInitializerConfig`


   Configuration for `SparseUniformInitializer`.

   :param dtype: Dtype of the produced array.
   :type dtype: DTypeLike, default jnp.float16
   :param scale: Upper end of the range drawn from.
   :type scale: int or float, default 1
   :param density: Expected fraction of non-zero entries.
   :type density: float, default 0.2
   :param min_value: Lower clip applied after drawing.
   :type min_value: int or float or None, default None
   :param max_value: Upper clip applied after drawing.
   :type max_value: int or float or None, default None


   .. py:attribute:: __class_ref__
      :type:  ClassVar[str]
      :value: 'SparseUniformInitializer'



   .. py:attribute:: density
      :type:  float


.. py:class:: NormalizedSparseUniformInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`SparseUniformInitializer`


   Sparse uniform draw normalized along chosen axes.

   `SparseUniformInitializer` followed by a division by the sum over ``norm_axes``, so every
   group along those axes sums to ``scale``. For a synaptic kernel this fixes the total input
   a postsynaptic unit receives, independent of how many presynaptic units survived the
   sparsification.

   :param config: Initializer configuration. Its fields may also be given as keyword arguments.
   :type config: NormalizedSparseUniformInitializerConfig, optional

   :raises ValueError: If the shape is one dimensional, or if ``norm_axes`` holds a repeated or
       out-of-range axis.
   :raises TypeError: If ``dtype`` is not a float type.

   .. rubric:: Notes

   A group that sums to zero is left as it is rather than divided.

   .. seealso::

      :py:obj:`SparseUniformInitializer`
          The same draw, without normalization.


   .. py:attribute:: config
      :type:  NormalizedSparseUniformInitializerConfig


   .. py:method:: __call__(key, shape)

      Draws a sparse uniform array normalized along ``norm_axes``.

      :param key: PRNG key. Split once, for the values and for the zeroing mask.
      :type key: jax.Array
      :param shape: Shape of the array to draw. Must have two dimensions or more.
      :type shape: tuple of int

      :returns: The drawn array, with every group along ``norm_axes`` summing to ``scale``, clipped
                to ``[min_value, max_value]`` and cast to ``dtype``.
      :rtype: jax.Array

      :raises ValueError: If ``shape`` is one dimensional, or if ``norm_axes`` holds a repeated or
          out-of-range axis.
      :raises TypeError: If ``dtype`` is not a float type.



.. py:class:: NormalizedSparseUniformInitializerConfig

   Bases: :py:obj:`SparseUniformInitializerConfig`


   Configuration for `NormalizedSparseUniformInitializer`.

   :param dtype: Dtype of the produced array. Must be a float type.
   :type dtype: DTypeLike, default jnp.float16
   :param scale: Factor applied after normalization, so each normalized group sums to ``scale``.
   :type scale: int or float, default 1
   :param density: Expected fraction of non-zero entries.
   :type density: float, default 0.2
   :param norm_axes: Axes the sums are taken over. Set by the module that requests the array; a synapse
                     passes its postsynaptic axes.
   :type norm_axes: tuple of int, default (0,)
   :param min_value: Lower clip applied after normalization.
   :type min_value: int or float or None, default None
   :param max_value: Upper clip applied after normalization.
   :type max_value: int or float or None, default None


   .. py:attribute:: __class_ref__
      :type:  ClassVar[str]
      :value: 'NormalizedSparseUniformInitializer'



   .. py:attribute:: norm_axes
      :type:  tuple[int, ...] | None


.. py:class:: NormalInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.Initializer`


   Draws every entry uniformly from ``[0, scale)``.

   :param config: Initializer configuration. Its fields may also be given as keyword arguments.
   :type config: NormalInitializerConfig, optional

   .. rubric:: Notes

   ``min_value`` and ``max_value`` clip the result, so they narrow the range rather than
   shift it.

   .. seealso::

      :py:obj:`SparseNormalInitializer`
          The same draw with a fraction of the entries zeroed.


   .. py:attribute:: config
      :type:  NormalInitializerConfig


   .. py:method:: __call__(key, shape)

      Draws every entry uniformly from ``[0, scale)``.

      :param key: PRNG key.
      :type key: jax.Array
      :param shape: Shape of the array to draw.
      :type shape: tuple of int

      :returns: The drawn array, clipped to ``[min_value, max_value]`` and cast to ``dtype``.
      :rtype: jax.Array



.. py:class:: NormalInitializerConfig

   Bases: :py:obj:`spark.nn.initializers.base.InitializerConfig`


   Configuration for `NormalInitializer`.

   :param dtype: Dtype of the produced array.
   :type dtype: DTypeLike, default jnp.float16
   :param mu: Mean (“centre”) of the distribution.
   :type mu: float, default 0
   :param sigma: Standard deviation (spread or “width”) of the distribution. Must be non-negative.
   :type sigma: float, default 1
   :param min_value: Lower clip applied after drawing.
   :type min_value: int or float or None, default None
   :param max_value: Upper clip applied after drawing.
   :type max_value: int or float or None, default None


   .. py:attribute:: __class_ref__
      :type:  ClassVar[str]
      :value: 'NormalInitializer'



   .. py:attribute:: mu
      :type:  float


   .. py:attribute:: sigma
      :type:  float


.. py:class:: SparseNormalInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`NormalInitializer`


   Draws every entry uniformly from ``[0, scale)``.

   :param config: Initializer configuration. Its fields may also be given as keyword arguments.
   :type config: SparseNormalInitializerConfig, optional

   .. rubric:: Notes

   ``min_value`` and ``max_value`` clip the result, so they narrow the range rather than
   shift it.

   .. seealso::

      :py:obj:`SparseSparseNormalInitializer`
          The same draw with a fraction of the entries zeroed.


   .. py:attribute:: config
      :type:  SparseNormalInitializerConfig


   .. py:method:: __call__(key, shape)

      Draws every entry uniformly from ``[0, scale)``.

      :param key: PRNG key.
      :type key: jax.Array
      :param shape: Shape of the array to draw.
      :type shape: tuple of int

      :returns: The drawn array, clipped to ``[min_value, max_value]`` and cast to ``dtype``.
      :rtype: jax.Array



.. py:class:: SparseNormalInitializerConfig

   Bases: :py:obj:`NormalInitializerConfig`


   Configuration for `SparseNormalInitializer`.

   :param dtype: Dtype of the produced array.
   :type dtype: DTypeLike, default jnp.float16
   :param mu: Mean (“centre”) of the distribution.
   :type mu: float, default 0
   :param sigma: Standard deviation (spread or “width”) of the distribution. Must be non-negative.
   :type sigma: float, default 1
   :param min_value: Lower clip applied after drawing.
   :type min_value: int or float or None, default None
   :param max_value: Upper clip applied after drawing.
   :type max_value: int or float or None, default None


   .. py:attribute:: __class_ref__
      :type:  ClassVar[str]
      :value: 'SparseNormalInitializer'



   .. py:attribute:: density
      :type:  float


.. py:class:: LogNormalInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.Initializer`


   Draws every entry uniformly from ``[0, scale)``.

   :param config: Initializer configuration. Its fields may also be given as keyword arguments.
   :type config: LogNormalInitializerConfig, optional

   .. rubric:: Notes

   ``min_value`` and ``max_value`` clip the result, so they narrow the range rather than
   shift it.

   .. seealso::

      :py:obj:`SparseLogNormalInitializer`
          The same draw with a fraction of the entries zeroed.


   .. py:attribute:: config
      :type:  LogNormalInitializerConfig


   .. py:method:: __call__(key, shape)

      Draws every entry uniformly from ``[0, scale)``.

      :param key: PRNG key.
      :type key: jax.Array
      :param shape: Shape of the array to draw.
      :type shape: tuple of int

      :returns: The drawn array, clipped to ``[min_value, max_value]`` and cast to ``dtype``.
      :rtype: jax.Array



.. py:class:: LogNormalInitializerConfig

   Bases: :py:obj:`spark.nn.initializers.base.InitializerConfig`


   Configuration for `LogNormalInitializer`.

   :param dtype: Dtype of the produced array.
   :type dtype: DTypeLike, default jnp.float16
   :param mu: Mean (“centre”) of the distribution.
   :type mu: float, default 0
   :param sigma: Standard deviation (spread or “width”) of the distribution. Must be non-negative.
   :type sigma: float, default 1
   :param min_value: Lower clip applied after drawing.
   :type min_value: int or float or None, default None
   :param max_value: Upper clip applied after drawing.
   :type max_value: int or float or None, default None


   .. py:attribute:: __class_ref__
      :type:  ClassVar[str]
      :value: 'LogNormalInitializer'



   .. py:attribute:: mu
      :type:  float


   .. py:attribute:: sigma
      :type:  float


.. py:class:: SparseLogNormalInitializer(*, config = None, **kwargs)

   Bases: :py:obj:`LogNormalInitializer`


   Draws every entry uniformly from ``[0, scale)``.

   :param config: Initializer configuration. Its fields may also be given as keyword arguments.
   :type config: SparseLogNormalInitializerConfig, optional

   .. rubric:: Notes

   ``min_value`` and ``max_value`` clip the result, so they narrow the range rather than
   shift it.

   .. seealso::

      :py:obj:`SparseSparseLogNormalInitializer`
          The same draw with a fraction of the entries zeroed.


   .. py:attribute:: config
      :type:  SparseLogNormalInitializerConfig


   .. py:method:: __call__(key, shape)

      Draws every entry uniformly from ``[0, scale)``.

      :param key: PRNG key.
      :type key: jax.Array
      :param shape: Shape of the array to draw.
      :type shape: tuple of int

      :returns: The drawn array, clipped to ``[min_value, max_value]`` and cast to ``dtype``.
      :rtype: jax.Array



.. py:class:: SparseLogNormalInitializerConfig

   Bases: :py:obj:`LogNormalInitializerConfig`


   Configuration for `SparseLogNormalInitializer`.

   :param dtype: Dtype of the produced array.
   :type dtype: DTypeLike, default jnp.float16
   :param mu: Mean (“centre”) of the distribution.
   :type mu: float, default 0
   :param sigma: Standard deviation (spread or “width”) of the distribution. Must be non-negative.
   :type sigma: float, default 1
   :param density: Expected fraction of non-zero entries.
   :type density: float, default 0.2
   :param min_value: Lower clip applied after drawing.
   :type min_value: int or float or None, default None
   :param max_value: Upper clip applied after drawing.
   :type max_value: int or float or None, default None


   .. py:attribute:: __class_ref__
      :type:  ClassVar[str]
      :value: 'SparseLogNormalInitializer'



   .. py:attribute:: density
      :type:  float


