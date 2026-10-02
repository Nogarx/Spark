spark.nn.initializers.common
============================

.. py:module:: spark.nn.initializers.common


Classes
-------

.. autoapisummary::

   spark.nn.initializers.common.ConstantInitializerConfig
   spark.nn.initializers.common.ConstantInitializer
   spark.nn.initializers.common.UniformInitializerConfig
   spark.nn.initializers.common.UniformInitializer
   spark.nn.initializers.common.SparseUniformInitializerConfig
   spark.nn.initializers.common.SparseUniformInitializer
   spark.nn.initializers.common.NormalizedSparseUniformInitializerConfig
   spark.nn.initializers.common.NormalizedSparseUniformInitializer


Module Contents
---------------

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



