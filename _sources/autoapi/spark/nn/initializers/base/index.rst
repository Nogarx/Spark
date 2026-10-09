spark.nn.initializers.base
==========================

.. py:module:: spark.nn.initializers.base


Attributes
----------

.. autoapisummary::

   spark.nn.initializers.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.initializers.base.InitializerConfig
   spark.nn.initializers.base.Initializer
   spark.nn.initializers.base.MaskedInitializer


Module Contents
---------------

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


.. py:data:: ConfigT

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



