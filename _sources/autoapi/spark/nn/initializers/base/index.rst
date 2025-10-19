spark.nn.initializers.base
==========================

.. py:module:: spark.nn.initializers.base


Attributes
----------

.. autoapisummary::

   spark.nn.initializers.base.T
   spark.nn.initializers.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.initializers.base.InitializerConfig
   spark.nn.initializers.base.Initializer


Module Contents
---------------

.. py:data:: T

.. py:class:: InitializerConfig(**kwargs)

   Bases: :py:obj:`spark.core.config.BaseSparkConfig`


   Base initializers configuration class.


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


   .. py:attribute:: scale
      :type:  T


   .. py:attribute:: min_value
      :type:  T | None


   .. py:attribute:: max_value
      :type:  T | None


.. py:data:: ConfigT

.. py:class:: Initializer(*, config = None, **kwargs)

   Bases: :py:obj:`abc.ABC`


   Base (abstract) class for all Spark initializers.


   .. py:attribute:: config
      :type:  InitializerConfig


   .. py:attribute:: default_config
      :type:  type[ConfigT]


   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:



   .. py:method:: __call__(key, shape)
      :abstractmethod:



