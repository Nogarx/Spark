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


Module Contents
---------------

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



   .. py:method:: get_config_spec()
      :classmethod:


      Returns the default configuration class associated with this module.



   .. py:method:: __call__(key, shape)
      :abstractmethod:



