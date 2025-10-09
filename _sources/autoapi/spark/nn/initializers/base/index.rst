spark.nn.initializers.base
==========================

.. py:module:: spark.nn.initializers.base


Classes
-------

.. autoapisummary::

   spark.nn.initializers.base.InitializerConfig
   spark.nn.initializers.base.Initializer


Module Contents
---------------

.. py:class:: InitializerConfig(**kwargs)

   Bases: :py:obj:`spark.core.config.BaseSparkConfig`


   Base initializers configuration class.


.. py:class:: Initializer

   Bases: :py:obj:`Protocol`


   Base (abstract) class for all spark initializers.


   .. py:method:: __call__(key, shape, dtype = jnp.float16)
      :staticmethod:

      :abstractmethod:



