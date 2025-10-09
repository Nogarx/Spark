spark.nn.initializers.delay
===========================

.. py:module:: spark.nn.initializers.delay


Classes
-------

.. autoapisummary::

   spark.nn.initializers.delay.DelayInitializerConfig
   spark.nn.initializers.delay.ConstantDelayInitializerConfig
   spark.nn.initializers.delay.UniformDelayInitializerConfig


Functions
---------

.. autoapisummary::

   spark.nn.initializers.delay.constant_delay_initializer
   spark.nn.initializers.delay.uniform_delay_initializer


Module Contents
---------------

.. py:class:: DelayInitializerConfig(**kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.InitializerConfig`


   DelayInitializer configuration class.


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


.. py:class:: ConstantDelayInitializerConfig(**kwargs)

   Bases: :py:obj:`DelayInitializerConfig`


   ConstantDelayInitializer configuration class.


   .. py:attribute:: name
      :type:  Literal['constant_delay_initializer']
      :value: 'constant_delay_initializer'



.. py:function:: constant_delay_initializer(config)

   Builds an initializer that returns a constant positive integer array.


.. py:class:: UniformDelayInitializerConfig(**kwargs)

   Bases: :py:obj:`DelayInitializerConfig`


   UniformDelayInitializer configuration class.


   .. py:attribute:: name
      :type:  Literal['uniform_delay_initializer']
      :value: 'uniform_delay_initializer'



.. py:function:: uniform_delay_initializer(config)

   Builds an initializer that returns positive integers uniformly-distributed random arrays.


