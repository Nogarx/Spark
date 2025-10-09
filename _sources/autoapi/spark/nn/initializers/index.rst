spark.nn.initializers
=====================

.. py:module:: spark.nn.initializers


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/initializers/base/index
   /autoapi/spark/nn/initializers/delay/index
   /autoapi/spark/nn/initializers/kernel/index


Classes
-------

.. autoapisummary::

   spark.nn.initializers.DelayInitializerConfig
   spark.nn.initializers.ConstantDelayInitializerConfig
   spark.nn.initializers.UniformDelayInitializerConfig
   spark.nn.initializers.KernelInitializerConfig
   spark.nn.initializers.UniformKernelInitializerConfig
   spark.nn.initializers.SparseUniformKernelInitializerConfig


Functions
---------

.. autoapisummary::

   spark.nn.initializers.constant_delay_initializer
   spark.nn.initializers.uniform_delay_initializer
   spark.nn.initializers.uniform_kernel_initializer
   spark.nn.initializers.sparse_uniform_kernel_initializer


Package Contents
----------------

.. py:class:: DelayInitializerConfig(**kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.InitializerConfig`


   DelayInitializer configuration class.


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


.. py:function:: constant_delay_initializer(config)

   Builds an initializer that returns a constant positive integer array.


.. py:class:: ConstantDelayInitializerConfig(**kwargs)

   Bases: :py:obj:`DelayInitializerConfig`


   ConstantDelayInitializer configuration class.


   .. py:attribute:: name
      :type:  Literal['constant_delay_initializer']
      :value: 'constant_delay_initializer'



.. py:function:: uniform_delay_initializer(config)

   Builds an initializer that returns positive integers uniformly-distributed random arrays.


.. py:class:: UniformDelayInitializerConfig(**kwargs)

   Bases: :py:obj:`DelayInitializerConfig`


   UniformDelayInitializer configuration class.


   .. py:attribute:: name
      :type:  Literal['uniform_delay_initializer']
      :value: 'uniform_delay_initializer'



.. py:class:: KernelInitializerConfig(**kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.InitializerConfig`


   KernelInitializer configuration class.


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: scale
      :type:  float


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


.. py:function:: uniform_kernel_initializer(config)

   Builds an initializer that returns real uniformly-distributed random arrays.


.. py:class:: UniformKernelInitializerConfig(**kwargs)

   Bases: :py:obj:`KernelInitializerConfig`


   UniformKernelInitializer configuration class.


   .. py:attribute:: name
      :type:  Literal['uniform_kernel_initializer']
      :value: 'uniform_kernel_initializer'



.. py:function:: sparse_uniform_kernel_initializer(config)

   Builds an initializer that returns a real sparse uniformly-distributed random arrays.


.. py:class:: SparseUniformKernelInitializerConfig(**kwargs)

   Bases: :py:obj:`KernelInitializerConfig`


   SparseUniformKernelInitializer configuration class.


   .. py:attribute:: name
      :type:  Literal['sparse_uniform_kernel_initializer']
      :value: 'sparse_uniform_kernel_initializer'



   .. py:attribute:: density
      :type:  float


