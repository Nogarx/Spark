spark.nn.initializers.kernel
============================

.. py:module:: spark.nn.initializers.kernel


Attributes
----------

.. autoapisummary::

   spark.nn.initializers.kernel.BASE_SCALE


Classes
-------

.. autoapisummary::

   spark.nn.initializers.kernel.KernelInitializerConfig
   spark.nn.initializers.kernel.UniformKernelInitializerConfig
   spark.nn.initializers.kernel.SparseUniformKernelInitializerConfig


Functions
---------

.. autoapisummary::

   spark.nn.initializers.kernel.uniform_kernel_initializer
   spark.nn.initializers.kernel.sparse_uniform_kernel_initializer


Module Contents
---------------

.. py:data:: BASE_SCALE
   :value: 3


.. py:class:: KernelInitializerConfig(**kwargs)

   Bases: :py:obj:`spark.nn.initializers.base.InitializerConfig`


   KernelInitializer configuration class.


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: scale
      :type:  float


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


.. py:class:: UniformKernelInitializerConfig(**kwargs)

   Bases: :py:obj:`KernelInitializerConfig`


   UniformKernelInitializer configuration class.


   .. py:attribute:: name
      :type:  Literal['uniform_kernel_initializer']
      :value: 'uniform_kernel_initializer'



.. py:function:: uniform_kernel_initializer(config)

   Builds an initializer that returns real uniformly-distributed random arrays.


.. py:class:: SparseUniformKernelInitializerConfig(**kwargs)

   Bases: :py:obj:`KernelInitializerConfig`


   SparseUniformKernelInitializer configuration class.


   .. py:attribute:: name
      :type:  Literal['sparse_uniform_kernel_initializer']
      :value: 'sparse_uniform_kernel_initializer'



   .. py:attribute:: density
      :type:  float


.. py:function:: sparse_uniform_kernel_initializer(config)

   Builds an initializer that returns a real sparse uniformly-distributed random arrays.


