spark.nn.interfaces.input.topological
=====================================

.. py:module:: spark.nn.interfaces.input.topological


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.input.topological.TopologicalSpikerConfig
   spark.nn.interfaces.input.topological.TopologicalPoissonSpikerConfig
   spark.nn.interfaces.input.topological.TopologicalPoissonSpiker
   spark.nn.interfaces.input.topological.TopologicalLinearSpikerConfig
   spark.nn.interfaces.input.topological.TopologicalLinearSpiker


Module Contents
---------------

.. py:class:: TopologicalSpikerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterfaceConfig`


   Base TopologicalSpiker configuration class.


   .. py:attribute:: glue
      :type:  jax.Array


   .. py:attribute:: mins
      :type:  jax.Array


   .. py:attribute:: maxs
      :type:  jax.Array


   .. py:attribute:: resolution
      :type:  int


   .. py:attribute:: sigma
      :type:  float


.. py:class:: TopologicalPoissonSpikerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`TopologicalSpikerConfig`, :py:obj:`spark.nn.interfaces.input.poisson.PoissonSpikerConfig`


   TopologicalPoissonSpiker configuration class.


.. py:class:: TopologicalPoissonSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Transforms a continuous signal to a spiking signal.
   This transformation maps a vector (a point in a hypercube) into a simple manifold with/without its borders glued.
   This transformation assumes a very simple poisson neuron model without any type of adaptation or plasticity.

   Init:
       glue: jax.Array
       mins: jax.Array
       maxs: jax.Array
       resolution: int
       max_freq: float [Hz]
       sigma: float

   Input:
       signal: FloatArray

   Output:
       spikes: SpikeArray


   .. py:attribute:: config
      :type:  TopologicalPoissonSpikerConfig


   .. py:attribute:: resolution


   .. py:attribute:: max_freq


   .. py:attribute:: sigma


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(signal)

      Input interface operation.

      Input: A FloatArray of values in the range [mins, maxs].
      Output: A SpikeArray of the same shape as the input.



.. py:class:: TopologicalLinearSpikerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`TopologicalSpikerConfig`, :py:obj:`spark.nn.interfaces.input.linear.LinearSpikerConfig`


   TopologicalLinearSpiker configuration class.


.. py:class:: TopologicalLinearSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Transforms a continuous signal to a spiking signal.
   This transformation maps a vector (a point in a hypercube) into a simple manifold with/without its borders glued.
   This transformation assumes a very simple linear neuron model without any type of adaptation or plasticity.

   Init:
       glue: jax.Array
       mins: jax.Array
       maxs: jax.Array
       resolution: int
       tau: float [ms]
       cd: float [ms]
       max_freq: float [Hz]
       sigma: float

   Input:
       signal: FloatArray

   Output:
       spikes: SpikeArray


   .. py:attribute:: config
      :type:  TopologicalLinearSpikerConfig


   .. py:attribute:: resolution


   .. py:attribute:: tau


   .. py:attribute:: cd


   .. py:attribute:: max_freq


   .. py:attribute:: sigma


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: __call__(signal)

      Input interface operation.

      Input: A FloatArray of values in the range [mins, maxs].
      Output: A SpikeArray of the same shape as the input.



