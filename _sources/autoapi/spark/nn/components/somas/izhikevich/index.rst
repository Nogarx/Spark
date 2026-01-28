spark.nn.components.somas.izhikevich
====================================

.. py:module:: spark.nn.components.somas.izhikevich


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.izhikevich.IzhikevichSomaConfig
   spark.nn.components.somas.izhikevich.IzhikevichSoma


Module Contents
---------------

.. py:class:: IzhikevichSomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.base.SomaConfig`


   IzhikevichSoma model configuration class.


   .. py:attribute:: potential_rest
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: potential_reset
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: resistance
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: threshold
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: recovery_timescale
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: recovery_sensitivity
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: recovery_update
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: IzhikevichSoma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.base.Soma`


   Izhikevich soma model.

   Init:
       units: tuple[int, ...]
       potential_rest: float | jax.Array
       potential_reset: float | jax.Array
       resistance: float | jax.Array
       threshold: float | jax.Array
       recovery_timescale: float | jax.Array
       recovery_sensitivity: float | jax.Array
       recovery_update: float | jax.Array

   Input:
       in_spikes: SpikeArray

   Output:
       out_spikes: SpikeArray

   Reference:
       Simple Model of Spiking Neurons
       Eugene M. Izhikevich
       IEEE Transactions on Neural Networks, vol. 14, no. 6, pp. 1569-1572, Nov. 2003
       https://doi.org/10.1109/TNN.2003.820440


   .. py:attribute:: config
      :type:  IzhikevichSomaConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



