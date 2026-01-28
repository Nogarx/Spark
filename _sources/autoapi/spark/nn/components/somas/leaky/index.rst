spark.nn.components.somas.leaky
===============================

.. py:module:: spark.nn.components.somas.leaky


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.leaky.LeakySomaConfig
   spark.nn.components.somas.leaky.LeakySoma
   spark.nn.components.somas.leaky.RefractoryLeakySomaConfig
   spark.nn.components.somas.leaky.RefractoryLeakySoma
   spark.nn.components.somas.leaky.StrictRefractoryLeakySomaConfig
   spark.nn.components.somas.leaky.StrictRefractoryLeakySoma
   spark.nn.components.somas.leaky.AdaptiveLeakySomaConfig
   spark.nn.components.somas.leaky.AdaptiveLeakySoma


Module Contents
---------------

.. py:class:: LeakySomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.base.SomaConfig`


   LeakySoma model configuration class.


   .. py:attribute:: potential_rest
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: potential_reset
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: potential_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: resistance
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: threshold
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: LeakySoma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.base.Soma`


   Leaky soma model.

   Init:
       units: tuple[int, ...]
       potential_rest: float | jax.Array
       potential_reset: float | jax.Array
       potential_tau: float | jax.Array
       resistance: float | jax.Array
       threshold: float | jax.Array

   Input:
       in_spikes: SpikeArray

   Output:
       out_spikes: SpikeArray

   Reference:
       Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
       Gerstner W, Kistler WM, Naud R, Paninski L.
       Chapter 1.3 Integrate-And-Fire Models
       https://neuronaldynamics.epfl.ch/online/Ch1.S3.html


   .. py:attribute:: config
      :type:  LeakySomaConfig


   .. py:method:: build(input_specs)

      Build method.



.. py:class:: RefractoryLeakySomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`LeakySomaConfig`


   RefractoryLeakySoma model configuration class.


   .. py:attribute:: cooldown
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: RefractoryLeakySoma(config = None, **kwargs)

   Bases: :py:obj:`LeakySoma`


   Leaky soma with refractory time model.

   Init:
       units: tuple[int, ...]
       potential_rest: float | jax.Array
       potential_reset: float | jax.Array
       potential_tau: float | jax.Array
       resistance: float | jax.Array
       threshold: float | jax.Array
       cooldown: float | jax.Array

   Input:
       in_spikes: SpikeArray

   Output:
       out_spikes: SpikeArray

   Reference:
       Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
       Gerstner W, Kistler WM, Naud R, Paninski L.
       Chapter 1.3 Integrate-And-Fire Models
       https://neuronaldynamics.epfl.ch/online/Ch1.S3.html


   .. py:attribute:: config
      :type:  RefractoryLeakySomaConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



.. py:class:: StrictRefractoryLeakySomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`RefractoryLeakySomaConfig`


   StrictRefractoryLeakySoma model configuration class.


.. py:class:: StrictRefractoryLeakySoma(config = None, **kwargs)

   Bases: :py:obj:`RefractoryLeakySoma`


   Leaky soma with strict refractory time model.
   Note: This model is here mostly for didactic/historical reasons.

   Init:
       units: tuple[int, ...]
       potential_rest: float | jax.Array
       potential_reset: float | jax.Array
       potential_tau: float | jax.Array
       resistance: float | jax.Array
       threshold: float | jax.Array
       cooldown: float | jax.Array

   Input:
       in_spikes: SpikeArray

   Output:
       out_spikes: SpikeArray

   Reference:
       Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
       Gerstner W, Kistler WM, Naud R, Paninski L.
       Chapter 1.3 Integrate-And-Fire Models
       https://neuronaldynamics.epfl.ch/online/Ch1.S3.html


   .. py:attribute:: config
      :type:  StrictRefractoryLeakySomaConfig


.. py:class:: AdaptiveLeakySomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`RefractoryLeakySomaConfig`


   AdaptiveLeakySoma model configuration class.


   .. py:attribute:: threshold_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: threshold_delta
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: AdaptiveLeakySoma(config = None, **kwargs)

   Bases: :py:obj:`RefractoryLeakySoma`


   Adaptive leaky soma model.

   Init:
       units: tuple[int, ...]
       potential_rest: float | jax.Array
       potential_reset: float | jax.Array
       potential_tau: float | jax.Array
       resistance: float | jax.Array
       threshold: float | jax.Array
       cooldown: float | jax.Array
       threshold_tau: float | jax.Array
       threshold_delta: float | jax.Array

   Input:
       in_spikes: SpikeArray

   Output:
       out_spikes: SpikeArray

   Reference:
       Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
       Gerstner W, Kistler WM, Naud R, Paninski L.
       Chapter 5.1 Thresholds in a nonlinear integrate-and-fire model
       https://neuronaldynamics.epfl.ch/online/Ch5.S1.html


   .. py:attribute:: config
      :type:  AdaptiveLeakySomaConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



