spark.nn.components.somas.exponential
=====================================

.. py:module:: spark.nn.components.somas.exponential


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.exponential.ExponentialSomaConfig
   spark.nn.components.somas.exponential.ExponentialSoma
   spark.nn.components.somas.exponential.RefractoryExponentialSomaConfig
   spark.nn.components.somas.exponential.RefractoryExponentialSoma
   spark.nn.components.somas.exponential.AdaptiveExponentialSomaConfig
   spark.nn.components.somas.exponential.AdaptiveExponentialSoma
   spark.nn.components.somas.exponential.SimplifiedAdaptiveExponentialSomaConfig
   spark.nn.components.somas.exponential.SimplifiedAdaptiveExponentialSoma


Module Contents
---------------

.. py:class:: ExponentialSomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.base.SomaConfig`


   ExponentialSoma model configuration class.


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


   .. py:attribute:: rheobase_threshold
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: spike_slope
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: ExponentialSoma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.somas.base.Soma`


   Exponential soma model.

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
       How Spike Generation Mechanisms Determine the Neuronal Response to Fluctuating Inputs
       Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk, and Nicolas Brunel
       The Journal of Neuroscience, December 17, 2003
       https://www.jneurosci.org/content/23/37/11628
       Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
       Gerstner W, Kistler WM, Naud R, Paninski L.
       Chapter 5.2 Exponential Integrate-and-Fire Model
       https://neuronaldynamics.epfl.ch/online/Ch5.S2.html


   .. py:attribute:: config
      :type:  ExponentialSomaConfig


   .. py:method:: build(input_specs)

      Build method.



.. py:class:: RefractoryExponentialSomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`ExponentialSomaConfig`


   RefractoryExponentialSoma model configuration class.


   .. py:attribute:: cooldown
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: RefractoryExponentialSoma(config = None, **kwargs)

   Bases: :py:obj:`ExponentialSoma`


   Exponential soma with refractory time model.

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
       How Spike Generation Mechanisms Determine the Neuronal Response to Fluctuating Inputs
       Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk, and Nicolas Brunel
       The Journal of Neuroscience, December 17, 2003
       https://www.jneurosci.org/content/23/37/11628
       Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
       Gerstner W, Kistler WM, Naud R, Paninski L.
       Chapter 5.2 Exponential Integrate-and-Fire Model
       https://neuronaldynamics.epfl.ch/online/Ch5.S2.html


   .. py:attribute:: config
      :type:  RefractoryExponentialSomaConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



.. py:class:: AdaptiveExponentialSomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`ExponentialSomaConfig`


   AdaptiveExponentialSoma model configuration class.


   .. py:attribute:: adaptation_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: adaptation_delta
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: adaptation_subthreshold
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: AdaptiveExponentialSoma(config = None, **kwargs)

   Bases: :py:obj:`ExponentialSoma`


   Adaptive Exponential soma model.

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

       Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity.
       Romain Brette and Gerstner Wulfram
       Gerstner W, Kistler WM, Naud R, Paninski L.
       Journal of Neurophysiology vol. 94, no. 5, pp. 3637-3642, 2005
       https://doi.org/10.1152/jn.00686.2005
       Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
       Gerstner W, Kistler WM, Naud R, Paninski L.
       Chapter 5.2 Exponential Integrate-and-Fire Model
       https://neuronaldynamics.epfl.ch/online/Ch5.S2.html


   .. py:attribute:: config
      :type:  AdaptiveExponentialSomaConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets neuron states to their initial values.



.. py:class:: SimplifiedAdaptiveExponentialSomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`RefractoryExponentialSomaConfig`


   SimplifiedAdaptiveExponentialSoma model configuration class.


   .. py:attribute:: threshold_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: threshold_delta
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: SimplifiedAdaptiveExponentialSoma(config = None, **kwargs)

   Bases: :py:obj:`RefractoryExponentialSoma`


   Simplified Adaptive Exponential soma model. This model drops the subthreshold adaptation.

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

       Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity.
       Romain Brette and Gerstner Wulfram
       Gerstner W, Kistler WM, Naud R, Paninski L.
       Journal of Neurophysiology vol. 94, no. 5, pp. 3637-3642, 2005
       https://doi.org/10.1152/jn.00686.2005
       Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
       Gerstner W, Kistler WM, Naud R, Paninski L.
       Chapter 5.2 Exponential Integrate-and-Fire Model
       https://neuronaldynamics.epfl.ch/online/Ch5.S2.html


   .. py:attribute:: config
      :type:  SimplifiedAdaptiveExponentialSomaConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



