spark.nn.components.somas
=========================

.. py:module:: spark.nn.components.somas


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/somas/base/index
   /autoapi/spark/nn/components/somas/exponential/index
   /autoapi/spark/nn/components/somas/izhikevich/index
   /autoapi/spark/nn/components/somas/leaky/index


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.Soma
   spark.nn.components.somas.SomaOutput
   spark.nn.components.somas.LeakySoma
   spark.nn.components.somas.LeakySomaConfig
   spark.nn.components.somas.RefractoryLeakySoma
   spark.nn.components.somas.RefractoryLeakySomaConfig
   spark.nn.components.somas.StrictRefractoryLeakySoma
   spark.nn.components.somas.StrictRefractoryLeakySomaConfig
   spark.nn.components.somas.AdaptiveLeakySoma
   spark.nn.components.somas.AdaptiveLeakySomaConfig
   spark.nn.components.somas.ExponentialSoma
   spark.nn.components.somas.ExponentialSomaConfig
   spark.nn.components.somas.RefractoryExponentialSoma
   spark.nn.components.somas.RefractoryExponentialSomaConfig
   spark.nn.components.somas.AdaptiveExponentialSoma
   spark.nn.components.somas.AdaptiveExponentialSomaConfig
   spark.nn.components.somas.SimplifiedAdaptiveExponentialSoma
   spark.nn.components.somas.SimplifiedAdaptiveExponentialSomaConfig
   spark.nn.components.somas.IzhikevichSoma
   spark.nn.components.somas.IzhikevichSomaConfig


Package Contents
----------------

.. py:class:: Soma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract soma model.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets neuron states to their initial values.



   .. py:method:: __call__(current)

      Update neuron's states and compute spikes.



.. py:class:: SomaOutput

   Bases: :py:obj:`TypedDict`


   Generic soma model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: spikes
      :type:  spark.core.payloads.SpikeArray


   .. py:attribute:: potential
      :type:  spark.core.payloads.PotentialArray


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



.. py:class:: RefractoryLeakySomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`LeakySomaConfig`


   RefractoryLeakySoma model configuration class.


   .. py:attribute:: cooldown
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


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


.. py:class:: StrictRefractoryLeakySomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`RefractoryLeakySomaConfig`


   StrictRefractoryLeakySoma model configuration class.


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



.. py:class:: AdaptiveLeakySomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`RefractoryLeakySomaConfig`


   AdaptiveLeakySoma model configuration class.


   .. py:attribute:: threshold_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: threshold_delta
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



.. py:class:: RefractoryExponentialSomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`ExponentialSomaConfig`


   RefractoryExponentialSoma model configuration class.


   .. py:attribute:: cooldown
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



.. py:class:: AdaptiveExponentialSomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`ExponentialSomaConfig`


   AdaptiveExponentialSoma model configuration class.


   .. py:attribute:: adaptation_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: adaptation_delta
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: adaptation_subthreshold
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



.. py:class:: SimplifiedAdaptiveExponentialSomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`RefractoryExponentialSomaConfig`


   SimplifiedAdaptiveExponentialSoma model configuration class.


   .. py:attribute:: threshold_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: threshold_delta
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


