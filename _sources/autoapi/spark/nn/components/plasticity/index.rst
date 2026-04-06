spark.nn.components.plasticity
==============================

.. py:module:: spark.nn.components.plasticity


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/plasticity/base/index
   /autoapi/spark/nn/components/plasticity/hebbian_rule/index
   /autoapi/spark/nn/components/plasticity/quadruplet_rule/index
   /autoapi/spark/nn/components/plasticity/three_factor_rule/index
   /autoapi/spark/nn/components/plasticity/zenke_rule/index


Classes
-------

.. autoapisummary::

   spark.nn.components.plasticity.Plasticity
   spark.nn.components.plasticity.PlasticityConfig
   spark.nn.components.plasticity.PlasticityOutput
   spark.nn.components.plasticity.ZenkeRule
   spark.nn.components.plasticity.ZenkeRuleConfig
   spark.nn.components.plasticity.HebbianRule
   spark.nn.components.plasticity.HebbianRuleConfig
   spark.nn.components.plasticity.OjaRule
   spark.nn.components.plasticity.OjaRuleConfig
   spark.nn.components.plasticity.QuadrupletRule
   spark.nn.components.plasticity.QuadrupletRuleConfig
   spark.nn.components.plasticity.QuadrupletRuleTensor
   spark.nn.components.plasticity.QuadrupletRuleTensorConfig
   spark.nn.components.plasticity.ThreeFactorHebbianRule
   spark.nn.components.plasticity.ThreeFactorHebbianRuleConfig


Package Contents
----------------

.. py:class:: Plasticity(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract plasticity rule model.


.. py:class:: PlasticityConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Abstract plasticity rule configuration class.


.. py:class:: PlasticityOutput

   Bases: :py:obj:`TypedDict`


   Generic plasticity rule model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: kernel
      :type:  spark.core.payloads.FloatArray


.. py:class:: ZenkeRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Zenke plasticy rule model. This model is an extension of the classic Hebbian Rule.

   Init:
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       post_slow_tau: float | jax.Array
       target_tau: float | jax.Array
       a: float | jax.Array
       b: float | jax.Array
       c: float | jax.Array
       d: float | jax.Array
       P: float | jax.Array
       eta: float | jax.Array

   Input:
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       kernel: FloatArray

   Output:
       kernel: FloatArray


   .. py:attribute:: config
      :type:  ZenkeRuleConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(pre_spikes, post_spikes, kernel)

      Computes and returns the next kernel update.



.. py:class:: ZenkeRuleConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   ZenkeRule configuration class.


   .. py:attribute:: pre_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: post_slow_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: target_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: a
      :type:  float | jax.Array


   .. py:attribute:: b
      :type:  float | jax.Array


   .. py:attribute:: c
      :type:  float | jax.Array


   .. py:attribute:: d
      :type:  float | jax.Array


   .. py:attribute:: p
      :type:  float | jax.Array


   .. py:attribute:: eta
      :type:  float | jax.Array


.. py:class:: HebbianRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Hebbian plasticy rule model.

   Init:
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       gamma: float | jax.Array

   Input:
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       kernel: FloatArray

   Output:
       kernel: FloatArray


   .. py:attribute:: config
      :type:  HebbianRuleConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(pre_spikes, post_spikes, kernel)

      Computes and returns the next kernel update.



.. py:class:: HebbianRuleConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   HebbianRule configuration class.


   .. py:attribute:: pre_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: eta
      :type:  float


.. py:class:: OjaRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Oja's plasticy rule model.

   Init:
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       gamma: float | jax.Array

   Input:
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       kernel: FloatArray

   Output:
       kernel: FloatArray


   .. py:attribute:: config
      :type:  OjaRuleConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(pre_spikes, post_spikes, kernel)

      Computes and returns the next kernel update.



.. py:class:: OjaRuleConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   Abstract plasticity rule configuration class.


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: eta
      :type:  float


.. py:class:: QuadrupletRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Quadruplet plasticy rule model.

   Init:
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       q_alpha: float | jax.Array
       q_beta: float | jax.Array
       q_gamma: float | jax.Array
       q_delta: float | jax.Array
       eta: float | jax.Array

   Input:
       modulation: FloatArray
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       kernel: FloatArray

   Output:
       kernel: FloatArray


   .. py:attribute:: config
      :type:  QuadrupletRuleConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(modulation, pre_spikes, post_spikes, kernel)

      Computes and returns the next kernel update.



.. py:class:: QuadrupletRuleConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   QuadrupletRule configuration class.


   .. py:attribute:: pre_tau
      :type:  float | jax.Array


   .. py:attribute:: post_tau
      :type:  float | jax.Array


   .. py:attribute:: q_alpha
      :type:  float | jax.Array


   .. py:attribute:: q_beta
      :type:  float | jax.Array


   .. py:attribute:: q_gamma
      :type:  float | jax.Array


   .. py:attribute:: q_delta
      :type:  float | jax.Array


   .. py:attribute:: gamma
      :type:  float | jax.Array


.. py:class:: QuadrupletRuleTensor(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Quadruplet plasticy rule model (tensor).

   Init:
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       eta: float | jax.Array

   Input:
       modulation: FloatArray
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       kernel: FloatArray

   Output:
       kernel: FloatArray


   .. py:attribute:: config
      :type:  QuadrupletRuleTensorConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(modulation, pre_spikes, post_spikes, kernel)

      Computes and returns the next kernel update.



.. py:class:: QuadrupletRuleTensorConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   QuadrupletRuleTensor configuration class.


   .. py:attribute:: pre_tau
      :type:  tuple[float, float, float, float]


   .. py:attribute:: post_tau
      :type:  tuple[float, float, float, float]


   .. py:attribute:: q_alpha
      :type:  tuple[float, float, float, float]


   .. py:attribute:: q_beta
      :type:  tuple[float, float, float, float]


   .. py:attribute:: q_gamma
      :type:  tuple[float, float, float, float]


   .. py:attribute:: q_delta
      :type:  tuple[float, float, float, float]


   .. py:attribute:: max_clip
      :type:  tuple[float, float, float, float]


   .. py:attribute:: eta
      :type:  float


.. py:class:: ThreeFactorHebbianRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.Plasticity`


   Three-factor Hebbian plasticy rule model.

   Init:
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       gamma: float | jax.Array

   Input:
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       kernel: FloatArray

   Output:
       kernel: FloatArray


   .. py:attribute:: config
      :type:  ThreeFactorHebbianRuleConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(reward, pre_spikes, post_spikes, kernel)

      Computes and returns the next kernel update.



.. py:class:: ThreeFactorHebbianRuleConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   ThreeFactorHebbianRule configuration class.


   .. py:attribute:: pre_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: gamma
      :type:  float | jax.Array


