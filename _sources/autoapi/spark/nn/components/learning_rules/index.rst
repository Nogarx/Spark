spark.nn.components.learning_rules
==================================

.. py:module:: spark.nn.components.learning_rules


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/learning_rules/base/index
   /autoapi/spark/nn/components/learning_rules/hebbian_rule/index
   /autoapi/spark/nn/components/learning_rules/quadruplet_rule/index
   /autoapi/spark/nn/components/learning_rules/three_factor_rule/index
   /autoapi/spark/nn/components/learning_rules/zenke_rule/index


Classes
-------

.. autoapisummary::

   spark.nn.components.learning_rules.LearningRule
   spark.nn.components.learning_rules.LearningRuleConfig
   spark.nn.components.learning_rules.LearningRuleOutput
   spark.nn.components.learning_rules.ZenkeRule
   spark.nn.components.learning_rules.ZenkeRuleConfig
   spark.nn.components.learning_rules.HebbianRule
   spark.nn.components.learning_rules.HebbianRuleConfig
   spark.nn.components.learning_rules.OjaRule
   spark.nn.components.learning_rules.OjaRuleConfig
   spark.nn.components.learning_rules.QuadrupletRule
   spark.nn.components.learning_rules.QuadrupletRuleConfig
   spark.nn.components.learning_rules.QuadrupletRuleTensor
   spark.nn.components.learning_rules.QuadrupletRuleTensorConfig
   spark.nn.components.learning_rules.ThreeFactorHebbianRule
   spark.nn.components.learning_rules.ThreeFactorHebbianRuleConfig


Package Contents
----------------

.. py:class:: LearningRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract learning rule model.


.. py:class:: LearningRuleConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Abstract learning rule configuration class.


.. py:class:: LearningRuleOutput

   Bases: :py:obj:`TypedDict`


   Generic learning rule model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: kernel
      :type:  spark.core.payloads.FloatArray


.. py:class:: ZenkeRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRule`


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



.. py:class:: ZenkeRuleConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRuleConfig`


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

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRule`


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



.. py:class:: HebbianRuleConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRuleConfig`


   HebbianRule configuration class.


   .. py:attribute:: pre_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: eta
      :type:  float


.. py:class:: OjaRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRule`


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



.. py:class:: OjaRuleConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRuleConfig`


   Abstract learning rule configuration class.


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: eta
      :type:  float


.. py:class:: QuadrupletRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRule`


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



.. py:class:: QuadrupletRuleConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRuleConfig`


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

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRule`


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



.. py:class:: QuadrupletRuleTensorConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRuleConfig`


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

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRule`


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



.. py:class:: ThreeFactorHebbianRuleConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRuleConfig`


   ThreeFactorHebbianRule configuration class.


   .. py:attribute:: pre_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: gamma
      :type:  float | jax.Array


