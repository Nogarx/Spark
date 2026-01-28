spark.nn.components.learning_rules.quadruplet_rule
==================================================

.. py:module:: spark.nn.components.learning_rules.quadruplet_rule


Classes
-------

.. autoapisummary::

   spark.nn.components.learning_rules.quadruplet_rule.QuadrupletRuleConfig
   spark.nn.components.learning_rules.quadruplet_rule.QuadrupletRule
   spark.nn.components.learning_rules.quadruplet_rule.QuadrupletRuleTensorConfig
   spark.nn.components.learning_rules.quadruplet_rule.QuadrupletRuleTensor


Module Contents
---------------

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



