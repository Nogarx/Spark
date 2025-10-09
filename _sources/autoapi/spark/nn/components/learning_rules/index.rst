spark.nn.components.learning_rules
==================================

.. py:module:: spark.nn.components.learning_rules


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/learning_rules/base/index
   /autoapi/spark/nn/components/learning_rules/hebbian_rule/index
   /autoapi/spark/nn/components/learning_rules/zenke_rule/index


Classes
-------

.. autoapisummary::

   spark.nn.components.learning_rules.LearningRule
   spark.nn.components.learning_rules.LearningRuleOutput
   spark.nn.components.learning_rules.ZenkeRule
   spark.nn.components.learning_rules.ZenkeRuleConfig
   spark.nn.components.learning_rules.HebbianRule
   spark.nn.components.learning_rules.HebbianRuleConfig
   spark.nn.components.learning_rules.OjaRule
   spark.nn.components.learning_rules.OjaRuleConfig


Package Contents
----------------

.. py:class:: LearningRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract learning rule model.


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
       async_spikes: bool
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       post_slow_tau: float | jax.Array
       target_tau: float | jax.Array
       a: float | jax.Array
       b: float | jax.Array
       c: float | jax.Array
       d: float | jax.Array
       P: float | jax.Array
       gamma: float | jax.Array

   Input:
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       current_kernel: FloatArray

   Output:
       kernel: FloatArray


   .. py:attribute:: config
      :type:  ZenkeRuleConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(pre_spikes, post_spikes, current_kernel)

      Computes and returns the next kernel update.



.. py:class:: ZenkeRuleConfig(**kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRuleConfig`


   ZenkeRule configuration class.


   .. py:attribute:: async_spikes
      :type:  bool


   .. py:attribute:: pre_tau
      :type:  float | jax.Array


   .. py:attribute:: post_tau
      :type:  float | jax.Array


   .. py:attribute:: post_slow_tau
      :type:  float | jax.Array


   .. py:attribute:: target_tau
      :type:  float | jax.Array


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


   .. py:attribute:: gamma
      :type:  float | jax.Array


.. py:class:: HebbianRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRule`


   Hebbian plasticy rule model.

   Init:
       async_spikes: bool
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       gamma: float | jax.Array

   Input:
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       current_kernel: FloatArray

   Output:
       kernel: FloatArray


   .. py:attribute:: config
      :type:  HebbianRuleConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(pre_spikes, post_spikes, current_kernel)

      Computes and returns the next kernel update.



.. py:class:: HebbianRuleConfig(**kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRuleConfig`


   HebbianRule configuration class.


   .. py:attribute:: async_spikes
      :type:  bool


   .. py:attribute:: pre_tau
      :type:  float | jax.Array


   .. py:attribute:: post_tau
      :type:  float | jax.Array


   .. py:attribute:: gamma
      :type:  float | jax.Array


.. py:class:: OjaRule(config = None, **kwargs)

   Bases: :py:obj:`HebbianRule`


   Oja's plasticy rule model.

   Init:
       async_spikes: bool
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       gamma: float | jax.Array

   Input:
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       current_kernel: FloatArray

   Output:
       kernel: FloatArray


.. py:class:: OjaRuleConfig(**kwargs)

   Bases: :py:obj:`HebbianRuleConfig`


   HebbianRule configuration class.


   .. py:attribute:: stabilization_factor
      :type:  bool


