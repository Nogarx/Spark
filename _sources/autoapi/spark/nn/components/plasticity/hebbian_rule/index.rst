spark.nn.components.plasticity.hebbian_rule
===========================================

.. py:module:: spark.nn.components.plasticity.hebbian_rule


Classes
-------

.. autoapisummary::

   spark.nn.components.plasticity.hebbian_rule.HebbianRuleConfig
   spark.nn.components.plasticity.hebbian_rule.HebbianRule
   spark.nn.components.plasticity.hebbian_rule.OjaRuleConfig
   spark.nn.components.plasticity.hebbian_rule.OjaRule


Module Contents
---------------

.. py:class:: HebbianRuleConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   HebbianRule configuration class.


   .. py:attribute:: pre_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: eta
      :type:  float


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



.. py:class:: OjaRuleConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.plasticity.base.PlasticityConfig`


   Abstract plasticity rule configuration class.


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



