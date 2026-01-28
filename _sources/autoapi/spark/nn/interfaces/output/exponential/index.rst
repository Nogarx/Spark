spark.nn.interfaces.output.exponential
======================================

.. py:module:: spark.nn.interfaces.output.exponential


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.output.exponential.ExponentialIntegratorConfig
   spark.nn.interfaces.output.exponential.ExponentialIntegrator


Module Contents
---------------

.. py:class:: ExponentialIntegratorConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.output.base.OutputInterfaceConfig`


   ExponentialIntegrator configuration class.


   .. py:attribute:: num_outputs
      :type:  int


   .. py:attribute:: saturation_freq
      :type:  float


   .. py:attribute:: tau
      :type:  float


   .. py:attribute:: output_map
      :type:  jax.Array | None


   .. py:attribute:: shuffle
      :type:  bool


   .. py:attribute:: smooth_trace
      :type:  bool


.. py:class:: ExponentialIntegrator(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.output.base.OutputInterface`


   Transforms a discrete spike signal to a continuous signal.
   This transformation assumes a very simple integration model model without any type of adaptation or plasticity.
   Spikes are grouped into k non-overlaping clusters and every neuron contributes the same amount to the ouput.

   Init:
       num_outputs: int
       saturation_freq: float [Hz]
       tau: float [ms]
       shuffle: bool
       smooth_trace: bool

   Input:
       spikes: SpikeArray

   Output:
       signal: FloatArray


   .. py:attribute:: config
      :type:  ExponentialIntegratorConfig


   .. py:attribute:: num_outputs


   .. py:attribute:: saturation_freq


   .. py:attribute:: tau


   .. py:attribute:: shuffle


   .. py:attribute:: smooth_trace


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: __call__(spikes)

      Transform incomming spikes into a output signal.



