spark.nn.components.delays
==========================

.. py:module:: spark.nn.components.delays


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/delays/base/index
   /autoapi/spark/nn/components/delays/n2n_delays/index
   /autoapi/spark/nn/components/delays/n_delays/index


Classes
-------

.. autoapisummary::

   spark.nn.components.delays.Delays
   spark.nn.components.delays.DelaysOutput
   spark.nn.components.delays.NDelays
   spark.nn.components.delays.NDelaysConfig
   spark.nn.components.delays.N2NDelays
   spark.nn.components.delays.N2NDelaysConfig


Package Contents
----------------

.. py:class:: Delays(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract synaptic delay model.


   .. py:method:: reset()
      :abstractmethod:


      Resets component state.



   .. py:method:: __call__(in_spikes)
      :abstractmethod:


      Execution method.



.. py:class:: DelaysOutput

   Bases: :py:obj:`TypedDict`


   Generic delay model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: out_spikes
      :type:  spark.core.payloads.SpikeArray


.. py:class:: NDelays(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.delays.base.Delays`


   Data structure for spike storage and retrival for efficient neuron spike delay implementation.
   This synaptic delay model implements a generic conduction delay of the outputs spikes of neruons.
   Example: Neuron A fires, every neuron that listens to A recieves its spikes K timesteps later,
           neuron B fires, every neuron that listens to B recieves its spikes L timesteps later.

   Init:
       max_delay: float
       delay_initializer: DelayInitializerConfig

   Input:
       in_spikes: SpikeArray

   Output:
       out_spikes: SpikeArray


   .. py:attribute:: config
      :type:  NDelaysConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: get_dense()

      Convert bitmask to dense vector (aligned with MSB-first packing).



   .. py:method:: __call__(in_spikes)

      Execution method.



.. py:class:: NDelaysConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.delays.base.DelaysConfig`


   NDelays configuration class.


   .. py:attribute:: max_delay
      :type:  float


   .. py:attribute:: delays
      :type:  jax.Array | spark.nn.initializers.base.Initializer


.. py:class:: N2NDelays(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.delays.base.Delays`


   Data structure for spike storage and retrival for efficient neuron to neuron spike delay implementation.
   This synaptic delay model implements specific conduction delays between specific neruons.
   Example: Neuron A fires and neuron B, C, and D listens to A; neuron B recieves A's spikes I timesteps later,
            neuron C recieves A's spikes J timesteps later and neuron D recieves A's spikes K timesteps later.

   Init:
       units: tuple[int, ...]
       max_delay: float
       delays: jnp.ndarray | Initializer

   Input:
       in_spikes: SpikeArray

   Output:
       out_spikes: SpikeArray


   .. py:attribute:: config
      :type:  N2NDelaysConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: get_dense()

      Convert bitmask to dense vector (aligned with MSB-first packing).



   .. py:method:: __call__(in_spikes)

      Execution method.



.. py:class:: N2NDelaysConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.delays.n_delays.NDelaysConfig`


   N2NDelays configuration class.


   .. py:attribute:: units
      :type:  tuple[int, Ellipsis]


