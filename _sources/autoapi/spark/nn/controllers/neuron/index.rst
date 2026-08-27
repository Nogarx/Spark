spark.nn.controllers.neuron
===========================

.. py:module:: spark.nn.controllers.neuron


Classes
-------

.. autoapisummary::

   spark.nn.controllers.neuron.NeuronMeta
   spark.nn.controllers.neuron.NeuronConfig
   spark.nn.controllers.neuron.Neuron


Module Contents
---------------

.. py:class:: NeuronMeta

   Bases: :py:obj:`spark.nn.controllers.base.ControllerMeta`


   Metaclass for `Neuron`.


.. py:class:: NeuronConfig

   Bases: :py:obj:`spark.nn.controllers.base.ControllerConfig`


   Configuration for `Neuron`.

   :param modules_specs: The components the neuron holds, and how their ports are wired.
   :type modules_specs: tuple of ModuleSpecs
   :param units: Shape of the pool of neurons.
   :type units: tuple of int
   :param inhibitory_rate: Fraction of the pool that is inhibitory. Must lie in ``[0, 1]``.
   :type inhibitory_rate: float, default 0.2
   :param seed: Seed for the random draws of the neuron and its modules. Drawn from the operating
                system when omitted.
   :type seed: int, optional
   :param dt: Integration step, in ms.
   :type dt: float, default 1.0

   .. rubric:: Notes

   ``dt`` and ``units`` are handed down to every configuration the neuron contains, so a pool
   is sized and clocked in one place. Both names are reserved for that purpose.


   .. py:attribute:: units
      :type:  tuple[int, ...]


   .. py:attribute:: inhibitory_rate
      :type:  float


   .. py:method:: __post_init__()


.. py:class:: Neuron(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.base.Controller`


   Pool of neurons built from components.

   A neuron holds the components one neuron model is made of, such as delays, synapses, a
   soma and a plasticity rule, and steps them in dependency order within a single timestep.
   A module therefore reads what the modules before it produced on the same step, rather than
   on the previous one.

   Because the step is ordered, a cycle in the wiring can only be resolved if the module
   closing it declares a recurrent contract, which states what its outputs look like before
   it has run.

   :param config: Controller configuration. Its fields may also be given as keyword arguments.
   :type config: NeuronConfig

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Derived from the modules: a module input wired to ``__call__`` becomes an input port of
                 the controller, under the name the port map gives it.

   :Output Ports: **\*\*outputs** (*SparkPayload*) -- Derived from the modules: a module output named in ``outputs`` becomes an output port of
                  the controller.

   :Properties: **inhibition_mask** (*BooleanMask*) -- Marks the inhibitory units of the pool. Read only, and supplied to any module declaring
                an ``inhibition_mask`` input without being wired.

   .. rubric:: Notes

   Which units are inhibitory is a property of the pool, not of the module that emits the
   spikes. The mask is drawn once from ``inhibitory_rate`` and handed to any module declaring
   an ``inhibition_mask`` input, without being wired. From there the spikes carry the
   distinction themselves. A declared connection takes precedence.

   ``inhibition_mask`` is exposed as a read-only property, so it can be read by the graph but
   not written.

   .. seealso::

      :py:obj:`Brain`
          Controller that steps its modules against a cache of the previous step.

      :py:obj:`LIFNeuron`
          Prebuilt leaky integrate-and-fire neuron.


   .. py:attribute:: config
      :type:  NeuronConfig


   .. py:attribute:: units
      :value: ()



   .. py:method:: recurrent_contract()

      Returns expected-like outputs and properties of the module.

      This function is a binding contract that allows the modules to accept self connections.



   .. py:method:: has_recurrent_contract()
      :classmethod:


      Returns True if the modules defines a recurrent contract, False otherwise.



   .. py:method:: inhibition_mask()


   .. py:method:: build(**abc_args)


   .. py:method:: __call__(**inputs)

      Advances every module one step, in dependency order.

      A module reads what the modules before it produced on this same step. Effects are applied
      once every module has run.

      :param \*\*inputs: One entry per input port of the neuron, as derived from the modules.
      :type \*\*inputs: SparkPayload

      :returns: One entry per output port of the neuron, as derived from the modules.
      :rtype: dict of str to SparkPayload



   .. py:method:: read_state(port_list)

      Returns the current state of the modules/cache.



