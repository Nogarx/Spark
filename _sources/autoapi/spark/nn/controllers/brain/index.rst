spark.nn.controllers.brain
==========================

.. py:module:: spark.nn.controllers.brain


Classes
-------

.. autoapisummary::

   spark.nn.controllers.brain.BrainMeta
   spark.nn.controllers.brain.BrainConfig
   spark.nn.controllers.brain.Brain


Module Contents
---------------

.. py:class:: BrainMeta

   Bases: :py:obj:`spark.nn.controllers.base.ControllerMeta`


   Metaclass for `Brain`.


.. py:class:: BrainConfig

   Bases: :py:obj:`spark.nn.controllers.base.ControllerConfig`


   Configuration for `Brain`.

   :param modules_specs: The neurons and interfaces the brain holds, and how their ports are wired.
   :type modules_specs: tuple of ModuleSpecs
   :param seed: Seed for the random draws of the brain and its modules. Drawn from the operating
                system when omitted.
   :type seed: int, optional
   :param dt: Integration step, in ms. Handed down to every module.
   :type dt: float, default 1.0


.. py:class:: Brain(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.controllers.base.Controller`


   Network of neurons and interfaces.

   Every module reads from a cache holding the outputs of the previous step, updates its own
   state, and writes its outputs back. Because no module waits for another, any wiring is
   legal, cycles included, and the modules of one step are independent of each other.
   Note that due to implementation details, modules within this controller have a one step
   latency per connection; which, for most cases, is negligible.

   :param config: Controller configuration. Its fields may also be given as keyword arguments.
   :type config: BrainConfig, optional

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Derived from the modules: a module input wired to ``__call__`` becomes an input port of
                 the controller, under the name the port map gives it.

   :Output Ports: **\*\*outputs** (*SparkPayload*) -- Derived from the modules: a module output named in ``outputs`` becomes an output port of
                  the controller.

   .. rubric:: Notes

   Input and output ports are derived from the modules, as for any controller.

   .. seealso::

      :py:obj:`Neuron`
          Controller without the cache, where a step is ordered by its dependencies.


   .. py:attribute:: config
      :type:  BrainConfig


   .. py:method:: build(**abc_args)


   .. py:method:: __call__(**inputs)

      Advances every module one step against the cache.

      Every module reads the outputs of the previous step, so the modules of one step are
      independent of each other. The cache is written once all of them have run.

      :param \*\*inputs: One entry per input port of the brain, as derived from the modules.
      :type \*\*inputs: SparkPayload

      :returns: One entry per output port of the brain, as derived from the modules.
      :rtype: dict of str to SparkPayload



   .. py:method:: read_state(port_list)

      Returns the current state of the modules/cache.



