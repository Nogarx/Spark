spark.nn.controllers.partition
==============================

.. py:module:: spark.nn.controllers.partition


Classes
-------

.. autoapisummary::

   spark.nn.controllers.partition.Partition


Functions
---------

.. autoapisummary::

   spark.nn.controllers.partition.crossing_name


Module Contents
---------------

.. py:function:: crossing_name(origin, port)

   Returns the name of an output that crosses devices, as an input and an output of the sub-brains.

   :param origin: Name of the module producing the output.
   :type origin: str
   :param port: Name of the output port of that module.
   :type port: str

   :returns: ``'<origin>:<port>'``, a name that is not a Python identifier and cannot collide with
             the ports of the brain.
   :rtype: str


.. py:class:: Partition(brain, parts)

   A brain divided among devices by its modules.

   Each device runs a part of the brain: a sub-brain, a `Brain` holding the modules of that
   part, with their configurations and their state. A module input wired to a module of
   another part becomes an input of the sub-brain, and the output it reads becomes an output of
   the sub-brain of that other part, both named ``'<origin>:<port>'``. The inputs and the
   outputs of the brain stay with the modules reading and producing them.

   In a brain, every module reads what the others produced on the step before, so the
   sub-brains of a step do not wait for each other: each device runs its own, and `exchange`
   copies the outputs that cross devices once every sub-brain has run. A step of every
   sub-brain followed by an exchange is a step of the brain.

   :param brain: A built brain. The sub-brains hold its modules.
   :type brain: Brain
   :param parts: The names of the modules each device runs. Every module of the brain is in one part.
   :type parts: dict of jax.Device to iterable of str

   :raises RuntimeError: If the brain is not built.
   :raises ValueError: If a module is in no part or in two, if a part is empty or names a module the brain
       does not have, or if a module reads a property of a module of another part, or has a
       property written by one.

   .. rubric:: Notes

   Properties are read and effects are applied within a step, so a module reading a property
   of another, or whose property another writes, is on the same device as that module.

   A sub-brain is called as the brain is, within ``jax.jit`` and ``jax.lax.scan``; a call
   compiles for the device its state is on and returns at once, so the devices run together.

   The devices may belong to several processes (``jax.distributed``). Every process builds
   the same brain, from a configuration with its seed, and the same partition, and runs the
   sub-brains of its own devices, `local_devices`. `initial`, `exchange` and `merge` gather
   what crosses processes with an all-gather, which every process calls in the same order;
   an output read within its own process is copied from device to device. On the processor,
   the processes exchange through gloo (``jax_cpu_collectives_implementation``).

   `run` calls the sub-brains of the devices of the process, and records them while a recorder
   of `spark.recording` is open, as a call running the brain is recorded.

   .. rubric:: Examples

   >>> partition = spark.Partition(brain, {gpu0: ['spiker', 'first_pool'], gpu1: ['second_pool', 'readout']})
   >>> graphs, states = partition.split(brain)
   >>> received = partition.initial(states)
   >>> for _ in range(steps):
   ...     outputs, states = partition.run(run, graphs, states, received, inputs, 1)
   ...     received = partition.exchange(outputs)
   >>> brain = partition.merge(states)

   ``run`` is the function running the brain, ``run(graph, state, steps, **inputs)``, which
   returns the outputs and the state. Called by hand, the loop over the devices is

   >>> outputs = {}
   >>> for device in partition.local_devices:
   ...     outputs[device], states[device] = run(graphs[device], states[device], 1,
   ...                                           **partition.inputs(device, inputs), **received[device])


   .. py:method:: balanced(brain, devices, tolerance = 0.1)
      :classmethod:


      Returns a partition of a brain over devices that balances their work and keeps what
      crosses devices small.

      :param brain: A built brain.
      :type brain: Brain
      :param devices: The devices to divide the brain among, in order of preference. A device gets no part
                      if the brain has fewer modules to place than devices.
      :type devices: sequence of jax.Device
      :param tolerance: How much more work than the lightest device a device may take, as a fraction, to hold
                        a module next to those it exchanges outputs with.
      :type tolerance: float, default 0.1

      :rtype: Partition

      :raises RuntimeError: If the brain is not built.
      :raises ValueError: If no device is given.

      .. rubric:: Notes

      The work of a module is the number of values in its state, which its weights, traces and
      buffers make up. Modules that read a property of one another, or write one, are placed
      together. The modules are placed from the heaviest, each on the device that keeps the
      work within ``tolerance`` and exchanges the most with it; then a module is moved to
      another device while the move lowers what crosses devices and keeps every device within
      ``1 + tolerance`` times the heaviest load, or an even share if that is more. What crosses
      is counted in values per step: the size of an output, once for every other device
      reading it.

      The parts depend on the brain and the devices only, so every process gets the same.



   .. py:property:: devices
      :type: tuple[jax.Device, ...]


      The devices of the partition, in the order the parts were given, in every process.


   .. py:property:: local_devices
      :type: tuple[jax.Device, ...]


      The devices of the partition that belong to this process, in the order the parts were
      given: every device, with one process.


   .. py:property:: parts
      :type: dict[jax.Device, tuple[str, ...]]


      The names of the modules each device runs, in the order of the brain.


   .. py:property:: configs
      :type: dict[jax.Device, spark.nn.controllers.brain.BrainConfig]


      The configuration of the sub-brain of each device.


   .. py:method:: device_of(name)

      Returns the device that runs the module ``name``.



   .. py:method:: split(brain)

      Returns the graph and the state of the sub-brain of every device of this process, the
      state placed on its device.

      :param brain: The brain the partition was made from, or one of the same structure, such as one
                    returned by `merge`. Its modules give the state of the sub-brains.
      :type brain: Brain

      :returns: * **graphs** (*dict of jax.Device to GraphDef*) -- The graph of each sub-brain.
                * **states** (*dict of jax.Device to State*) -- The state of each sub-brain, on its device.



   .. py:method:: merge(states, device = None)

      Returns the brain whose modules have the states of the sub-brains.

      With several processes, every process calls it, and every process gets the brain: the
      states of the sub-brains of the other processes are gathered.

      :param states: The state of the sub-brain of each device of this process.
      :type states: dict of jax.Device to State
      :param device: The device the brain is placed on. The first device of the partition in this process
                     when omitted.
      :type device: jax.Device, optional

      :returns: The brain, as a brain that ran unpartitioned would be.
      :rtype: Brain



   .. py:method:: run(function, graphs, states, received, inputs, *args, **kwargs)

      Runs the sub-brain of every device of this process, and returns their outputs and states.

      :param function: Runs a model: ``function(graph, state, *args, **inputs, **kwargs)`` returns its outputs and
                       its state, as the loop running the brain does. A `spark.jit` function is recorded.
      :type function: callable
      :param graphs: The graph of each sub-brain, as `split` gives them.
      :type graphs: dict of jax.Device to GraphDef
      :param states: The state of each sub-brain.
      :type states: dict of jax.Device to State
      :param received: What each sub-brain reads from the others, as `initial` or `exchange` gives it.
      :type received: dict of jax.Device to dict of str to SparkPayload
      :param inputs: The inputs of the brain, by name.
      :type inputs: dict of str to SparkPayload
      :param \*args: Passed to ``function`` after the graph and the state, such as the steps of the call.
      :param \*\*kwargs: Passed to ``function`` after the graph and the state, such as the steps of the call.

      :returns: * **outputs** (*dict of jax.Device to dict of str to SparkPayload*) -- The outputs ``function`` returned for each device.
                * **states** (*dict of jax.Device to State*) -- The state of each sub-brain after the call.

      :raises ValueError: With an open recorder, when a probe addresses the brain rather than a module or an
          input of the brain.
      :raises NotImplementedError: With an open recorder, when the partition spans several processes.

      .. rubric:: Notes

      While a recorder of `spark.recording` is open, the calls of a `spark.jit` function on the
      devices are one call of the recorder, as a call running the brain is. Each records the
      probes of the modules of its sub-brain, and an input of the brain is recorded by the first
      device reading it. The records are joined on the first device and handed over once. The
      calls on the devices run the same steps.



   .. py:method:: inputs(device, inputs)

      Returns the inputs of the brain that the sub-brain of ``device`` reads, placed on that device.

      :param device: A device of the partition in this process.
      :type device: jax.Device
      :param inputs: The inputs of the brain, by name.
      :type inputs: dict of str to SparkPayload

      :rtype: dict of str to SparkPayload



   .. py:method:: initial(states)

      Returns what each sub-brain reads from the others on its next step, from their states.

      The cache of a sub-brain holds the outputs of its last step, so this is also what
      `exchange` returns after that step.

      :param states: The state of the sub-brain of each device of this process.
      :type states: dict of jax.Device to State

      :returns: For each device of this process, the outputs of the other parts its sub-brain reads,
                on that device.
      :rtype: dict of jax.Device to dict of str to SparkPayload



   .. py:method:: exchange(outputs)

      Copies the outputs that cross devices to the devices reading them.

      :param outputs: The outputs of the step of the sub-brain of each device of this process.
      :type outputs: dict of jax.Device to dict of str to SparkPayload

      :returns: For each device of this process, the outputs of the other parts its sub-brain reads,
                on that device. The copies between devices of one process are asynchronous; those
                between processes wait for every process to have run its step.
      :rtype: dict of jax.Device to dict of str to SparkPayload



   .. py:method:: outputs(outputs)

      Returns the outputs of the brain, from the outputs of the sub-brains.

      :param outputs: The outputs of the sub-brain of each device of this process.
      :type outputs: dict of jax.Device to dict of str to SparkPayload

      :returns: One entry per output port of the brain produced in this process, on the device
                producing it.
      :rtype: dict of str to SparkPayload



