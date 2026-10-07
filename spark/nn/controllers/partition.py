#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import copy
import jax
import numpy as np
import flax.nnx as nnx
from jax.experimental import multihost_utils

from spark.core.backend import split, merge
from spark.core.backend.transforms import Jit
from spark.core.recording_hooks import OPEN_RECORDERS, recording_hooks
from spark.core.specs import ModuleSpecs, PortMap
from spark.core.payloads import SparkPayload
from spark.nn.controllers.brain import Brain, BrainConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

_CALL = '__call__'
_SELF = '__self__'
_CACHE = '_cache'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def crossing_name(origin: str, port: str) -> str:
    """
        Returns the name of an output that crosses devices, as an input and an output of the sub-brains.

        Parameters
        ----------
        origin : str
            Name of the module producing the output.
        port : str
            Name of the output port of that module.

        Returns
        -------
        str
            ``'<origin>:<port>'``, a name that is not a Python identifier and cannot collide with
            the ports of the brain.
    """
    return f'{origin}:{port}'

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class Partition:
    """
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

        Parameters
        ----------
        brain : Brain
            A built brain. The sub-brains hold its modules.
        parts : dict of jax.Device to iterable of str
            The names of the modules each device runs. Every module of the brain is in one part.

        Raises
        ------
        RuntimeError
            If the brain is not built.
        ValueError
            If a module is in no part or in two, if a part is empty or names a module the brain
            does not have, or if a module reads a property of a module of another part, or has a
            property written by one.

        Notes
        -----
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

        Examples
        --------
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
    """

    def __init__(self, brain: Brain, parts: dict[jax.Device, tp.Iterable[str]]) -> None:
        if not isinstance(brain, Brain):
            raise TypeError(f'A partition divides a Brain, got "{type(brain).__name__}".')
        if not getattr(brain, '__built__', False):
            raise RuntimeError(
                'The brain is not built: call it once with example inputs before partitioning it.'
            )
        self._names, self._device_of = self._parts(brain, parts)
        self._crossings = self._crossings_of(brain)
        # The output trees of the crossings, to read them back from the caches of the sub-brains.
        self._trees = {key: jax.tree.structure(brain._cache[key]) for key in self._crossings}
        self._outputs = {name: self._device_of[origin] for name, origin, _ in brain._contoller_output_map}
        graph, state = split(brain)
        self._graph, self._fixed_brain = graph, self._without_modules(state, brain._modules_names)
        # What a gather between processes carries: the crossing outputs read in another process than the one
        # producing them, and, for `merge`, every module with its outputs in the cache.
        self._remote = [
            key for key, readers in self._crossings.items()
            if any(device.process_index != self._device_of[key[0]].process_index for device in readers)
        ]
        self._spans_processes = len({device.process_index for device in self._names}) > 1
        self._crossing_templates = {crossing_name(*key): self._template(brain._cache[key]) for key in self._remote}
        self._module_templates = {name: self._template(self._module_of(state, name)) for name in brain._modules_names}
        brain_inputs = brain.get_controller_inputs()
        self._configs: dict[jax.Device, BrainConfig] = {}
        self._inputs: dict[jax.Device, tuple[str, ...]] = {}
        for device in self.devices:
            self._configs[device] = config = self._sub_config(brain, device)
            self._inputs[device] = tuple(name for name in Brain._get_controller_input_specs(config.modules_specs) if name in brain_inputs)
        self._graphs: dict[jax.Device, nnx.GraphDef] = {}
        self._fixed: dict[jax.Device, dict] = {}
        for device in self.local_devices:
            graph, state = self._sub_brain(brain, device)
            self._graphs[device], self._fixed[device] = graph, self._without_modules(state, self._names[device])

    @classmethod
    def balanced(cls, brain: Brain, devices: tp.Sequence[jax.Device], tolerance: float = 0.1) -> Partition:
        """
            Returns a partition of a brain over devices that balances their work and keeps what
            crosses devices small.

            Parameters
            ----------
            brain : Brain
                A built brain.
            devices : sequence of jax.Device
                The devices to divide the brain among, in order of preference. A device gets no part
                if the brain has fewer modules to place than devices.
            tolerance : float, default 0.1
                How much more work than the lightest device a device may take, as a fraction, to hold
                a module next to those it exchanges outputs with.

            Returns
            -------
            Partition

            Raises
            ------
            RuntimeError
                If the brain is not built.
            ValueError
                If no device is given.

            Notes
            -----
            The work of a module is the number of values in its state, which its weights, traces and
            buffers make up. Modules that read a property of one another, or write one, are placed
            together. The modules are placed from the heaviest, each on the device that keeps the
            work within ``tolerance`` and exchanges the most with it; then a module is moved to
            another device while the move lowers what crosses devices and keeps every device within
            ``1 + tolerance`` times the heaviest load, or an even share if that is more. What crosses
            is counted in values per step: the size of an output, once for every other device
            reading it.

            The parts depend on the brain and the devices only, so every process gets the same.
        """
        if not getattr(brain, '__built__', False):
            raise RuntimeError(
                'The brain is not built: call it once with example inputs before partitioning it.'
            )
        devices = list(devices)
        if not devices:
            raise ValueError('A brain is divided among one device or more, got none.')
        names = brain._modules_names
        # Modules that read a property of one another, or write one, go together: each set is placed whole.
        root = {name: name for name in names}
        def find(name: str) -> str:
            while root[name] != name:
                name = root[name]
            return name
        for spec in brain._modules_specs:
            linked = [port_map for maps in spec.inputs.values() for port_map in maps if port_map.is_property]
            linked += [port_map for maps in spec.effects.values() for port_map in maps]
            for port_map in linked:
                if port_map.origin not in (_CALL, _SELF):
                    root[find(port_map.origin)] = find(spec.name)
        sets = {}
        for name in names:
            sets.setdefault(find(name), []).append(name)
        set_of = {name: find(name) for name in names}
        mapping = split(brain)[1].raw_mapping
        work = {
            key: max(1, sum(np.size(leaf) for name in members for leaf in jax.tree.leaves(mapping[name])))
            for key, members in sets.items()
        }
        # The outputs read in another set: their size, and the sets reading them.
        readers: dict[tuple[str, str], set[str]] = {}
        for spec in brain._modules_specs:
            for maps in spec.inputs.values():
                for port_map in maps:
                    if port_map.origin in (_CALL, _SELF) or port_map.is_property or set_of[port_map.origin] == set_of[spec.name]:
                        continue
                    readers.setdefault((port_map.origin, port_map.port), set()).add(set_of[spec.name])
        size = {key: sum(np.size(leaf) for leaf in jax.tree.leaves(brain._cache[key])) for key in readers}
        def crossing(place: dict[str, int]) -> int:
            return sum(
                size[key] * len({place[reader] for reader in sets_reading} - {place[set_of[key[0]]]})
                for key, sets_reading in readers.items()
            )
        def shared(key: str, device: int, place: dict[str, int]) -> int:
            # What the set ``key`` exchanges with the sets already on ``device``.
            total = 0
            for output, sets_reading in readers.items():
                origin = set_of[output[0]]
                if origin == key:
                    total += size[output] * any(place.get(other) == device for other in sets_reading)
                elif key in sets_reading and place.get(origin) == device:
                    total += size[output]
            return total
        # The heaviest first, each on the device that exchanges the most with it among those it keeps balanced.
        place: dict[str, int] = {}
        load = [0] * len(devices)
        for key in sorted(sets, key=lambda key: -work[key]):
            lightest = min(load)
            fitting = [index for index in range(len(devices)) if load[index] + work[key] <= (lightest + work[key]) * (1 + tolerance)]
            index = max(fitting, key=lambda index: (shared(key, index, place), -load[index], -index))
            place[key] = index
            load[index] += work[key]
        # Then one set at a time, while a move lowers what crosses without unbalancing the devices.
        limit = (1 + tolerance) * max(max(load), sum(work.values()) / len(devices))
        current, moved = crossing(place), True
        while moved:
            moved = False
            for key in sets:
                for index in range(len(devices)):
                    if index == place[key] or load[index] + work[key] > limit:
                        continue
                    trial = {**place, key: index}
                    if crossing(trial) < current:
                        load[place[key]] -= work[key]
                        load[index] += work[key]
                        place, current, moved = trial, crossing(trial), True
                        break
        # The parts take the devices in the order given.
        order = {index: rank for rank, index in enumerate(sorted(set(place.values())))}
        parts = {devices[rank]: [name for name in names if order[place[set_of[name]]] == rank] for rank in order.values()}
        return cls(brain, parts)

    @property
    def devices(self) -> tuple[jax.Device, ...]:
        """
            The devices of the partition, in the order the parts were given, in every process.
        """
        return tuple(self._names)

    @property
    def local_devices(self) -> tuple[jax.Device, ...]:
        """
            The devices of the partition that belong to this process, in the order the parts were
            given: every device, with one process.
        """
        return tuple(device for device in self._names if self._is_local(device))

    @property
    def parts(self) -> dict[jax.Device, tuple[str, ...]]:
        """
            The names of the modules each device runs, in the order of the brain.
        """
        return dict(self._names)

    @property
    def configs(self) -> dict[jax.Device, BrainConfig]:
        """
            The configuration of the sub-brain of each device.
        """
        return dict(self._configs)

    def device_of(self, name: str) -> jax.Device:
        """
            Returns the device that runs the module ``name``.
        """
        return self._device_of[name]

    def split(self, brain: Brain) -> tuple[dict[jax.Device, nnx.GraphDef], dict[jax.Device, nnx.State]]:
        """
            Returns the graph and the state of the sub-brain of every device of this process, the
            state placed on its device.

            Parameters
            ----------
            brain : Brain
                The brain the partition was made from, or one of the same structure, such as one
                returned by `merge`. Its modules give the state of the sub-brains.

            Returns
            -------
            graphs : dict of jax.Device to GraphDef
                The graph of each sub-brain.
            states : dict of jax.Device to State
                The state of each sub-brain, on its device.
        """
        _, state = split(brain)
        states = {}
        for device in self.local_devices:
            states[device] = jax.device_put(nnx.State(self._with_modules(self._fixed[device], state, self._names[device])), device)
        return dict(self._graphs), states

    def merge(self, states: dict[jax.Device, nnx.State], device: jax.Device | None = None) -> Brain:
        """
            Returns the brain whose modules have the states of the sub-brains.

            With several processes, every process calls it, and every process gets the brain: the
            states of the sub-brains of the other processes are gathered.

            Parameters
            ----------
            states : dict of jax.Device to State
                The state of the sub-brain of each device of this process.
            device : jax.Device, optional
                The device the brain is placed on. The first device of the partition in this process
                when omitted.

            Returns
            -------
            Brain
                The brain, as a brain that ran unpartitioned would be.
        """
        state = self._fixed_brain
        for part_device in self.local_devices:
            state = self._with_modules(state, states[part_device], self._names[part_device])
        if self._spans_processes:
            local = {
                name: self._module_of(states[part_device], name)
                for part_device in self.local_devices for name in self._names[part_device]
            }
            owner = {name: part_device.process_index for name, part_device in self._device_of.items()}
            gathered = self._gather(local, owner, self._module_templates)
            for name, module in gathered.items():
                if name not in local:
                    cache = {} if module['cache'] is None else {name: module['cache']}
                    state = self._with_modules(state, {name: module['state'], _CACHE: cache}, (name,))
        if device is None:
            device = self.local_devices[0] if self.local_devices else jax.local_devices()[0]
        return merge(self._graph, jax.device_put(nnx.State(state), device))

    def run(
            self,
            function: tp.Callable,
            graphs: dict[jax.Device, nnx.GraphDef],
            states: dict[jax.Device, nnx.State],
            received: dict[jax.Device, dict[str, SparkPayload]],
            inputs: dict[str, SparkPayload],
            *args: tp.Any,
            **kwargs: tp.Any,
        ) -> tuple[dict[jax.Device, dict[str, SparkPayload]], dict[jax.Device, nnx.State]]:
        """
            Runs the sub-brain of every device of this process, and returns their outputs and states.

            Parameters
            ----------
            function : callable
                Runs a model: ``function(graph, state, *args, **inputs, **kwargs)`` returns its outputs and
                its state, as the loop running the brain does. A `spark.jit` function is recorded.
            graphs : dict of jax.Device to GraphDef
                The graph of each sub-brain, as `split` gives them.
            states : dict of jax.Device to State
                The state of each sub-brain.
            received : dict of jax.Device to dict of str to SparkPayload
                What each sub-brain reads from the others, as `initial` or `exchange` gives it.
            inputs : dict of str to SparkPayload
                The inputs of the brain, by name.
            *args, **kwargs
                Passed to ``function`` after the graph and the state, such as the steps of the call.

            Returns
            -------
            outputs : dict of jax.Device to dict of str to SparkPayload
                The outputs ``function`` returned for each device.
            states : dict of jax.Device to State
                The state of each sub-brain after the call.

            Raises
            ------
            ValueError
                With an open recorder, when a probe addresses the brain rather than a module or an
                input of the brain.
            NotImplementedError
                With an open recorder, when the partition spans several processes.

            Notes
            -----
            While a recorder of `spark.recording` is open, the calls of a `spark.jit` function on the
            devices are one call of the recorder, as a call running the brain is. Each records the
            probes of the modules of its sub-brain, and an input of the brain is recorded by the first
            device reading it. The records are joined on the first device and handed over once. The
            calls on the devices run the same steps.
        """
        parts = {
            device: ((graphs[device], states[device], *args), {**self.inputs(device, inputs), **received[device], **kwargs})
            for device in self.local_devices
        }
        if OPEN_RECORDERS and isinstance(function, Jit):
            if self._spans_processes:
                raise NotImplementedError(
                    'A partition across several processes is not recorded yet: the records of the parts of every process '
                    'would have to be gathered on process 0.'
                )
            results = recording_hooks().parts(function, parts, self._owner, self.local_devices[0])
        else:
            results = {device: function(*part_args, **part_kwargs) for device, (part_args, part_kwargs) in parts.items()}
        return {device: result[0] for device, result in results.items()}, {device: result[1] for device, result in results.items()}

    def inputs(self, device: jax.Device, inputs: dict[str, SparkPayload]) -> dict[str, SparkPayload]:
        """
            Returns the inputs of the brain that the sub-brain of ``device`` reads, placed on that device.

            Parameters
            ----------
            device : jax.Device
                A device of the partition in this process.
            inputs : dict of str to SparkPayload
                The inputs of the brain, by name.

            Returns
            -------
            dict of str to SparkPayload
        """
        missing = [name for name in self._inputs[device] if name not in inputs]
        if missing:
            raise KeyError(f'Missing inputs of the brain: {missing}.')
        return {name: jax.device_put(inputs[name], device) for name in self._inputs[device]}

    def initial(self, states: dict[jax.Device, nnx.State]) -> dict[jax.Device, dict[str, SparkPayload]]:
        """
            Returns what each sub-brain reads from the others on its next step, from their states.

            The cache of a sub-brain holds the outputs of its last step, so this is also what
            `exchange` returns after that step.

            Parameters
            ----------
            states : dict of jax.Device to State
                The state of the sub-brain of each device of this process.

            Returns
            -------
            dict of jax.Device to dict of str to SparkPayload
                For each device of this process, the outputs of the other parts its sub-brain reads,
                on that device.
        """
        def read(origin: str, port: str) -> SparkPayload:
            leaves = jax.tree.leaves(states[self._device_of[origin]][_CACHE][origin][port])
            return jax.tree.unflatten(self._trees[origin, port], leaves)
        return self._send(read)

    def exchange(self, outputs: dict[jax.Device, dict[str, SparkPayload]]) -> dict[jax.Device, dict[str, SparkPayload]]:
        """
            Copies the outputs that cross devices to the devices reading them.

            Parameters
            ----------
            outputs : dict of jax.Device to dict of str to SparkPayload
                The outputs of the step of the sub-brain of each device of this process.

            Returns
            -------
            dict of jax.Device to dict of str to SparkPayload
                For each device of this process, the outputs of the other parts its sub-brain reads,
                on that device. The copies between devices of one process are asynchronous; those
                between processes wait for every process to have run its step.
        """
        return self._send(lambda origin, port: outputs[self._device_of[origin]][crossing_name(origin, port)])

    def outputs(self, outputs: dict[jax.Device, dict[str, SparkPayload]]) -> dict[str, SparkPayload]:
        """
            Returns the outputs of the brain, from the outputs of the sub-brains.

            Parameters
            ----------
            outputs : dict of jax.Device to dict of str to SparkPayload
                The outputs of the sub-brain of each device of this process.

            Returns
            -------
            dict of str to SparkPayload
                One entry per output port of the brain produced in this process, on the device
                producing it.
        """
        return {name: outputs[device][name] for name, device in self._outputs.items() if self._is_local(device)}

    def _parts(
            self,
            brain: Brain,
            parts: dict[jax.Device, tp.Iterable[str]],
        ) -> tuple[dict[jax.Device, tuple[str, ...]], dict[str, jax.Device]]:
        """
            Validates the parts, and returns the names of each part in the order of the brain and the device of every module.
        """
        names = brain._modules_names
        device_of: dict[str, jax.Device] = {}
        for device, part in parts.items():
            part = (part,) if isinstance(part, str) else tuple(part)
            if not part:
                raise ValueError(f'The part of device {device} is empty.')
            for name in part:
                if name not in names:
                    raise ValueError(f'"{name}" is not a module of the brain, whose modules are {list(names)}.')
                if name in device_of:
                    raise ValueError(f'"{name}" is in the parts of devices {device_of[name]} and {device}.')
                device_of[name] = device
        missing = [name for name in names if name not in device_of]
        if missing:
            raise ValueError(f'Every module of the brain runs on a device: {missing} are in no part.')
        ordered = {device: tuple(name for name in names if device_of[name] == device) for device in parts}
        return ordered, device_of

    def _crosses(self, port_map: PortMap, device: jax.Device) -> bool:
        """
            Returns whether ``port_map`` reads a module of another part than the one of ``device``.
        """
        return port_map.origin not in (_CALL, _SELF) and self._device_of[port_map.origin] != device

    def _crossings_of(self, brain: Brain) -> dict[tuple[str, str], tuple[jax.Device, ...]]:
        """
            Returns the outputs read across devices, with the devices reading each.
        """
        crossings: dict[tuple[str, str], list[jax.Device]] = {}
        for spec in brain._modules_specs:
            device = self._device_of[spec.name]
            for maps in spec.inputs.values():
                for port_map in maps:
                    if not self._crosses(port_map, device):
                        continue
                    if port_map.is_property:
                        raise ValueError(
                            f'Module "{spec.name}" reads the property "{port_map.port}" of module "{port_map.origin}", '
                            f'which is on another device. Properties are read within a step: put both modules in one part.'
                        )
                    readers = crossings.setdefault((port_map.origin, port_map.port), [])
                    if device not in readers:
                        readers.append(device)
            for property_name, maps in spec.effects.items():
                for port_map in maps:
                    if self._crosses(port_map, device):
                        raise ValueError(
                            f'The property "{property_name}" of module "{spec.name}" is written by module '
                            f'"{port_map.origin}", which is on another device. Effects are applied within a step: put both '
                            f'modules in one part.'
                        )
        return {key: tuple(readers) for key, readers in crossings.items()}

    def _sub_config(self, brain: Brain, device: jax.Device) -> BrainConfig:
        """
            Returns the configuration of the sub-brain of ``device``.
        """
        specs = []
        for spec in brain._modules_specs:
            if self._device_of[spec.name] != device:
                continue
            inputs = {
                port_name: [
                    PortMap(_CALL, crossing_name(port_map.origin, port_map.port)) if self._crosses(port_map, device) else port_map
                    for port_map in maps
                ]
                for port_name, maps in spec.inputs.items()
            }
            outputs = dict(spec.outputs)
            for origin, port in self._crossings:
                if origin == spec.name:
                    outputs[crossing_name(origin, port)] = port
            specs.append(ModuleSpecs(
                name=spec.name, module_cls=spec.module_cls, inputs=inputs, config=copy.deepcopy(spec.config),
                outputs=outputs, effects=copy.deepcopy(spec.effects),
            ))
        return brain.config.merge(modules_specs=tuple(specs))

    def _sub_brain(self, brain: Brain, device: jax.Device) -> tuple[nnx.GraphDef, nnx.State]:
        """
            Builds the sub-brain of ``device``, holding the modules of the brain, and returns its graph and its state.
        """
        sub = type(brain)(config=self._configs[device])
        brain_inputs = brain.get_controller_inputs()
        examples = {}
        for name in sub.get_controller_inputs():
            if name in brain_inputs:
                examples[name] = brain._cache[_CALL, name]
            else:
                origin, port = name.split(':', 1)
                examples[name] = brain._cache[origin, port]
        sub(**examples)
        # The sub-brain takes the modules of the brain, and with them the values they hold in the graph.
        for name in self._names[device]:
            setattr(sub, name, getattr(brain, name))
        return split(sub)

    def _is_local(self, device: jax.Device) -> bool:
        """
            Returns whether ``device`` belongs to this process.
        """
        return device.process_index == jax.process_index()

    def _owner(self, path: tuple[str, ...], name: str) -> jax.Device:
        """
            Returns the device recording the probe of ``path`` and ``name``: that of its module, or the first
            reading the input of the brain it addresses.
        """
        if path[:1] == (_CALL,):
            for device in self.devices:
                if name in self._inputs[device]:
                    return device
        elif path and path[0] in self._device_of:
            return self._device_of[path[0]]
        raise ValueError(
            f'A partition records the modules of the brain and its inputs, and "{".".join((*path, name))}" addresses '
            f'neither.'
        )

    @staticmethod
    def _module_of(state: nnx.State | dict, name: str) -> dict[str, tp.Any]:
        """
            Returns the state of the module ``name`` within a brain state, with its outputs in the cache, or None for
            a module with none there.
        """
        mapping = state.raw_mapping if isinstance(state, nnx.State) else state
        return {'state': mapping[name], 'cache': mapping[_CACHE].get(name, None)}

    @staticmethod
    def _template(tree: tp.Any) -> tuple[tp.Any, list[jax.ShapeDtypeStruct]]:
        """
            Returns the structure of ``tree`` and the shape and dtype of its leaves.
        """
        leaves, treedef = jax.tree.flatten(tree)
        return treedef, [jax.ShapeDtypeStruct(np.shape(leaf), leaf.dtype) for leaf in leaves]

    @staticmethod
    def _gather(
            local: dict[str, tp.Any],
            owner: dict[str, int],
            templates: dict[str, tuple[tp.Any, list[jax.ShapeDtypeStruct]]],
        ) -> dict[str, tp.Any]:
        """
            Returns every tree of ``templates`` as the process ``owner[key]`` holds it, given in ``local`` the trees
            this process holds. Every process calls it, with the same templates.
        """
        this = jax.process_index()
        sent = {}
        for key, (_, shapes) in templates.items():
            leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(local[key])] if owner[key] == this else \
                [np.zeros(shape.shape, shape.dtype) for shape in shapes]
            # NOTE: The collectives do not carry booleans: they travel as bytes.
            sent[key] = [leaf.view(np.uint8) if leaf.dtype == np.bool_ else leaf for leaf in leaves]
        gathered = multihost_utils.process_allgather(sent)
        result = {}
        for key, (treedef, shapes) in templates.items():
            leaves = [leaf[owner[key]] for leaf in gathered[key]]
            leaves = [leaf.view(np.bool_) if shape.dtype == np.bool_ else leaf for leaf, shape in zip(leaves, shapes)]
            result[key] = jax.tree.unflatten(treedef, leaves)
        return result

    @staticmethod
    def _without_modules(state: nnx.State, names: tp.Iterable[str]) -> dict:
        """
            Returns the mapping of a brain state without the state of the modules ``names`` and their outputs in
            the cache.
        """
        names = set(names)
        mapping = state.raw_mapping
        fixed = {key: value for key, value in mapping.items() if key not in names}
        fixed[_CACHE] = {key: value for key, value in mapping[_CACHE].items() if key not in names}
        return fixed

    @staticmethod
    def _with_modules(fixed: dict, state: nnx.State | dict, names: tp.Iterable[str]) -> dict:
        """
            Returns ``fixed`` with the state of the modules ``names``, and their outputs in the cache, taken from
            ``state``.
        """
        # The nested levels of a state are plain dictionaries, as `split` gives them.
        mapping = state.raw_mapping if isinstance(state, nnx.State) else state
        result = dict(fixed)
        cache = dict(fixed[_CACHE])
        for name in names:
            result[name] = mapping[name]
            if name in mapping[_CACHE]:
                cache[name] = mapping[_CACHE][name]
        result[_CACHE] = cache
        return result

    def _send(self, read: tp.Callable[[str, str], SparkPayload]) -> dict[jax.Device, dict[str, SparkPayload]]:
        """
            Places every output that crosses devices on the devices of this process reading it. ``read`` gives an
            output produced in this process; one produced in another is gathered from it.
        """
        produced = {key: read(*key) for key in self._crossings if self._is_local(self._device_of[key[0]])}
        gathered = {}
        if self._remote:
            gathered = self._gather(
                {crossing_name(*key): produced[key] for key in self._remote if key in produced},
                {crossing_name(*key): self._device_of[key[0]].process_index for key in self._remote},
                self._crossing_templates,
            )
        received: dict[jax.Device, dict[str, SparkPayload]] = {device: {} for device in self.local_devices}
        for (origin, port), readers in self._crossings.items():
            name = crossing_name(origin, port)
            for device in readers:
                if self._is_local(device):
                    payload = produced[origin, port] if (origin, port) in produced else gathered[name]
                    received[device][name] = jax.device_put(payload, device)
        return received

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
