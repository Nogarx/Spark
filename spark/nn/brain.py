#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
	
import jax.numpy as jnp
import typing as tp
import dataclasses as dc
from math import prod
import spark.core.utils as utils
from spark.core.cache import Cache
from spark.core.module import SparkModule, SparkMeta
from spark.core.specs import PortSpecs, PortMap, ModuleSpecs
from spark.core.variables import Variable
from spark.core.payloads import SparkPayload, FloatArray, SpikeArray, ValueSparkPayload
from spark.core.registry import register_config, REGISTRY
from spark.core.config import BaseSparkConfig
import jax

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class BrainMeta(SparkMeta):
	"""
		Brain metaclass.
	"""
	pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class BrainConfig(BaseSparkConfig):
	"""
		Configuration class for Brain's.
	"""
	input_map: dict[str, PortSpecs] = dc.field(
		metadata = {
			'description': 'Input map configuration.',
		})
	output_map: dict[str, dict] = dc.field(
		metadata = {
			'description': 'Output map configuration.',
		})
	modules_map: dict[str, ModuleSpecs] = dc.field(
		metadata = {
			'description': 'Modules map configuration.',
		})



	def _validate_maps(self,):
		"""
			Basic validation of the configuration maps to ensure that brain can properly read the data.
		"""
		# Input map validation
		if not isinstance(self.input_map, dict):
			raise TypeError(
				f'\"input_map\" must be a dictionary, but got \"{type(self.input_map).__name__}\".'
			)
		for key in self.input_map.keys():
			if not isinstance(key, str):
				raise TypeError(
					f'All keys in \"input_map\" must be strings, but found key \"{key}\" of type {type(key).__name__}.'
				)
			if not isinstance(self.input_map[key], PortSpecs):
				raise TypeError(
					f'All values in \"input_map\" must be PortSpecs, but found value \"{self.input_map[key]}\" '
					f'of type {type(self.input_map[key]).__name__}.'
				)
		# Output map validation
		if not isinstance(self.output_map, dict):
			raise TypeError(
				f'\"output_map\" must be a dictionary, but got \"{type(self.output_map).__name__}\".'
			)
		for output_name, output_details in self.output_map.items():
			if not isinstance(output_name, str):
				raise TypeError(
					f'All keys in \"output_map\" must be strings, but found key \"{output_name}\" of type {type(output_name).__name__}.'
				)
			if not isinstance(output_details, dict):
				raise TypeError(
					f'All values in \"output_map\" must be dict, but found value \"{output_details}\" '
					f'of type {type(output_details).__name__}.'
				)		
			_input = output_details.get('input', None)
			if not isinstance(_input, PortMap):
				raise TypeError(
					f'Expected \"output_map[\"{output_name}\"][\"input\"]\" to be of type PortMap, but got \"{_input}\".'
				)
			spec = output_details.get('spec', None)
			if not isinstance(spec, PortSpecs):
				raise TypeError(
					f'Expected \"output_map[\"{output_name}\"][\"spec\"]\" to be of type PortSpecs, but got \"{spec}\".'
				)
		# Modules map validation
		if not isinstance(self.modules_map, dict):
			raise TypeError(
				f'\"modules_map\" must be a dictionary, but got \"{type(self.modules_map).__name__}\".'
			)
		for key in self.modules_map.keys():
			if not isinstance(key, str):
				raise TypeError(
					f'All keys in \"modules_map\" must be strings, but found key \"{key}\" of type \"{type(key).__name__}\".'
				)
			if not isinstance(self.modules_map[key], ModuleSpecs):
				raise TypeError(
					f'All values in \"modules_map\" must be type \"ModuleSpecs\", but found value \"{self.modules_map[key]}\" '
					f'of type \"{type(self.modules_map[key]).__name__}\".'
				)
		# Validate sources.
		valid_sources = set(self.input_map.keys()) | set(self.modules_map.keys())
		# Does the origin exist in the set of valid sources?
		for module_name, module_specs in self.modules_map.items():
			for port_name, connections in module_specs.inputs.items():
				for port_map in connections:
					if port_map.origin not in valid_sources and port_map.origin != '__call__':
						raise ValueError(
							f'In module \"{module_name}\", connection for port \"{port_name}\" refers to an '
							f'unknown origin: \"{port_map.origin}\". Valid sources are: {valid_sources}.'
						)



	def validate(self,) -> None:
		# Brain specific validation.
		self._validate_maps()
		# Standard config validation.
		super().validate()



	def _parse_tree_structure(self, current_depth: int, simplified: bool = False, header: str | None= None) -> str:
		"""
			Parses the tree to produce a string with the appropiate format for the ascii_tree method.
		"""
		level_header = f'{header}: ' if header else ''
		rep = current_depth * ' ' + f'{level_header}{self.__class__.__name__}\n'

		# Expand inputs specs
		rep += (current_depth + 1) * ' ' + f'Input Map:\n'
		for spec_name, spec in self.input_map.items():
			rep += (current_depth + 2) * ' ' + f'{spec_name} <- {spec}\n'
		# Expand outputs specs
		rep += (current_depth + 1) * ' ' + f'Output Map:\n'
		for spec_name, spec in self.output_map.items():
			rep += (current_depth + 2) * ' ' + f'{spec_name} <- {spec['input']} | {spec['spec']}\n'
		# Expand module specs
		rep += (current_depth + 1) * ' ' + f'Modules Map:\n'
		for spec_name, module_spec in self.modules_map.items():
			if not simplified:
				rep += (current_depth + 2) * ' ' + f'{spec_name}: {module_spec.module_cls.__name__}\n'
				rep += (current_depth + 3) * ' ' + f'Inputs:\n'
				for input_name, port_spec_list in module_spec.inputs.items():
					rep += (current_depth + 4) * ' ' + f'{input_name}:\n'
					for port in port_spec_list:
						rep += (current_depth + 5) * ' ' + f'{port}\n'
				rep += module_spec.config._parse_tree_structure(current_depth+3, simplified=simplified)
			else:
				rep += module_spec.config._parse_tree_structure(current_depth+2, simplified=simplified)
		return rep



	def refresh_seeds(self):
		"""
			Utility method to recompute all seed variables within the SparkConfig.
			Useful when creating several populations from the same config.
		"""
		for spec in self.modules_map.values():
			spec.config = spec.config.with_new_seeds()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Brain(SparkModule, metaclass=BrainMeta):
	"""
		Brain model.

		A brain is a pipeline object used to represent and coordinate a collection of neurons and interfaces.
		This implementation relies on a cache system to simplify parallel computations; every timestep all the modules
		in the Brain read from the cache, update its internal state and update the cache state. 
		Note that this introduces a small latency between elements of the brain, which for most cases is negligible, and for
		such a reason it is recommended that only full neuron models and interfaces are used within a Brain.
	"""
	config: BrainConfig

	# Typing annotations.
	_modules_list: list[str]
	_cache: dict[str, dict[str, Cache]]
	_modules_input_map: dict[str, dict[str, list[PortMap]]]
	_modules_input_specs: dict[str, dict[str, PortSpecs]]
	_modules_output_specs: dict[str, dict[str, PortSpecs]]

	def __init__(self, config: BrainConfig = None, **kwargs):
		# Initialize super.
		super().__init__(config=config, **kwargs)
		# Build modules
		self._build_modules()



	def _build_modules(self,):
		"""
			Construct the modules defined in the modules_map of the configuration class.
		"""
		# Construc all modules.
		for module_name, module_specs in self.config.modules_map.items():
			setattr(self, module_name, REGISTRY.MODULES.get(module_specs.module_cls.__name__).class_ref(config=module_specs.config))
		self._modules_list = list(self.config.modules_map.keys())
		# Information flow map.
		self._modules_input_map = {}
		for module_name, module_specs in self.config.modules_map.items():
			self._modules_input_map[module_name] = module_specs.inputs



	# TODO: I think most of this validation can be moved to BrainConfig
	def _validate_connections(self,):
		"""
			Prevalidates that all modules are reachable.
		"""
		# Collect all modules input/output specs.
		self._modules_input_specs = {}
		self._modules_output_specs = {}
		for module_name in self._modules_list:
			module: SparkModule = getattr(self, module_name)
			self._modules_input_specs[module_name] = module.get_input_specs()
			self._modules_output_specs[module_name] = module.get_output_specs()

		# Validate that each module can be connected.
		for module_name in self._modules_list:
			
			# Validate that every non-optional input port exists
			module_inputs: dict[str, list[PortMap]] = self.config.modules_map[module_name].inputs
			required_ports = set(self._modules_input_specs[module_name].keys()).difference(module_inputs.keys())
			for port_name in required_ports:
				class_error = REGISTRY.MODULES.get(self.config.modules_map[module_name].module_cls).class_ref
				raise ValueError(
					f'Port \"{port_name}\" for module "{module_name}\" is not defined and '
					f'is a mandatory input of \"{class_error.__name__}\".'
				)
			
			# Validate that no extra input port were defined
			extra_ports = set(module_inputs.keys()).difference(set(self._modules_input_specs[module_name].keys()))
			if len(extra_ports) > 0:
				raise ValueError(
					f'Port \"{extra_ports[0]}\" for module \"{module_name}\" is not part of the specifications '
					f'of \"{self.config.modules_map[module_name].module_cls}\".'
				)
			
			# Validate the integrity of the ports 
			for port_name, ports_list in module_inputs.items():
				
				# Check input module defines the output port
				for port_map in ports_list:
					if port_map.origin == '__call__':
						if not port_map.port in self.config.input_map:
							raise ValueError(
								f'Input port \"{port_name}\" for module \"{module_name}\" request port \"{port_map.port}\" '
								f'from \"{port_map.origin}\", but inputs_map does not define such port.'
							)
					elif not port_map.port in self._modules_output_specs[port_map.origin]:
						raise ValueError(
							f'Input port \"{port_name}\" for module \"{module_name}\" request port \"{port_map.port}\" from '
							f'\"{port_map.origin}\", but \"{self.config.modules_map[port_map.origin].module_cls}\" does not define such port.'
						)
				
				# Validate ports can be safely merged.
				if len(ports_list) > 1:
					payload_type = self._modules_output_specs[ports_list[0].origin][ports_list[0].port].payload_type
					for i in range(1,len(ports_list)):
						target_payload_type = self._modules_output_specs[ports_list[i].origin][ports_list[i].port].payload_type
						if not payload_type == target_payload_type:
							raise TypeError(
								f'Input port \"{port_name}\" for module \"{module_name}\" request port \"{ports_list[i].port}\" '
								f'from \"{ports_list[i].origin}\", but payload type is not compatible, expected \"{payload_type}\" '
								f'and got \"{target_payload_type}\".'
							) 
				
				# Validate shapes
				shape = 0
				if len(ports_list) > 1:
					# Many-to-one input-output. Inputs need to be merged, default merged behaviour is to flat everything.
					for port_map in ports_list:
						if port_map.origin == '__call__':
							shape += prod(self.config.input_map[port_map.port].shape)
						else:
							output_specs: PortSpecs = getattr(self, port_map.origin).get_output_specs()[port_map.port]
							shape += prod(output_specs.shape)
				else:
					# One-to-one input-output.
					if port_map.origin == '__call__':
						shape = self.config.input_map[port_map.port].shape
					else:
						output_specs: PortSpecs = getattr(self, port_map.origin).get_output_specs()[port_map.port]
						shape = output_specs.shape
				# Normalize and compare shapes.
				shape = utils.validate_shape(shape)
				expected_input_shape = self._modules_input_specs[module_name][port_name].shape
				if expected_input_shape != shape:
					raise ValueError(
						f'Input port \"{port_name}\" for module \"{module_name}\" expected input shape '
						f'{expected_input_shape} but got shape \"{shape}\".'
					) 



	def resolve_initialization_order(self,):
		"""
			Resolves the initialization order of the modules.
		"""
		import networkx as nx
		# Create directed graph
		graph = {}
		for key, values in self.config.modules_map.items():
			dependencies = set()
			for input_port in values.inputs.values():
				for port in input_port:
					dependencies.add(port.origin)
			graph[key] = dependencies
		nx_graph = nx.DiGraph(graph)
		# Get strongly connected components
		scc = list(nx.strongly_connected_components(nx_graph))
		# Condensate graph
		condensation = nx.condensation(nx_graph, scc=scc)
		order = [scc[i] for i in list(nx.topological_sort(condensation))[::-1]]
		return order



	# TODO: Current approach to build the cache is rather silly, there should be a better more robust way to construct it.
	def build(self, input_specs: dict[str, PortSpecs]):
		# Get build order.
		order = self.resolve_initialization_order()
		# Build cache. Cache needs to be computed a step at a time. 
		modules_output_specs = {'__call__': {name: spec for name, spec in input_specs.items()}}
		for group in order:
			for module_name in group:
				# Skip __call__
				if module_name == '__call__':
					continue
				# Collect module specs and construct a mock input
				mock_input = {}
				validate_async = True
				for port_name, port_map_list in self.config.modules_map[module_name].inputs.items():
					portspecs_list = []
					for port_map in port_map_list:
						if port_map.origin in modules_output_specs:
							# Modules was already built grab, get the spec.
							portspecs_list.append(
								modules_output_specs[port_map.origin][port_map.port]
							)
						elif port_map.origin in group:
							# Module is not built yet, it must define a recurrent spec to be part of a cyclic dependency.
							origin_module: SparkModule = getattr(self, port_map.origin)
							portspecs_list.append(
								origin_module.get_contract_output_specs()[port_map.port]
							)
							# Turn off async validation.
							
							
							validate_async = False
						else:
							# Something weird happend. System is trying to get something from a module that should have been called later.
							raise RuntimeError(
								f'Trying to get port from module "{port_map.origin}" for "{module_name}"... '
								f'418 I\'m a teapot.'
							)
					
					mock_port_spec = PortSpecs.from_portspecs_list(portspecs_list, validate_async=validate_async)
					mock_input[port_name] = mock_port_spec._create_mock_input()
				# Initialize module
				module: SparkModule = getattr(self, module_name)
				abc_output = module(**mock_input)
				# Add output specs to list
				modules_output_specs[module_name] = module.get_output_specs()
		# Build cache.
		self._cache = {
			module_name : {
				port: Cache._from_spec(spec) for port, spec in module_output_specs.items()
			} for module_name, module_output_specs in modules_output_specs.items()
		}
		# Set built flag
		self.__built__ = True
		# Get a simplified version of the output map for __call__ iteration.
		self._flat_output_map = []
		for output_name, output_details in self.config.output_map.items():
			self._flat_output_map.append((output_name, output_details['input'].origin, output_details['input'].port))
		# Validate modules connections.
		self._validate_connections()



	# NOTE: We need to override _construct_input_specs since sig_parser.get_input_specs will lead to an incorrect 
	# signature because Brain can have an arbitrary number of inputs all under the key "inputs".
	def _construct_input_specs(self, abc_args: dict[str, dict[str, SparkPayload]]) -> dict[str, PortSpecs]:
		# Extract the real inputs from abc_args, they are all under the key 'inputs'
		abc_args = abc_args['inputs']
		# Validate specs and abc_args match.
		expected_input = self.config.input_map
		if expected_input.keys() != abc_args.keys():
			# Check if missing key is optional.
			set_diff = set(self.config.input_map.keys()).difference(abc_args.keys())
			for key in set_diff:
				raise ValueError(
					f'Module \"{self.name}\" expects variable \"{key}\" but it was not provided.'
				)
			# Check if extra keys are provided.
			set_diff = set(abc_args.keys()).difference(expected_input.keys())
			for key in set_diff:
				raise ValueError(
					f'Module \"{self.name}\" received an extra variable \"{key}\" but it is not part of the specification.'
				)
		# Finish specs, use abc_args to skip optional missing keys.
		input_specs = {}
		for key, payload in abc_args.items():
			# Sanity check
			if not isinstance(payload, SparkPayload):
				raise TypeError(
					f'Expected non-optional input payload \"{key}\" of module \"{self.name}\" to be '
					f'of type \"{SparkPayload.__name__}\" but got type {type(payload)}'
				)
			# PortSpecs are immutable, we need to create a new one.
			input_specs[key] = PortSpecs(
				payload_type=payload.__class__,
				shape=payload.shape,
				dtype=payload.dtype,
				description=f'Auto-generated input spec for input \"{key}\" of module \"{self.name}\".',
			)
		return input_specs



	# NOTE: We need to override _construct_output_specs since sig_parser.get_output_specs will lead to an incorrect 
	# signature because Brain can have an arbitrary number of outputs all under the key "outputs".
	def _construct_output_specs(self, abc_args: dict[str, SparkPayload]) -> dict[str, PortSpecs]:
		# Output is constructed dynamically from the output map, there is no ground truth.
		expected_output = self.config.output_map
		output_specs = {}
		for output_name in expected_output.keys():
			port_map: PortMap = expected_output[output_name]['input']
			port_spec: PortSpecs = expected_output[output_name]['spec']
			output_specs[output_name] = PortSpecs(
				payload_type=port_spec.payload_type,
				shape=abc_args[output_name].shape,
				dtype=abc_args[output_name].dtype,
				description=f'Auto-generated output spec for output \"{output_name}\".',
			)
		return output_specs



	def _build_neuron_list(self):
		"""
			Inspect the object to collect the names of all child of type Neuron.
		"""
		self._neuron_names = []
		# Get all attribute names
		all_attr_names = []
		if hasattr(self, '__dict__'):
			all_attr_names = list(vars(self).keys())
		# Add attributes from __slots__ if they exist
		if hasattr(self, '__slots__'):
			all_attr_names.extend(self.__slots__)
		# Check the attribute's type
		for name in set(all_attr_names):
			try:
				from spark.nn.neurons import Neuron
				if isinstance(getattr(self, name), Neuron):
					self._neuron_names.append(name)
			except AttributeError:
				continue



	def reset(self):
		"""
			Resets all the modules to its initial state.
		"""
		# Build components list. TODO: This should be called post init. 
		if self._neuron_names is not None:
			self._build_neuron_list()
		# Reset components.
		for name in self._neuron_names:
			getattr(self, name).reset()


	
	def _concatenate_payloads(self, args: list[SparkPayload]) -> SparkPayload:
		payload_type = type(args[0])
		if issubclass(payload_type, SpikeArray):
			return payload_type._from_encoding(jnp.concatenate([x._encoding.reshape(-1) for x in args]))
		else:
			return payload_type(jnp.concatenate([x.value.reshape(-1) for x in args]))



	def __call__(self, **inputs: SparkPayload) -> dict[str, SparkPayload]:
		"""
			Update brain's states.
		"""
		# Put input into the cache.
		for input_name in self.config.input_map.keys():
			self._cache['__call__'][input_name].set(inputs[input_name])
		# Update modules
		outputs = {}
		for module_name in self._modules_list:
			# Reconstruct module input using the cache cache.
			input_args = {}
			for port_name, ports_list in self._modules_input_map[module_name].items():
				# TODO: This does not support unflatten inputs.
				input_args[port_name] = self._concatenate_payloads(
					[self._cache[port_map.origin][port_map.port].get() for port_map in ports_list])
			outputs[module_name] = getattr(self, module_name)(**input_args)
		# Update cache
		for module_name in self._modules_list:
			for port_name in self._modules_output_specs[module_name].keys():
				self._cache[module_name][port_name].set(outputs[module_name][port_name])
		# Gather output
		brain_output = {
			name: outputs[origin][port] for name, origin, port in self._flat_output_map 
		}
		return brain_output



	# TODO: This needs to be set differently, perhaps through some sort of mask. 
	def get_spikes_from_cache(self) -> dict:
		"""
			Collect the brain's spikes.
		"""
		brain_spikes = {}
		for module_name in self._modules_list:
			if 'spikes' in self._cache[module_name]:
				brain_spikes[module_name] = self._cache[module_name]['spikes'].get()
			elif 'out_spikes' in self._cache[module_name]:
				brain_spikes[module_name] = self._cache[module_name]['out_spikes'].get()
		return brain_spikes



	def _parse_tree_structure(self, current_depth: int = 0, name: str | None = None) -> str:
		"""
			Parses the tree with to produce a string with the appropiate format for the ascii_tree method.
		"""
		if name:
			rep = current_depth * ' ' + f'{name} ({self.__class__.__name__})\n'
		else:
			rep = current_depth * ' ' + f'{self.__class__.__name__}\n'
		for module_name in self._modules_list:
			module: SparkModule = getattr(self, module_name)
			rep += module._parse_tree_structure(current_depth+1, name=module_name)
		return rep

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################