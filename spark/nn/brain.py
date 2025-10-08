#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
	
import jax.numpy as jnp
import typing as tp
import dataclasses as dc
from math import prod
from spark.core.cache import Cache
from spark.core.module import SparkModule, SparkMeta
from spark.core.specs import PortSpecs, PortMap, OutputSpec, InputSpec, ModuleSpecs
from spark.core.variables import Variable
from spark.core.shape import Shape
from spark.core.payloads import SparkPayload, FloatArray
from spark.core.registry import register_config, REGISTRY
from spark.core.config import BaseSparkConfig

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
	input_map: dict[str, InputSpec] = dc.field(
		metadata = {
			'description': 'Input map configuration.',
		})
	output_map: dict[str, dict[str, OutputSpec]] = dc.field(
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
			if not isinstance(self.input_map[key], InputSpec):
				raise TypeError(
					f'All values in \"input_map\" must be InputSpec, but found value \"{self.input_map[key]}\" '
					f'of type {type(self.input_map[key]).__name__}.'
				)
		# Output map validation
		if not isinstance(self.output_map, dict):
			raise TypeError(
				f'\"output_map\" must be a dictionary, but got \"{type(self.output_map).__name__}\".'
			)
		for module_name, module_output_details in self.output_map.items():
			if not isinstance(module_name, str):
				raise TypeError(
					f'All keys in \"output_map\" must be strings, but found key \"{module_name}\" of type {type(module_name).__name__}.'
				)
			if not isinstance(module_output_details, dict):
				raise TypeError(
					f'All values in \"output_map\" must be dict, but found value \"{module_output_details}\" '
					f'of type {type(module_output_details).__name__}.'
				)
			for port_name, port_spec in module_output_details.items():
				if not isinstance(key, str):
					raise TypeError(
						f'All keys in \"output_map[\"module\"]\" must be strings, '
						f'but found key \"{port_name}\" of type \"{type(port_name).__name__}\".'
					)
				if not isinstance(port_spec, OutputSpec):
					raise TypeError(
						f'All values in \"output_map[\"module\"]\" must be OutputSpec, '
						f'but found value \"{port_spec}\" of type \"{type(port_spec).__name__}\".'
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
							f'unknown origin: "{port_map.origin}". Valid sources are: {valid_sources}.'
						)



	def validate(self,) -> None:
		# Brain specific validation.
		self._validate_maps()
		# Standard config validation.
		super().validate()


#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Brain(SparkModule, metaclass=BrainMeta):
	"""
		Abstract brain model.
		This is more a convenience class used to synchronize data more easily.
	"""
	config: BrainConfig

	# Typing annotations.
	_modules_list: list[str]
	_cache: dict[str, dict[str, Cache]]
	_modules_input_map: dict[str, dict[str, list[PortMap]]]
	_modules_input_specs: dict[str, dict[str, InputSpec]]
	_modules_output_specs: dict[str, dict[str, OutputSpec]]

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
				if not self._modules_input_specs[module_name][port_name].is_optional:
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
							output_specs: OutputSpec = getattr(self, port_map.origin).get_output_specs()[port_map.port]
							shape += prod(output_specs.shape)
				else:
					# One-to-one input-output.
					if port_map.origin == '__call__':
						shape = self.config.input_map[port_map.port].shape
					else:
						output_specs: OutputSpec = getattr(self, port_map.origin).get_output_specs()[port_map.port]
						shape = output_specs.shape
				# Normalize and compare shapes.
				shape = Shape(shape)
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
	def build(self, input_specs: dict[str, InputSpec]):
		# Get build order.
		order = self.resolve_initialization_order()
		# Build cache. Cache needs to be computed a step at a time. 
		self._cache = {
			'__call__': {
				name: Cache(
					Variable(jnp.zeros(spec.shape), dtype=spec.dtype), FloatArray
				) for name, spec in input_specs.items()
			}
		}
		# Build cache. Cache needs to be computed a step at a time. 
		for group in order:
			for module_name in group:
				# Skip __call__
				if module_name == '__call__':
					continue
				# Collect module initialization specs
				dummy_input: dict[str, SparkPayload] = {}
				for port_name, port_map_list in self.config.modules_map[module_name].inputs.items():
					shapes = []
					for port_map in port_map_list:
						if port_map.origin in self._cache:
							# If origin module is in cache it was already been built; use it shape.
							shapes.append(self._cache[port_map.origin][port_map.port].shape)
							dtype = self._cache[port_map.origin][port_map.port].dtype
							payload_type = self._cache[port_map.origin][port_map.port].payload_type
						elif port_map.origin in group:
							# If origin is in group it must predefine its output shape in order to be part of a cyclic dependency.
							origin_module: SparkModule = getattr(self, port_map.origin)
							recurrent_shape_contract = origin_module.get_recurrent_shape_contract()
							shapes.append(recurrent_shape_contract[port_map.port])
							dtype = origin_module._dtype
							module_output_specs = origin_module.__class__._get_output_specs()
							payload_type = module_output_specs[port_map.port].payload_type
						else:
							# Something weird happend. System is trying to get something from a module that should have been called later.
							raise RuntimeError(
								f'Trying to get port from module "{port_map.origin}" for "{module_name}"... '
								f'418 I\'m a teapot.'
							)
					# TODO: This is extremely brittle, it will fall apart as soon as we allow more interesting shape mixin.
					# Construct a dummy input
					dummy_value = jnp.concatenate([jnp.zeros(s).reshape(-1) for s in shapes]).astype(dtype)
					dummy_input[port_name] = payload_type(dummy_value)
				# Initialize module
				module: SparkModule = getattr(self, module_name)
				dummy_output = module(**dummy_input)
				# Update cache
				self._cache[module_name] = {
					name: Cache(
						Variable(jnp.zeros(payload.shape), dtype=payload.dtype), payload.__class__
					) for name, payload in dummy_output.items()
				}
		# Set built flag
		self.__built__ = True
		# Validate modules connections.
		self._validate_connections()



	# NOTE: We need to override _construct_input_specs since sig_parser.get_input_specs will lead to an incorrect 
	# signature because Brain can have an arbitrary number of inputs all under the key "inputs".
	def _construct_input_specs(self, abc_args: dict[str, dict[str, SparkPayload]]) -> dict[str, InputSpec]:
		# Extract the real inputs from abc_args, they are all under the key 'inputs'
		abc_args = abc_args['inputs']
		# Validate specs and abc_args match.
		expected_input = self.config.input_map
		if expected_input.keys() != abc_args.keys():
			# Check if missing key is optional.
			set_diff = set(self.config.input_map.keys()).difference(abc_args.keys())
			for key in set_diff:
				if not expected_input[key].is_optional:
					raise ValueError(
						f'Module \"{self.name}\" expects non-optional variable \"{key}\" but it was not provided.'
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
			# InputSpec are immutable, we need to create a new one.
			input_specs[key] = InputSpec(
				payload_type=payload.__class__,
				shape=payload.shape,
				dtype=payload.dtype,
				is_optional=False,
				description=f'Auto-generated input spec for input \"{key}\" of module \"{self.name}\".',
			)
		return input_specs



	# NOTE: We need to override _construct_output_specs since sig_parser.get_output_specs will lead to an incorrect 
	# signature because Brain can have an arbitrary number of outputs all under the key "outputs".
	def _construct_output_specs(self, abc_args: dict[str, SparkPayload]) -> dict[str, OutputSpec]:
		# Output is constructed dynamically from the output map, there is no ground truth.
		expected_output = self.config.output_map
		output_specs = {}
		for module in expected_output.keys():
			for port_name, port_spec in expected_output[module].items():
				output_specs[f'{module}.{port_name}'] = OutputSpec(
					payload_type=port_spec.payload_type,
					shape=abc_args[f'{module}.{port_name}'].shape,
					dtype=abc_args[f'{module}.{port_name}'].dtype,
					description=f'Auto-generated output spec for input \"{module}.{port_name}\" of module \"{self.name}\".',
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



	def _concatenate_payloads(self, args: list[SparkPayload]):
		payload_type = type(args[0])
		return payload_type(jnp.concatenate([x.value.reshape(-1) for x in args]))



	def __call__(self, **inputs: SparkPayload) -> tuple[SparkPayload]:
		"""
			Update brain's states.
		"""
		# Put input into the cache.
		for input_name in self.config.input_map.keys():
			self._cache['__call__'][input_name].value = inputs[input_name].value
		# Update modules
		outputs = {}
		for module_name in self._modules_list:
			# Reconstruct module input using the cache cache.
			input_args = {}
			for port_name, ports_list in self._modules_input_map[module_name].items():
				# TODO: This does not support unflatten inputs.
				input_args[port_name] = self._concatenate_payloads(
					[self._cache[port_map.origin][port_map.port].value for port_map in ports_list])
			outputs[module_name] = getattr(self, module_name)(**input_args)
		# Update cache
		for module_name in self._modules_list:
			for port_name in self._modules_output_specs[module_name].keys():
				self._cache[module_name][port_name].value = outputs[module_name][port_name].value
		# Gather output
		brain_output = {
			f'{module_name}.{output_port}': outputs[module_name][output_port] 
			for module_name, port_specs in self.config.output_map.items() 
			for output_port in port_specs.keys()
		}
		return brain_output


	# TODO: This needs to be set differently, perhaps through some sort of mask. 
	def get_spikes_from_cache(self):
		"""
			Collect the brain's spikes.
		"""
		brain_spikes = {}
		for module_name in self._modules_list:
			if 'spikes' in self._cache[module_name]:
				brain_spikes[module_name] = self._cache[module_name]['spikes'].value
			elif 'out_spikes' in self._cache[module_name]:
				brain_spikes[module_name] = self._cache[module_name]['out_spikes'].value
		return brain_spikes

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################