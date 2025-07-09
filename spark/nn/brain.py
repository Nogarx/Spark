#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import jax.numpy as jnp
import inspect
from math import prod
from typing import Dict, List
from spark.core.module import SparkModule, SparkMeta
from spark.core.specs import PortSpecs, PortMap, CacheSpec, OutputSpecs, InputSpecs, ModuleSpecs
from spark.core.variable_containers import Variable
from spark.core.shape import Shape, normalize_shape
from spark.core.payloads import SparkPayload

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class BrainMeta(SparkMeta):
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Brain(SparkModule, metaclass=BrainMeta):
	"""
		Abstract brain model.
		This is more a convenience class used to synchronize data more easily.
	"""

	# Typing annotations.
	_input_map: Dict[str, PortSpecs]
	_output_map: Dict[str, Dict[str, PortSpecs]]
	_input_shapes: List[Shape]
	_output_shapes: List[Shape]
	_modules_list: List[str]
	_cache: Dict[str, Dict[str, CacheSpec]]
	_modules_input_map: Dict[str, Dict[str, List[PortMap]]]
	_modules_input_specs: Dict[str, Dict[str, InputSpecs]]
	_modules_output_specs: Dict[str, Dict[str, OutputSpecs]]

	def __init__(self, input_map: Dict[str, PortSpecs], output_map: Dict[str, Dict[str, PortSpecs]], modules_map: Dict[str, ModuleSpecs], **kwargs):
		super().__init__(**kwargs)
		# Validate maps.
		self._validate_maps(input_map, output_map, modules_map)
		# Build modules
		self._build_modules(modules_map)
		# Validate modules connections.
		self._validate_connections(input_map, modules_map)
		# Input/Output shapes
		self._input_map = input_map
		self._output_map = output_map
		self._input_shapes = [spec.shape for spec in self._input_map.values()]
		self._output_shapes = [spec.shape for output_port in self._output_map.values() for spec in output_port.values()]
		# Build cache.
		self._build_cache()
		# Replace call signature
		#self._override_signatures()

	def _validate_maps(self, input_map: Dict[str, PortSpecs], output_map: Dict[str, Dict[str, PortSpecs]], modules_map: Dict[str, ModuleSpecs]):
		# Basic Type Validation
		if not isinstance(input_map, dict):
			raise TypeError(f'"input_map" must be a dictionary, but got {type(input_map).__name__}.')
		for key in input_map.keys():
			if not isinstance(key, str):
				raise TypeError(f'All keys in "input_map" must be strings, but found key "{key}" of type {type(key).__name__}.')
			if not isinstance(input_map[key], PortSpecs):
				raise TypeError(f'All values in "input_map" must be PortSpecs, but found value "{input_map[key]}" of type {type(input_map[key]).__name__}.')
			
		if not isinstance(output_map, dict):
			raise TypeError(f'"output_map" must be a dictionary, but got {type(output_map).__name__}.')
		for module_name, module_output_details in output_map.items():
			if not isinstance(module_name, str):
				raise TypeError(f'All keys in "output_map" must be strings, but found key "{module_name}" of type {type(module_name).__name__}.')
			if not isinstance(module_output_details, dict):
				raise TypeError(f'All values in "output_map" must be dict, but found value "{module_output_details}" of type {type(module_output_details).__name__}.')
			for port_name, port_spec in module_output_details.items():
				if not isinstance(key, str):
					raise TypeError(f'All keys in "output_map["module"]" must be strings, but found key "{port_name}" of type {type(port_name).__name__}.')
				if not isinstance(port_spec, PortSpecs):
					raise TypeError(f'All values in "output_map["module"]" must be PortSpecs, but found value "{port_spec}" of type {type(port_spec).__name__}.')

		if not isinstance(modules_map, dict):
			raise TypeError(f'"modules_map" must be a dictionary, but got {type(modules_map).__name__}.')
		for key in modules_map.keys():
			if not isinstance(key, str):
				raise TypeError(f'All keys in "modules_map" must be strings, but found key "{key}" of type {type(key).__name__}.')
			if not isinstance(modules_map[key], ModuleSpecs):
				raise TypeError(f'All values in "modules_map" must be ModuleSpecs, but found value "{modules_map[key]}" of type {type(modules_map[key]).__name__}.')
			
		# Get all valid sources.
		valid_sources = set(input_map.keys()) | set(modules_map.keys())

		# Does the origin exist in the set of valid sources?
		for module_name, module_specs in modules_map.items():
			for port_name, connections in module_specs.inputs.items():
				for port_map in connections:
					if port_map.origin not in valid_sources and port_map.origin != '__call__':
						raise ValueError(f'In module "{module_name}", connection for port "{port_name}" refers to an unknown origin: "{port_map.origin}".'
					   					 f'Valid sources are: {valid_sources}')
			
	def _build_modules(self, modules_map: Dict[str, ModuleSpecs]):
		# Registry lazy import
		from spark.core.registry import REGISTRY
		# Construc all modules.
		for module_name, module_specs in modules_map.items():
			setattr(self, module_name, REGISTRY.MODULES.get(module_specs.module_cls).class_ref(**module_specs.init_args))
		self._modules_list = list(modules_map.keys())
		# Information flow map.
		self._modules_input_map = {}
		for module_name, module_specs in modules_map.items():
			self._modules_input_map[module_name] = module_specs.inputs

	def _validate_connections(self, input_map: Dict[str, PortSpecs], modules_map: Dict[str, ModuleSpecs]):
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
			module_inputs: dict[str, List[PortMap]] = modules_map[module_name].inputs
			required_ports = set(self._modules_input_specs[module_name].keys()).difference(module_inputs.keys())
			for port_name in required_ports:
				if not self._modules_input_specs[module_name][port_name].is_optional:
					class_error = REGISTRY.MODULES.get(modules_map[module_name].module_cls).class_ref
					raise ValueError(f'Port "{port_name}" for module "{module_name}" is not defined and is a mandatory input of "{class_error.__name__}".')
			
			# Validate that no extra input port were defined
			extra_ports = set(module_inputs.keys()).difference(set(self._modules_input_specs[module_name].keys()))
			if len(extra_ports) > 0:
				raise ValueError(f'Port "{extra_ports[0]}" for module "{module_name}" is not part of the specifications of '
					 			 f'"{modules_map[module_name].module_cls}".')
			
			# Validate the integrity of the ports 
			for port_name, ports_list in module_inputs.items():
				
				# Check input module defines the output port
				for port_map in ports_list:
					if port_map.origin == '__call__':
						if not port_map.port in input_map:
							raise ValueError(f'Input port "{port_name}" for module "{module_name}" request port "{port_map.port}" from '
											 f'"{port_map.origin}", but inputs_map does not define such port.')
					elif not port_map.port in self._modules_output_specs[port_map.origin]:
						raise ValueError(f'Input port "{port_name}" for module "{module_name}" request port "{port_map.port}" from '
										 f'"{port_map.origin}", but "{modules_map[port_map.origin].module_cls}" does not define such port.')
				
				# Validate ports can be safely merged.
				if len(ports_list) > 1:
					payload_type = self._modules_output_specs[ports_list[0].origin][ports_list[0].port].payload_type
					for i in range(1,len(ports_list)):
						target_payload_type = self._modules_output_specs[ports_list[i].origin][ports_list[i].port].payload_type
						if not payload_type == target_payload_type:
							raise TypeError(f'Input port "{port_name}" for module "{module_name}" request port "{ports_list[i].port}" from '
											f'"{ports_list[i].origin}", but payload type is not compatible, expected "{payload_type}" and got "{target_payload_type}".') 
				
				# Validate shapes
				shape = 0
				if len(ports_list) > 1:
					# Many-to-one input-output. Inputs need to be merged, default merged behaviour is to flat everything.
					for port_map in ports_list:
						if port_map.origin == '__call__':
							shape += prod(input_map[port_map.port].shape)
						else:
							output_specs: OutputSpecs = getattr(self, port_map.origin).get_output_specs()[port_map.port]
							shape += prod(output_specs.shape)
				else:
					# One-to-one input-output.
					if port_map.origin == '__call__':
						shape = input_map[port_map.port].shape
					else:
						output_specs: OutputSpecs = getattr(self, port_map.origin).get_output_specs()[port_map.port]
						shape = output_specs.shape
				# Normalize and compare shapes.
				shape = normalize_shape(shape)
				expected_input_shape = self._modules_input_specs[module_name][port_name].shape
				if expected_input_shape != shape:
					raise ValueError(f'Input port "{port_name}" for module "{module_name}" expected input shape {expected_input_shape} but got shape "{shape}".') 

	def _build_cache(self,):
		self._cache = {}
		# Raw model input need to be treated as "outputs" of some virtual node.
		self._cache['__call__'] = {}
		for input_name, input_specs in self._input_map.items():
			self._cache['__call__'][input_name] = CacheSpec(
				var=Variable(jnp.zeros(input_specs.shape), dtype=input_specs.dtype),
				payload_type=input_specs.payload_type,
				dtype=input_specs.dtype
			)
		# Modules outputs
		for module_name in self._modules_list:
			self._cache[module_name] = {}
			for port_name, port_specs in self._modules_output_specs[module_name].items():
				self._cache[module_name][port_name] = CacheSpec(
				var=Variable(jnp.zeros(port_specs.shape), dtype=port_specs.dtype),
				payload_type=port_specs.payload_type,
				dtype=port_specs.dtype
			)

	def _override_call(self,):
		# Create new signature
		annotations = [inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
		for port_name, port_specs in self._input_map.items():
			arg_annotation = inspect.Parameter(
				name=port_name,
				kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
				annotation=port_specs.payload_type
			)
			annotations.append(arg_annotation)
		
		# Replace signature
		self.__call__.__signature__ = inspect.Signature(annotations)

	@property
	def input_shapes(self,) -> List[Shape]:
		return self._input_shapes

	@property
	def output_shapes(self,) -> List[Shape]:
		return self._output_shapes

	def get_input_specs(self) -> Dict[str, PortSpecs]:
		return self._input_map

	def get_output_specs(self) -> Dict[str, PortSpecs]:
		return self._output_map

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
			Resets neuron state to their initial values.
		"""
		# Build components list. TODO: This should be called post init. 
		if self._neuron_names is not None:
			self._build_neuron_list()
		# Reset components.
		for name in self._neuron_names:
			getattr(self, name).reset()

	#@abc.abstractmethod
	def __call__(self, **inputs: SparkPayload) -> tuple[SparkPayload]:
		"""
			Update brain's states.
		"""
		# Put input into the cache.
		for input_name in self._input_map.keys():
			self._cache['__call__'][input_name].var.value = inputs[input_name].value
		# Update modules
		outputs = {}
		for module_name in self._modules_list:
			# Reconstruct module input using the cache cache.
			input_args = {}
			for port_name, ports_list in self._modules_input_map[module_name].items():
				# TODO: This does not support unflatten inputs.
				args = jnp.concatenate([self._cache[port_map.origin][port_map.port].var.value.reshape(-1) for port_map in ports_list]) 
				input_args[port_name] = self._cache[ports_list[0].origin][ports_list[0].port].payload_type(args)
			outputs[module_name] = getattr(self, module_name)(**input_args)
		# Update cache
		for module_name in self._modules_list:
			for port_name in self._modules_output_specs[module_name].keys():
				self._cache[module_name][port_name].var.value = outputs[module_name][port_name].value
		# Gather output
		brain_output = {
			f'{module_name}.{output_port}': outputs[module_name][output_port] 
			for module_name, port_specs in self._output_map.items() 
			for output_port in port_specs.keys()
		}
		brain_spikes = []
		for module_name in self._modules_list:
			if 'spikes' in outputs[module_name]:
				brain_spikes.append(outputs[module_name]['spikes'])
		brain_spikes = tuple(brain_spikes)
		return brain_output, brain_spikes
	
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################