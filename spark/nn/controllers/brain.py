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

from spark.nn.controllers.base import ControllerConfig, ControllerMeta, Controller

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class BrainMeta(ControllerMeta):
	"""
		Brain metaclass.
	"""
	pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class BrainConfig(ControllerConfig):
	"""
		Configuration class for Brain's.
	"""
	pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Brain(Controller, metaclass=BrainMeta):
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
	_cache: dict[str, dict[str, Cache]]

	def __init__(self, config: BrainConfig = None, **kwargs):
		# Initialize super.
		super().__init__(config=config, **kwargs)
		# Build modules
		self._build_modules()



	# TODO: Current approach to build the cache is rather silly, there should be a better more robust way to construct it.
	def build(self, input_specs: dict[str, PortSpecs]):
		# Get build order.
		order = self.execution_order()
		# Build cache. Cache needs to be computed a step at a time. 
		modules_output_specs = {'__call__': {name: spec for name, spec in input_specs.items()}}
		modules_property_specs = {}
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
							if port_map.is_property:
								spec = modules_property_specs[port_map.origin][port_map.port]
							else:
								spec = modules_output_specs[port_map.origin][port_map.port]
							portspecs_list.append(spec)
						elif port_map.origin in group:
							# Module is not built yet, it must define a recurrent spec to be part of a cyclic dependency.
							origin_module: SparkModule = getattr(self, port_map.origin)
							if port_map.is_property:
								_, spec = origin_module.get_contract_specs()
								spec = spec[port_map.port]
							else:
								spec, _ = origin_module.get_contract_specs()
								spec = spec[port_map.port]
							portspecs_list.append(spec)
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
				modules_property_specs[module_name] = module.get_property_specs()
		# Build cache.
		self._cache = {
			module_name : {
				port: Cache._from_spec(spec) for port, spec in module_output_specs.items()
			} for module_name, module_output_specs in modules_output_specs.items() if module_name != '__call__'
		}
		# Set built flag
		self.__built__ = True
		# Get a simplified version of the output map for __call__ iteration.
		self._flat_output_map = []
		for output_name, output_details in self.config.output_map.items():
			self._flat_output_map.append((output_name, output_details['input'].origin, output_details['input'].port))
		# Validate modules connections.
		self._validate_connections()



	def __call__(self, **inputs: SparkPayload) -> dict[str, SparkPayload]:
		"""
			Update brain's states.
		"""
		# Update modules
		outputs = {}
		for module_name in self._modules_list:
			# Reconstruct module input using the current inputs/properties/cache
			input_args = {}
			for port_name, ports_list in self._modules_input_map[module_name].items():
				# TODO: This does not support unflatten inputs.
				input_args_list = []
				for port_map in ports_list:
					if port_map.origin == '__call__':
						input_args_list.append(inputs[port_map.port])
					elif port_map.is_property:
						input_args_list.append(getattr(port_map.origin, port_map.port))
					else:
						input_args_list.append(self._cache[port_map.origin][port_map.port].get())
				input_args[port_name] = self._concatenate_payloads(input_args_list)
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

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################