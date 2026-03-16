#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
	
import jax.numpy as jnp
import typing as tp
import dataclasses as dc
from math import prod
import spark.core.utils as utils
from spark.core.cache import Cache, CacheEntry
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

	def __init__(self, config: BrainConfig | None = None, **kwargs) -> None:
		# Initialize super.
		super().__init__(config=config, **kwargs)

	def build(self, input_specs: dict[str, PortSpecs]) -> None:
		# Get build order.
		order = self._execution_order(self._modules_specs)
		# Build cache. Cache needs to be computed a step at a time. 
		modules_output_specs = utils.TwoKeyDict()
		modules_output_specs['__call__'] = input_specs
		modules_property_specs = utils.TwoKeyDict()
		for group in order:
			for module_name in group:
				# Skip __call__
				if module_name == '__call__':
					continue
				# Collect module specs and construct a mock input
				mock_input = {}
				validate_async = True
				for port_name, port_map_list in self._modules_inputs_map[module_name].items():
					portspecs_list = []
					for port_map in port_map_list:
						if port_map.origin in modules_output_specs:
							# Modules was already built grab, get the spec.
							if port_map.is_property:
								spec = modules_property_specs[port_map.origin, port_map.port]
							else:
								spec = modules_output_specs[port_map.origin, port_map.port]
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
							# Something weird happend. The constructor is trying to get something from a module that should have been called later.
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
		self._cache = Cache.from_specs(modules_output_specs)
		# Set built flag
		self.__built__ = True

	def __call__(self, **inputs: SparkPayload) -> dict[str, SparkPayload]:
		"""
			Update brain's states.
		"""
		# Update modules
		outputs = {}
		for name in self._modules_names:
			# Reconstruct module input using the current inputs/properties/cache
			input_args = {}
			for port_name, ports_list in self._modules_inputs_map[name].items():
				# TODO: This does not support unflatten inputs.
				input_args_list = []
				for port_map in ports_list:
					if port_map.origin == '__call__':
						input_args_list.append(inputs[port_map.port])
					elif port_map.is_property:
						input_args_list.append(getattr(getattr(self, port_map.origin), port_map.port))
					else:
						input_args_list.append(self._cache[port_map.origin, port_map.port].get())
				input_args[port_name] = self._concatenate_payloads(input_args_list)
			outputs[name] = getattr(self, name)(**input_args)
		# Update cache
		for name in self._modules_names:
			for port_name in self._modules_output_map[name]:
				self._cache[name, port_name].set(outputs[name][port_name])
		# Compute effects

		# Gather output
		return {
			name: outputs[origin][port] for name, origin, port in self._contoller_output_map 
		}

	def read_state(self, port_list: list[PortMap]) -> dict:
		"""
			Returns the current state of the modules/cache.
		"""
		readout = {port_map.origin: {} for port_map in port_list}
		for port_map in port_list:
			if port_map.is_property:
				value = getattr(getattr(self, port_map.origin), port_map.port)
			else:
				value = self._cache[port_map.origin, port_map.port].get()
			readout[port_map.origin][port_map.port] = value
		return readout

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################