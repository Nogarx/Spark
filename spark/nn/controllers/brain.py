#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

from spark.core.registry import register_config
from spark.core.cache import Cache
from spark.core.specs import PortSpecs, PortMap
from spark.core.payloads import SparkPayload
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
		execution_order = self._execution_order(self._modules_specs)
		# Instantiate modules
		modules_output_specs, _ = self._instantiate_modules(input_specs, execution_order)
		# Build cache.
		self._cache = Cache.from_specs(modules_output_specs)

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
						if port_map.origin == '__self__':
							input_args_list.append(getattr(self, port_map.port))
						else:
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
		for name, effects in self._modules_effects_map.items():
			for property_name, ports_list in effects.items():
				# TODO: It is unclear whether it is necessary or ideal to support multi-port inputs for effects.
				# Currently we only accept the first defined input for a property port. 
				port_map = ports_list[0]
				setattr(getattr(self, name), property_name, outputs[port_map.origin, port_map.port])
		# Gather output
		return {
			name: outputs[origin][port] for name, origin, port in self._contoller_output_map 
		}

	#@partial(jax.jit, static_argnames=['port_list']) 
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