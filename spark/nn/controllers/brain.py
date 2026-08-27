#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
from spark.core.backend import data
from spark.core.registry import register_module, register_config
from spark.core.cache import Cache
from spark.core.specs import PortSpecs, PortMap
from spark.core.payloads import SparkPayload
from spark.nn.controllers.base import ControllerConfig, ControllerMeta, Controller

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class BrainMeta(ControllerMeta):
	"""
		Metaclass for `Brain`.
	"""
	pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class BrainConfig(ControllerConfig):
	"""
		Configuration for `Brain`.

		Parameters
		----------
		modules_specs : tuple of ModuleSpecs
			The neurons and interfaces the brain holds, and how their ports are wired.
		seed : int, optional
			Seed for the random draws of the brain and its modules. Drawn from the operating
			system when omitted.
		dt : float, default 1.0
			Integration step, in ms. Handed down to every module.
	"""
	pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class Brain(Controller, metaclass=BrainMeta):
	"""
		Network of neurons and interfaces.

		Every module reads from a cache holding the outputs of the previous step, updates its own
		state, and writes its outputs back. Because no module waits for another, any wiring is
		legal, cycles included, and the modules of one step are independent of each other.
		Note that due to implementation details, modules within this controller have a one step
		latency per connection; which, for most cases, is negligible.

		Parameters
		----------
		config : BrainConfig, optional
				Controller configuration. Its fields may also be given as keyword arguments.

		Input Ports
		-----------
		**inputs : SparkPayload
			Derived from the modules: a module input wired to ``__call__`` becomes an input port of
			the controller, under the name the port map gives it.

		Output Ports
		------------
		**outputs : SparkPayload
			Derived from the modules: a module output named in ``outputs`` becomes an output port of
			the controller.

		Notes
		-----
		Input and output ports are derived from the modules, as for any controller.

		See Also
		--------
		Neuron : Controller without the cache, where a step is ordered by its dependencies.
	"""
	config: BrainConfig

	def __init__(self, config: BrainConfig | None = None, **kwargs) -> None:
		# Initialize super.
		super().__init__(config=config, **kwargs)

	def build(self, **abc_args: SparkPayload) -> None:
		# Get build order.
		execution_order = self._execution_order(self._modules_specs)
		# Instantiate modules
		modules_outputs = self._instantiate_modules(abc_args, execution_order)
		# Build cache.
		self._cache = data(Cache.from_payloads(modules_outputs))

	def __call__(self, **inputs: SparkPayload) -> dict[str, SparkPayload]:
		"""
			Advances every module one step against the cache.

			Every module reads the outputs of the previous step, so the modules of one step are
			independent of each other. The cache is written once all of them have run.

			Parameters
			----------
			**inputs : SparkPayload
				One entry per input port of the brain, as derived from the modules.

			Returns
			-------
			dict of str to SparkPayload
				One entry per output port of the brain, as derived from the modules.
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
						input_args_list.append(self._cache[port_map.origin, port_map.port])
				input_args[port_name] = self._concatenate_payloads(input_args_list)
			outputs[name] = getattr(self, name)(**input_args)
		# Update cache
		for name in self._modules_names:
			for port_name in self._modules_output_map[name]:
				self._cache[name, port_name] = outputs[name][port_name]
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
				value = self._cache[port_map.origin, port_map.port]
			readout[port_map.origin][port_map.port] = value
		return readout

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################