#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import jax
import jax.numpy as jnp
import dataclasses as dc
from math import prod

import spark.core.utils as utils
from spark.core.variables import Constant
from spark.core.registry import register_config
from spark.core.decorators import spark_property
from spark.core.specs import PortSpecs, PortMap
from spark.core.payloads import SparkPayload, SpikeArray, BooleanMask
from spark.core.config_validation import TypeValidator, ZeroOneValidator
from spark.nn.controllers.base import ControllerConfig, ControllerMeta, Controller

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NeuronMeta(ControllerMeta):
	"""
		Neuron metaclass.
	"""
	pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class NeuronConfig(ControllerConfig):
	"""
		Configuration class for Neuron's.
	"""
	units: tuple[int, ...] = dc.field(
		metadata = {
		'validators': [
			TypeValidator,
		],
		'description': 'Shape of the pool of neurons.',
		}
	)
	inhibitory_rate: float = dc.field(
		default = 0.2, 
		metadata = {
			'units': 'ms',
			'validators': [
				TypeValidator,
				ZeroOneValidator,
			],
			'description': '',
		}
	)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: This class needs a proper way to set up the inhibitory masks and the recurrent contract.
class Neuron(Controller, metaclass=NeuronMeta):
	"""
		Neuron model.

		A neuron is a pipeline object used to represent and coordinate a collection of neurons and interfaces.
	"""
	config: NeuronConfig

	def __init__(self, config: NeuronConfig | None = None, **kwargs):
		# Initialize super.
		super().__init__(config=config, **kwargs)
		# Extract units
		self.units = utils.validate_shape(self.config.units)
		self._units = prod(self.units)
		# TODO: Below is a manual override to synchronize all the units across the controller.
		# This solution is probably good enough but it is not clear that will not clash with other user intentions.
		self.config.merge(partial={'_s_units': self.units})
		# Initialize inhibitory mask.
		inhibitory_units = int(self._units * self.config.inhibitory_rate)
		indices = jax.random.permutation(self.get_rng_keys(1), jnp.arange(self._units), independent=True)[:inhibitory_units]
		inhibition_mask = jnp.zeros((self._units,), dtype=jnp.bool)
		inhibition_mask = inhibition_mask.at[indices].set(True).reshape(self.units)
		self._inhibition_mask = Constant(inhibition_mask, dtype=jnp.bool)

	# TODO: Should we allow properties to be defined inside the build method? (Safe proof this method)
	def recurrent_contract(
			self, 
		) -> None:
		"""
			Returns the expected specs for the outputs and properties of the module.

			This function is a binding contract that allows the modules to accept self connections.
		"""
		# Output specs
		output_contract_specs = self._controller_output_specs
		for output_name, spec in output_contract_specs.items():
			output_contract_specs[output_name] = PortSpecs(
				payload_type=spec.payload_type,
				shape=self.units,
				dtype=spec.dtype,
				description=spec.description,
			)
		# Property specs. Properties should be defined inside __init__, so it is safe to inspect them.
		property_contract_specs = self._get_controller_property_specs()
		for property_name, spec in property_contract_specs.items():
			_property: SparkPayload = getattr(self, property_name, None)
			property_contract_specs[property_name] = PortSpecs(
				payload_type=spec.payload_type,
				shape=_property.shape if _property is not None else self.units,
				dtype=_property.dtype if _property is not None else spec.dtype,
				description=spec.description,
			)
		return output_contract_specs, property_contract_specs

	@classmethod
	def has_recurrent_contract(cls) -> bool:
		"""
			Returns True if the modules defines a recurrent contract, False otherwise.
		"""
		return True

	@spark_property
	def inhibition_mask(self,) -> BooleanMask:
		return BooleanMask(self._inhibition_mask.value)

	def build(self, input_specs: dict[str, PortSpecs]) -> None:
		# Get build order.
		self._order = self._execution_order(self._modules_specs)
		# Instantiate modules
		_, _ = self._instantiate_modules(input_specs, self._order)

	def __call__(self, **inputs: SparkPayload) -> dict[str, SparkPayload]:
		"""
			Update neuron's states.
		"""
		# Iterate over execution order groups
		outputs = {}
		for module_group in self._order:
			for name in module_group:
				# Reconstruct module input using the current inputs/properties/workspace
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
		# Compute effects
		# TODO: Currently effects require the ports to be defined inside a list. This is probably not desirable.
		for name, effects in self._modules_effects_map.items():
			for property_name, ports_list in effects.items():
				port_map = ports_list[0]
				setattr(getattr(self, name), property_name, outputs[port_map.origin, port_map.port])
		# Gather output
		return {
			name: outputs[origin][port] for name, origin, port in self._contoller_output_map 
		}

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
						input_args_list.append(outputs[port_map.origin][port_map.port])
				input_args[port_name] = self._concatenate_payloads(input_args_list)
			outputs[name] = getattr(self, name)(**input_args)
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
				readout[port_map.origin][port_map.port] = getattr(getattr(self, port_map.origin), port_map.port)
			else:
				raise ValueError(
					'Reading non-property variables is not supported by Neuron Controller objects.'
				)
		return readout

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################