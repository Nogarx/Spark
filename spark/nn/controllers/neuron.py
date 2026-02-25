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
from spark.core.variables import Variable, Constant
from spark.core.payloads import SparkPayload, FloatArray, SpikeArray, ValueSparkPayload
from spark.core.registry import register_config, REGISTRY
from spark.core.config import BaseSparkConfig
import jax

from spark.nn.components.base import Component
from spark.nn.components.somas import Soma
from spark.nn.components.delays import Delays
from spark.nn.components.synapses import Synanpses
from spark.nn.components.learning_rules import LearningRule
from spark.core.config_validation import TypeValidator, PositiveValidator, ZeroOneValidator
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

# TODO: We need to alliviate the need to define input/output maps
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

class Neuron(Controller, metaclass=NeuronMeta):
	"""
		Neuron model.

		A neuron is a pipeline object used to represent and coordinate a collection of neurons and interfaces.
		This implementation relies on a cache system to simplify parallel computations; every timestep all the modules
		in the Neuron read from the cache, update its internal state and update the cache state. 
		Note that this introduces a small latency between elements of the neuron, which for most cases is negligible, and for
		such a reason it is recommended that only full neuron models and interfaces are used within a Neuron.
	"""
	config: NeuronConfig

	# Typing annotations.
	_execution_order: list[list[type]]

	# TODO: This is poorly implemented and will break soon rather than later.
	def __init__(self, config: NeuronConfig = None, **kwargs):
		# Initialize super.
		super().__init__(config=config, **kwargs)
		# Extract units
		self.units = utils.validate_shape(self.config.units)
		self._units = prod(self.units)
		# Initialize inhibitory mask.
		inhibitory_units = int(self._units * self.config.inhibitory_rate)
		indices = jax.random.permutation(self.get_rng_keys(1), jnp.arange(self._units), independent=True)[:inhibitory_units]
		inhibition_mask = jnp.zeros((self._units,), dtype=jnp.bool)
		inhibition_mask = inhibition_mask.at[indices].set(True).reshape(self.units)
		self._inhibition_mask = Constant(inhibition_mask, dtype=jnp.bool)
		# Set output shapes earlier to allow cycles.
		if False:
			self.set_contract_specs(
				output_contract_specs={
					'out_spikes': PortSpecs(
						payload_type=SpikeArray,
						shape=self.units,
						dtype=jnp.float16
					)
				},
				property_contract_specs={},
			)
		# Build modules
		self._build_modules()



	# TODO: Current approach to build the cache is rather silly, there should be a better more robust way to construct it.
	def build(self, input_specs: dict[str, PortSpecs]):
		# Get build order.
		self._execution_order = self.execution_order()
		# Build cache. Cache needs to be computed a step at a time. 
		modules_output_specs = {'__call__': {name: spec for name, spec in input_specs.items()}}
		modules_property_specs = {}
		for group in self._execution_order:
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
								_, spec = origin_module.get_contract_specs()[port_map.port]
							else:
								spec, _ = origin_module.get_contract_specs()[port_map.port]
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
			Update neuron's states.
		"""
		outputs = {}
		# Iterate over execution order groups
		for module_group in self._execution_order:
			for module_name in module_group:
				# Reconstruct module input using the current inputs/properties/workspace
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
							input_args_list.append(outputs[port_map.origin][port_map.port])
					input_args[port_name] = self._concatenate_payloads(input_args_list)
				outputs[module_name] = getattr(self, module_name)(**input_args)
		# Gather output
		neuron_output = {
			name: outputs[origin][port] for name, origin, port in self._flat_output_map 
		}
		return neuron_output

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################