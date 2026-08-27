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
from spark.core.backend import Constant
from spark.core.registry import register_module, register_config
from spark.core.decorators import spark_property, limit_recursion
from spark.core.specs import PortSpecs, PortMap
from spark.core.payloads import SparkPayload, SpikeArray, BooleanMask
from spark.core.config_validation import TypeValidator, ZeroOneValidator
from spark.nn.controllers.base import ControllerConfig, ControllerMeta, Controller

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NeuronMeta(ControllerMeta):
	"""
		Metaclass for `Neuron`.
	"""
	pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class NeuronConfig(ControllerConfig):
	"""
		Configuration for `Neuron`.

		Parameters
		----------
		modules_specs : tuple of ModuleSpecs
			The components the neuron holds, and how their ports are wired.
		units : tuple of int
			Shape of the pool of neurons.
		inhibitory_rate : float, default 0.2
			Fraction of the pool that is inhibitory. Must lie in ``[0, 1]``.
		seed : int, optional
			Seed for the random draws of the neuron and its modules. Drawn from the operating
			system when omitted.
		dt : float, default 1.0
			Integration step, in ms.

		Notes
		-----
		``dt`` and ``units`` are handed down to every configuration the neuron contains, so a pool
		is sized and clocked in one place. Both names are reserved for that purpose.
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

	@limit_recursion(limit=1)
	def __post_init__(self,) -> None:
		# NOTE: Convinience controller synchronization of dt's and unit's. 
		# Both are 'reserved' names to denote integration times and the number of neurons in the pool.
		self._synchronize(_s_dt=self.dt, _s_units=self.units)
	
#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: This class needs a proper way to set up the inhibitory masks and the recurrent contract.
@register_module
class Neuron(Controller, metaclass=NeuronMeta):
	"""
		Pool of neurons built from components.

		A neuron holds the components one neuron model is made of, such as delays, synapses, a
		soma and a plasticity rule, and steps them in dependency order within a single timestep.
		A module therefore reads what the modules before it produced on the same step, rather than
		on the previous one.

		Because the step is ordered, a cycle in the wiring can only be resolved if the module
		closing it declares a recurrent contract, which states what its outputs look like before
		it has run.

		Parameters
		----------
		config : NeuronConfig
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

		Properties
		----------
		inhibition_mask : BooleanMask
			Marks the inhibitory units of the pool. Read only, and supplied to any module declaring
			an ``inhibition_mask`` input without being wired.

		Notes
		-----
		Which units are inhibitory is a property of the pool, not of the module that emits the
		spikes. The mask is drawn once from ``inhibitory_rate`` and handed to any module declaring
		an ``inhibition_mask`` input, without being wired. From there the spikes carry the
		distinction themselves. A declared connection takes precedence.

		``inhibition_mask`` is exposed as a read-only property, so it can be read by the graph but
		not written.

		See Also
		--------
		Brain : Controller that steps its modules against a cache of the previous step.
		LIFNeuron : Prebuilt leaky integrate-and-fire neuron.
	"""
	config: NeuronConfig

	def __init__(self, config: NeuronConfig | None = None, **kwargs):
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

	# TODO: Should we allow properties to be defined inside the build method? (Safe proof this method)
	def recurrent_contract(
			self, 
		) -> tuple[dict[str, SparkPayload], dict[str, SparkPayload]]:
		"""
			Returns expected-like outputs and properties of the module.

			This function is a binding contract that allows the modules to accept self connections.
		"""
		# Output specs
		output_contract_specs = {k:v['spec'] for k,v in self._controller_output_specs.items()}
		for output_name, spec in output_contract_specs.items():
			spec = PortSpecs(
				payload_type=spec.payload_type,
				shape=self.units,
				dtype=spec.dtype,
				description=spec.description,
			)
			_mock: SparkPayload = spec._create_mock_payload()
			if isinstance(_mock, SpikeArray):
				_mock = SpikeArray(spikes=_mock.spikes, inhibition_mask=self._inhibition_mask.value)
			output_contract_specs[output_name] = _mock
		# Property specs. Properties should be defined inside __init__, so it is safe to inspect them.
		property_contract_specs = self._get_controller_property_specs()
		for property_name, spec in property_contract_specs.items():
			_property: SparkPayload = getattr(self, property_name, None)
			if _property is None:
				spec = property_contract_specs[property_name].spec
				spec = PortSpecs(
					payload_type=spec.payload_type,
					shape=spec.shape if spec is not None else self.units,
					dtype=spec.dtype if spec is not None else spec.dtype,
					#is_optional=spec.is_optional,
					description=spec.description,
				)
				_property: SparkPayload = spec._create_mock_payload()
				if isinstance(_property, SpikeArray):
					_property  = SpikeArray(spikes=_mock.spikes, inhibition_mask=self._inhibition_mask.value)
			property_contract_specs[property_name] = _property
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

	def _implicit_inputs(self, module_name: str) -> dict[str, SparkPayload]:
		"""
			Supplies the inhibition mask to any module that declares it.

			Which units of the pool are inhibitory is a property of the neuron, not of the
			module that emits the spikes, so it is not wired. A module that declares an
			"inhibition_mask" input receives it here and stamps it onto the spikes it emits;
			from that point the spikes are signed and the rest of the graph carries the
			distinction on its own. A declared connection still takes precedence.
		"""
		module = getattr(self, module_name)
		if 'inhibition_mask' not in type(module)._get_input_specs():
			return {}
		return {'inhibition_mask': BooleanMask(self._inhibition_mask.value)}

	def build(self, **abc_args: SparkPayload) -> None:
		# Get build order.
		self._order = self._execution_order(self._modules_specs)
		# Instantiate modules
		self._instantiate_modules(abc_args, self._order)

	def __call__(self, **inputs: SparkPayload) -> dict[str, SparkPayload]:
		"""
			Advances every module one step, in dependency order.

			A module reads what the modules before it produced on this same step. Effects are applied
			once every module has run.

			Parameters
			----------
			**inputs : SparkPayload
				One entry per input port of the neuron, as derived from the modules.

			Returns
			-------
			dict of str to SparkPayload
				One entry per output port of the neuron, as derived from the modules.
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
							input_args_list.append(outputs[port_map.origin][port_map.port])
					input_args[port_name] = self._concatenate_payloads(input_args_list)
				for port_name, value in self._implicit_inputs(name).items():
					input_args.setdefault(port_name, value)
				outputs[name] = getattr(self, name)(**input_args)
		# Compute effects
		# TODO: Currently effects require the ports to be defined inside a list. This is probably not desirable.
		for name, effects in self._modules_effects_map.items():
			for property_name, ports_list in effects.items():
				port_map = ports_list[0]
				if port_map.is_property:
					value = getattr(getattr(self, port_map.origin), port_map.port)
				else:
					value = outputs[port_map.origin][port_map.port]
				setattr(getattr(self, name), property_name, value)
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