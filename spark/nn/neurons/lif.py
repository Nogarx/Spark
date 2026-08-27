#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import dataclasses as dc

from spark.core.registry import register_config, register_neuron
from spark.core.specs import PortMap, ModuleSpecs
from spark.nn.controllers import Neuron, NeuronConfig
from spark.nn.components.delays.n2n_delays import N2NDelays, N2NDelaysConfig
from spark.nn.components.synapses.linear import LinearSynapses, LinearSynapsesConfig
from spark.nn.components.somas.leaky import AdaptiveLeakySoma, AdaptiveLeakySomaConfig
from spark.nn.components.plasticity.hebbian_rule import HebbianRule, HebbianRuleConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class LIFNeuronConfig(NeuronConfig):
	"""
		Configuration for `LIFNeuron`.

		Parameters
		----------
		modules_specs : tuple of ModuleSpecs
			The four components listed under `LIFNeuron`, prewired. Replace an entry to swap a
			component, or edit its configuration to retune one.
		units : tuple of int
			Shape of the pool of neurons.
		inhibitory_rate : float, default 0.2
			Fraction of the pool that is inhibitory.
		seed : int, optional
			Seed for the random draws of the neuron and its modules.
		dt : float, default 1.0
			Integration step, in ms.

		Notes
		-----
		The default parameters of the components have not been calibrated against any particular
		dataset or firing regime.
	"""
	
	modules_specs: tuple[ModuleSpecs, ...] = dc.field(
		default = (
			# N2N delays
			ModuleSpecs(
				name ='delays', 
				module_cls = N2NDelays, 
				inputs = {
					'in_spikes': [PortMap(origin='__call__', port='in_spikes')],
				},
				config = N2NDelaysConfig.partial(),
			),
			# Linear synapses
			ModuleSpecs(
				name ='synapses', 
				module_cls = LinearSynapses, 
				inputs = {
					'spikes': [PortMap(origin='delays', port='out_spikes')],
				},
				effects = {
					'kernel': [PortMap(origin='hebbian_rule', port='kernel')],
				},
				config = LinearSynapsesConfig.partial(),
			),
			# Leaky soma with an absolute refractory period
			ModuleSpecs(
				name ='soma', 
				module_cls = AdaptiveLeakySoma, 
				inputs = {
					'current': [PortMap(origin='synapses', port='currents')],
				},
				outputs = {
					'out_spikes': 'spikes', 
				},
				config = AdaptiveLeakySomaConfig.partial(cooldown=3.0),
			),
			# Hebbian plasticity
			ModuleSpecs(
				name ='hebbian_rule', 
				module_cls = HebbianRule, 
				inputs = {
					'pre_spikes': [PortMap(origin='delays', port='out_spikes')],
					'post_spikes': [PortMap(origin='soma', port='spikes')],
					'kernel': [PortMap(origin='synapses', port='kernel', is_property=True)],
				},
				config = HebbianRuleConfig.partial(),
			),
		),
		metadata = {
			'description': 'Neuron component modules.',
		})

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_neuron
class LIFNeuron(Neuron):
	"""
		Leaky integrate-and-fire neuron with plastic synapses.

		A prewired `Neuron` holding four components:

		* ``delays``, an `N2NDelays` conduction delay, one per connection.
		* ``synapses``, `LinearSynapses`, whose weights the plasticity rule writes back.
		* ``soma``, an `AdaptiveLeakySoma` with a 3 ms refractory period.
		* ``hebbian_rule``, a `HebbianRule` reading the delayed presynaptic spikes, the emitted
		  spikes and the current weights.

		Parameters
		----------
		config : LIFNeuronConfig, optional
				Controller configuration. Its fields may also be given as keyword arguments.

		Input Ports
		-----------
		in_spikes : SpikeArray
			Spikes arriving at the pool.

		Output Ports
		------------
		out_spikes : SpikeArray
			Spikes emitted by the pool on this step.

		Properties
		----------
		inhibition_mask : BooleanMask
			Marks the inhibitory units of the pool. Read only.

		Notes
		-----
		Only the refractory period is enabled on the soma. Threshold adaptation and the adaptation
		current are available by setting their trigger parameters on the soma configuration.

		See Also
		--------
		ALIFNeuron : The same neuron with threshold adaptation.
		AdExNeuron : Adaptive exponential soma in place of the leaky one.
	"""
	config: LIFNeuronConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################