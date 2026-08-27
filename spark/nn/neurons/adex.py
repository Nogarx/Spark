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
from spark.nn.components.synapses.traced import TracedSynapses, TracedSynapsesConfig
from spark.nn.components.somas.exponential import AdaptiveExponentialSoma, AdaptiveExponentialSomaConfig
from spark.nn.components.plasticity.hebbian_rule import HebbianRule, HebbianRuleConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class AdExNeuronConfig(NeuronConfig):
	"""
		Configuration for `AdExNeuron`.

		Parameters
		----------
		modules_specs : tuple of ModuleSpecs
			The four components listed under `AdExNeuron`, prewired. Replace an entry to swap a
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
				module_cls = TracedSynapses, 
				inputs = {
					'spikes': [PortMap(origin='delays', port='out_spikes')],
				},
				effects = {
					'kernel': [PortMap(origin='hebbian_rule', port='kernel')],
				},
				config = TracedSynapsesConfig.partial(),
			),
			# Exponential soma with a spike triggered and subthreshold adaptation current
			ModuleSpecs(
				name ='soma', 
				module_cls = AdaptiveExponentialSoma, 
				inputs = {
					'current': [PortMap(origin='synapses', port='currents')],
				},
				outputs = {
					'out_spikes': 'spikes', 
				},
				config = AdaptiveExponentialSomaConfig.partial(adaptation_delta=7.0),
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
class AdExNeuron(Neuron):
	"""
		Adaptive exponential integrate-and-fire neuron with plastic synapses.

		A prewired `Neuron` holding four components:

		* ``delays``, an `N2NDelays` conduction delay, one per connection.
		* ``synapses``, `TracedSynapses`, giving each spike an exponential postsynaptic current.
		* ``soma``, an `AdaptiveExponentialSoma` with an adaptation current rising by 7 pA per
		  spike.
		* ``hebbian_rule``, a `HebbianRule` reading the delayed presynaptic spikes, the emitted
		  spikes and the current weights.

		The adaptation current subtracts from the input as the unit fires, which is the AdEx
		mechanism behind spike-frequency adaptation and bursting.

		Parameters
		----------
		config : AdExNeuronConfig, optional
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

		See Also
		--------
		ALIFNeuron : Adaptation through the threshold rather than through a current.
		LIFNeuron : Leaky soma without adaptation.
	"""
	config: AdExNeuronConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################