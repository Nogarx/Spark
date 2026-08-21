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
        Standard Adaptive Exponential (AdEx) neuron model with traced synapses, neuron-to-neuron delays and Hebbian learning.
		
		NOTE: Parameter calibration is still necessary.
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
			# Leaky soma
			ModuleSpecs(
				name ='soma', 
				module_cls = AdaptiveExponentialSoma, 
				inputs = {
					'current': [PortMap(origin='synapses', port='currents')],
					'inhibition_mask': [PortMap(origin='__self__', port='inhibition_mask', is_property=True)],
				},
				outputs = {
					'out_spikes': 'spikes', 
				},
				config = AdaptiveExponentialSomaConfig.partial(),
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
	config: AdExNeuronConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################