from spark.nn.components.plasticity.base import Plasticity, PlasticityConfig, PlasticityOutput
from spark.nn.components.plasticity.modulated import ModulatedPlasticity, ModulatedPlasticityConfig
from spark.nn.components.plasticity.zenke_rule import (
    ZenkeRule, ZenkeRuleConfig,
    ModulatedZenkeRule, ModulatedZenkeRuleConfig,
)
from spark.nn.components.plasticity.hebbian_rule import (
    HebbianRule, HebbianRuleConfig,
    OjaRule, OjaRuleConfig,
    ModulatedHebbianRule, ModulatedHebbianRuleConfig,
    ModulatedOjaRule, ModulatedOjaRuleConfig,
)
from spark.nn.components.plasticity.quadruplet_rule import (
    QuadrupletRule, QuadrupletRuleConfig,
    ModulatedQuadrupletRule, ModulatedQuadrupletRuleConfig,
)

__all__ = [
    'Plasticity', 'PlasticityConfig', 'PlasticityOutput',
    'ModulatedPlasticity', 'ModulatedPlasticityConfig',
    'ZenkeRule', 'ZenkeRuleConfig',
    'HebbianRule', 'HebbianRuleConfig',
    'OjaRule', 'OjaRuleConfig',
    'QuadrupletRule', 'QuadrupletRuleConfig',
    'ModulatedQuadrupletRule', 'ModulatedQuadrupletRuleConfig',
    'ModulatedHebbianRule', 'ModulatedHebbianRuleConfig',
    'ModulatedOjaRule', 'ModulatedOjaRuleConfig',
    'ModulatedZenkeRule', 'ModulatedZenkeRuleConfig',
]
