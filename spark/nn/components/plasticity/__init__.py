from spark.nn.components.plasticity.base import Plasticity, PlasticityConfig, PlasticityOutput
from spark.nn.components.plasticity.zenke_rule import ZenkeRule, ZenkeRuleConfig
from spark.nn.components.plasticity.hebbian_rule import (
    HebbianRule, HebbianRuleConfig,
    OjaRule, OjaRuleConfig
)
from spark.nn.components.plasticity.quadruplet_rule import (
    QuadrupletRule, QuadrupletRuleConfig,
    QuadrupletRuleTensor, QuadrupletRuleTensorConfig
)
from spark.nn.components.plasticity.three_factor_rule import (
    ThreeFactorHebbianRule, ThreeFactorHebbianRuleConfig,
)

__all__ = [
    'Plasticity', 'PlasticityConfig', 'PlasticityOutput',
    'ZenkeRule', 'ZenkeRuleConfig',
    'HebbianRule', 'HebbianRuleConfig',
    'OjaRule', 'OjaRuleConfig',
    'QuadrupletRule', 'QuadrupletRuleConfig',
    'QuadrupletRuleTensor', 'QuadrupletRuleTensorConfig',
    'ThreeFactorHebbianRule', 'ThreeFactorHebbianRuleConfig',
]