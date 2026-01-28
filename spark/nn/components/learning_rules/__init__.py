from spark.nn.components.learning_rules.base import LearningRule, LearningRuleConfig, LearningRuleOutput
from spark.nn.components.learning_rules.zenke_rule import ZenkeRule, ZenkeRuleConfig
from spark.nn.components.learning_rules.hebbian_rule import (
    HebbianRule, HebbianRuleConfig,
    OjaRule, OjaRuleConfig
)
from spark.nn.components.learning_rules.quadruplet_rule import (
    QuadrupletRule, QuadrupletRuleConfig,
    QuadrupletRuleTensor, QuadrupletRuleTensorConfig
)
from spark.nn.components.learning_rules.three_factor_rule import (
    ThreeFactorHebbianRule, ThreeFactorHebbianRuleConfig,
)

__all__ = [
    'LearningRule', 'LearningRuleConfig', 'LearningRuleOutput',
    'ZenkeRule', 'ZenkeRuleConfig',
    'HebbianRule', 'HebbianRuleConfig',
    'OjaRule', 'OjaRuleConfig',
    'QuadrupletRule', 'QuadrupletRuleConfig',
    'QuadrupletRuleTensor', 'QuadrupletRuleTensorConfig',
    'ThreeFactorHebbianRule', 'ThreeFactorHebbianRuleConfig',
]