from spark.nn.components.learning_rules.base import LearningRule, LearningRuleOutput
from spark.nn.components.learning_rules.zenke_rule import ZenkeRule, ZenkeRuleConfig
from spark.nn.components.learning_rules.hebbian_rule import (
    HebbianRule, HebbianRuleConfig,
    OjaRule, OjaRuleConfig
)

__all__ = [
    'LearningRule', 'LearningRuleOutput',
    'ZenkeRule', 'ZenkeRuleConfig',
    'HebbianRule', 'HebbianRuleConfig',
    'OjaRule', 'OjaRuleConfig',
]