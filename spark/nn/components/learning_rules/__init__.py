from spark.nn.components.learning_rules.base import LearningRule, LearningRuleOutput
from spark.nn.components.learning_rules.zenke_hebian_rule import HebbianRule, HebbianLearningConfig

__all__ = [
    'LearningRule', 'LearningRuleOutput',
    'HebbianRule', 'HebbianLearningConfig',
]