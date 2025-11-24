
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import numpy as np
import typing as tp
import spark
import dataclasses as dc
np.random.seed(42)
from spark.nn.initializers import Initializer, InitializerConfig, ConstantInitializer, ConstantInitializerConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def make_config(cls_name: str, annotations: dict[str, tp.Any], defaults: dict[str, tp.Any]) -> type:
    """
        Generate a Config subclass programmatically, recursively building nested
        Config subclasses when an annotation value is a mapping (dict).
    """

    # Shallow copy
    annotations = dict(annotations)  
    defaults = dict(defaults or {})

    ns_annotations: dict[str, tp.Any] = {}
    namespace: dict[str, tp.Any] = {}

    for attr, ann in annotations.items():
        if isinstance(ann, tp.Mapping):
            # Nested class name
            nested_cls_name = f'{cls_name}_NC_{attr.capitalize()}'
            # Construct nested class
            nested_spec = dict(ann)
            nested_annotations = dict(nested_spec)
            nested_defaults = defaults.get(attr, {}) if isinstance(defaults.get(attr, {}), tp.Mapping) else {}
            # Build nested class recursively
            nested_cls = make_config(nested_cls_name, nested_annotations, nested_defaults)
            # Annotate the attribute as the nested class type
            ns_annotations[attr] = nested_cls
            # Assign the nested defaults to the class
            if attr in defaults and isinstance(defaults[attr], tp.Mapping):
                namespace[attr] = nested_cls
        else:
            # Regular attribute
            ns_annotations[attr] = ann
            # Assign default if present
            if attr in defaults and not isinstance(defaults[attr], tp.Mapping):
                namespace[attr] = defaults[attr]
                
    # Repalce default cls_tags with type/instance
    for attr, value in defaults.items():
        if value == '__SELF_CLS__':
            namespace[attr] = ns_annotations[attr]
        elif value == '__SELF_CLS_INIT__':
            namespace[attr] = ns_annotations[attr]()
            
    # Create class dynamically
    namespace['__annotations__'] = ns_annotations
    return type(cls_name, (spark.nn.BaseConfig,), namespace)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

data_test_success = [
    (
        'Simple_1',
        {'foo': int, 'bar': float, 'baz': bool},
        {'foo': 10, 'bar': 15.0},
        {},
        {'foo': 10, 'bar': 15.0, 'baz': False},
    ),
    (
        'Simple_2',
        {'foo': int | Initializer, 'bar': float, 'baz': bool},
        {'foo': 10, 'bar': 15.0},
        {},
        {'foo': 10, 'bar': 15.0, 'baz': False},
    ),
    (
        'Simple_3',
        {'foo': int | InitializerConfig, 'bar': float, 'baz': bool},
        {'foo': 10, 'bar': 15.0},
        {},
        {'foo': 10, 'bar': 15.0, 'baz': False},
    ),
    (
        'Nested_1',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool}},
        {'foo': 10, 'bar': 15.0, 'baz': True},
        {},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 0, 'bar': 0, 'baz': False}},
    ),
    (
        'Nested_2',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool}},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': '__SELF_CLS__'},
        {},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 0, 'bar': 0, 'baz': False}},
    ),
    (
        'Nested_3',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool}}},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': '__SELF_CLS__'}},
        {},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 0, 'bar': 0, 'baz': False}}},
    ),
    (
        'Simple_Kwargs_1',
        {'foo': int, 'bar': float, 'baz': bool},
        {'foo': 10, 'bar': 15.0},
        {'foo': 1},
        {'foo': 1, 'bar': 15.0, 'baz': False},
    ),
    (
        'Simple_Kwargs_2',
        {'foo': int, 'bar': float, 'baz': bool},
        {'foo': 10, 'bar': 15.0},
        {'foo': 1, 'bar': 1.0, 'baz': True},
        {'foo': 1, 'bar': 1.0, 'baz': True},
    ),
    (
        'Simple_Kwargs_3',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool}}},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True}}},
        {'foo': 1},
        {'foo': 1, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True}}},
    ),
    (
        'Simple_Kwargs_4',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool}}},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True}}},
        {'config__foo': 1},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 1, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True}}},
    ),
    (
        'Simple_Kwargs_5',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool}}},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True}}},
        {'config__config__foo': 1},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 1, 'bar': 15.0, 'baz': True}}},
    ),
    (
        'Simple_Kwargs_6',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool}}},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True}}},
        {'_s_foo': 1},
        {'foo': 1, 'bar': 15.0, 'baz': True, 'config': {'foo': 1, 'bar': 15.0, 'baz': True, 'config': {'foo': 1, 'bar': 15.0, 'baz': True}}},
    ),
    (
        'ConfigInit_Kwargs_1',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool}}},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': '__SELF_CLS_INIT__'}},
        {'foo': 1},
        {'foo': 1, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 0, 'bar': 0.0, 'baz': False}}},
    ),
    (
        'ConfigInit_Kwargs_2',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool}}},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': '__SELF_CLS_INIT__'}},
        {'config__foo': 1},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 1, 'bar': 15.0, 'baz': True, 'config': {'foo': 0, 'bar': 0.0, 'baz': False}}},
    ),
    (
        'ConfigInit_Kwargs_3',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool}}},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': '__SELF_CLS_INIT__'}},
        {'config__config__foo': 1},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 1, 'bar': 0.0, 'baz': False}}},
    ),
    (
        'ConfigInit_Kwargs_4',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int, 'bar': float, 'baz': bool}}},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'foo': 10, 'bar': 15.0, 'baz': True, 'config': '__SELF_CLS_INIT__'}},
        {'_s_foo': 1},
        {'foo': 1, 'bar': 15.0, 'baz': True, 'config': {'foo': 1, 'bar': 15.0, 'baz': True, 'config': {'foo': 1, 'bar': 0.0, 'baz': False}}},
    ),
    (
        'Init_1',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar': ConstantInitializer},
        {},
        {'foo': 10, 'bar': ConstantInitializerConfig()},
    ),
    (
        'Init_2',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar': ConstantInitializer(scale=2, max_value=4)},
        {},
        {'foo': 10, 'bar': ConstantInitializerConfig(scale=2, max_value=4)},
    ),
    (
        'Init_3',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar': ConstantInitializerConfig},
        {},
        {'foo': 10, 'bar': ConstantInitializerConfig()},
    ),
    (
        'Init_4',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar': ConstantInitializerConfig(scale=2, max_value=4)},
        {},
        {'foo': 10, 'bar': ConstantInitializerConfig(scale=2, max_value=4)},
    ),
    (
        'Init_5',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar': dc.field(default=ConstantInitializer)},
        {},
        {'foo': 10, 'bar': ConstantInitializerConfig()},
    ),
    (
        'Init_6',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar': dc.field(default_factory=ConstantInitializer)},
        {},
        {'foo': 10, 'bar': ConstantInitializerConfig()},
    ),
    (
        'Init_7',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar': dc.field(default=ConstantInitializerConfig)},
        {},
        {'foo': 10, 'bar': ConstantInitializerConfig()},
    ),
    (
        'Init_8',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar': dc.field(default_factory=ConstantInitializerConfig)},
        {},
        {'foo': 10, 'bar': ConstantInitializerConfig()},
    ),
    (
        'Init_9',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar':  dc.field(default=ConstantInitializer(scale=2, max_value=4))},
        {},
        {'foo': 10, 'bar': ConstantInitializerConfig(scale=2, max_value=4)},
    ),
    (
        'Init_10',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar': dc.field(default=ConstantInitializerConfig(scale=2, max_value=4))},
        {},
        {'foo': 10, 'bar': ConstantInitializerConfig(scale=2, max_value=4)},
    )
]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.mark.parametrize('name, annotations, defaults, init_kwargs, expected', data_test_success)
def test_spark_config_success(name, annotations, defaults, init_kwargs, expected) -> None:
    """
        Common valid spark config initializations.
    """
    # Create class dynamically
    DynamicConfig = make_config(name, annotations, defaults)
    # Instantiate class with custom init kwargs
    c = DynamicConfig(**init_kwargs)
    # Compare class content with expected dictionary
    assert c.get_kwargs() == expected
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

data_test_failure = [
    (
        'Fail_1',
        {'foo': int | Initializer, 'bar': float, 'baz': bool},
        {'bar': 15.0},
        {},
        {},
    ),
    (
        'Fail_2',
        {'foo': int | InitializerConfig, 'bar': float, 'baz': bool},
        {'bar': 15.0},
        {},
        {},
    ),
    (
        'Fail_3',
        {'foo': int, 'bar': float, 'baz': bool, 'config': {'foo': int | Initializer, 'bar': float, 'baz': bool}},
        {'foo': 10, 'bar': 15.0, 'baz': True, 'config': {'bar': 15.0, 'baz': True}},
        {},
        {},
    ),
    (
        'Init_1',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar':  dc.field(default_factory=ConstantInitializer(scale=2, max_value=4))},
        {},
        {},
    ),
    (
        'Init_2',
        {'foo': int, 'bar': Initializer},
        {'foo': 10, 'bar': dc.field(default_factory=ConstantInitializerConfig(scale=2, max_value=4))},
        {},
        {},
    ),
]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.mark.parametrize('name, annotations, defaults, init_kwargs, expected', data_test_failure)
def test_spark_config_failure(name, annotations, defaults, init_kwargs, expected) -> None:
    """
        Common invalid spark config initializations.
    """
    # Test initialization fails
    fail = False
    try:
        # Create class dynamically
        DynamicConfig = make_config(name, annotations, defaults)
        # Instantiate class with custom init kwargs
        c = DynamicConfig(**init_kwargs)
    except:
        fail = True
    assert fail

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################