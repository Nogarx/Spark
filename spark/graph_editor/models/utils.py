#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import copy
import typing as tp
from spark.core.config import SparkConfig
from spark.nn.controllers.base import ControllerConfig

# TODO: Flattening/Unflattening of Config classes is only required when creating graph nodes, but may be useful 
# to abstract this code and made it available through the entire ecosystem.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def flattify_controller_config(config: ControllerConfig) -> SparkConfig:
    """
        Generate a Config subclass programmatically, recursively building nested Configs from a Controller Config.
    """

    # Shallow copy
    config = copy.deepcopy(config)
    # Cls namespace
    cls_name = f'{config.__class__.__name__}_FLAT'
    ns_annotations: dict[str, tp.Any] = {}
    namespace: dict[str, tp.Any] = {}
    # Iterate over config
    for name, field, value in config:
        # Map simple attributes directly
        if name != 'modules_specs':
            namespace[name] = value
            ns_annotations[name] = field.type
        else:
            # Extract internal configs from the module spec
            for module_spec in value:
                namespace[module_spec.name] = module_spec.config
                ns_annotations[module_spec.name] = type(module_spec.config)
    # Create class dynamically
    namespace['__annotations__'] = ns_annotations
    cls_flat = type(cls_name, (SparkConfig,), namespace)
    # Instantiate class
    flat_config = cls_flat()
    # Copy metadata
    flat_config.__metadata__ = config.__metadata__
    flat_config.__graph_editor_metadata__ = config.__graph_editor_metadata__
    return flat_config

def unflattify_controller_config(cls: type[ControllerConfig], flat_config: SparkConfig) -> ControllerConfig:
    """
        Generate a ControllerConfig subclass programmatically, recursively building the modules_specs list
        from a another Config spec and a ControllerConfig template.
    """

    # Initialize a partial config
    unflatten_config = cls._create_partial()
    expected_modules_names = [spec.name for spec in getattr(unflatten_config, 'modules_specs')]
    # Set simple attributes of the controller
    simple_attribute_names = []
    for name, _, _ in unflatten_config:
        if name != 'modules_specs':
            # Update attributes with the values from the flat_config
            value = getattr(flat_config, name)
            setattr(unflatten_config, name, value)
            simple_attribute_names.append(name)
    # Get modules
    modules_names = []
    modules_specs_map = {}
    for name, _, value in flat_config:
        # Skip simple attributes
        if name in simple_attribute_names:
            continue
        # Add config to list
        if not isinstance(value, SparkConfig):
            raise TypeError(
                f'Expected "{name}" value to be of type "SparkConfig" but got "{type(value).__name__}".'
            )
        modules_names.append(name)
        modules_specs_map[name] = value
    # Validate that all expected modules are present
    expected_modules_names = set(expected_modules_names)
    modules_names = set(modules_names)
    missing_names = expected_modules_names.difference(modules_names)
    extra_names = modules_names.difference(expected_modules_names)
    if len(missing_names) > 0:
        raise KeyError(
            f'The following config attributes "{list(missing_names)}" are not defined in "flat_config".'
        )
    if len(extra_names) > 0:
        raise KeyError(
            f'The following config attributes "{list(extra_names)}" are not part of the definition of "{cls.__name__}".'
        )
    # Set the module_specs
    modules_specs_list = getattr(unflatten_config, 'modules_specs')
    for module_spec in modules_specs_list:
        module_spec.config = modules_specs_map[module_spec.name]
    setattr(unflatten_config, 'modules_specs', modules_specs_list)
    # Copy metadata
    unflatten_config.__metadata__ = flat_config.__metadata__
    unflatten_config.__graph_editor_metadata__ = flat_config.__graph_editor_metadata__
    return unflatten_config

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################