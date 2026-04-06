#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import os
import abc
import jax
import copy
import inspect
import flax.nnx as nnx
import jax.numpy as jnp
import dataclasses as dc

import spark.core.utils as utils
import spark.core.signature_parser as sig_parser
from spark.core.variables import Variable
from spark.core.registry import REGISTRY
from spark.core.config import SparkConfig
from spark.core.module import SparkModule, SparkMeta
from spark.core.specs import PortSpecs, PortMap, ModuleSpecs
from spark.core.payloads import SparkPayload, SpikeArray
from spark.core.decorators import spark_property
from spark.core.typing import is_object_of_type
from spark.core.config_validation import TypeValidator, PositiveValidator

# TODO: Currently inputs and effects require the ports to be defined inside a list. 
# This is not ideal from the point of view of user, it makes everything slightly more annoying that it needs.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ControllerMeta(SparkMeta):
    """
        Controller metaclass.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ControllerConfig(SparkConfig):
    modules_specs: list[ModuleSpecs] = dc.field(
        metadata = {
            'validators': [
            ],
            'description': 'Controller modules.',
        })
    seed: int = dc.field(
        default_factory=lambda: int.from_bytes(os.urandom(4), 'little'), 
        metadata={
            'validators': [
                TypeValidator,
            ], 
            'description': 'Seed for internal random processes.',
        })
    dt: float = dc.field(
        default=1.0, 
        metadata={
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Deltatime integration constant.',
        })
    
    # TODO: Manual override to synchronize all time integration constants across the controller.
    # This solution is probably good enough but it is not clear that will not clash with other user intentions.
    # A similar situation is present in Neuron.__post_init__
    def __post_init__(self,) -> None:
        super().__post_init__()
        # Synchronize dt's. NOTE: Skip validation, otherwise will fall into an infinite loop.
        self.merge(partial={'_s_dt':self.dt}, skip_validation=True)

ConfigT = tp.TypeVar("ConfigT", bound=ControllerConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Controller(nnx.Module, abc.ABC, tp.Generic[ConfigT], metaclass=ControllerMeta):
    """
        Controller model.

        A controller is a pipeline object used to represent and coordinate a collection of Spark modules.
    """
    config: ConfigT
    default_config: type[ConfigT]

    # NOTE: This is a workaround to require all childs of SparkModule to define a, 
    # default_config while at the same time allow for a lazy definition of the property. 
    def __init_subclass__(cls, **kwargs) -> None:
        from spark.core.config import SparkConfig
        super().__init_subclass__(**kwargs)
        # Special cases and abstract classes dont need config yet they are SparkModules ¯\_(ツ)_/¯
        is_abc = inspect.isabstract(cls) and len(getattr(cls, '__abstractmethods__', set())) == 0
        if is_abc:
            return
        # Check if defines config
        resolved_hints = tp.get_type_hints(cls)
        config_type = resolved_hints.get('config', None)
        if not config_type or config_type == ConfigT or not issubclass(config_type, SparkConfig):
            raise AttributeError('SparkModules must define a valid config: type[SparkConfig] attribute.')
        cls.default_config = tp.cast(type[ConfigT], config_type)

    def __init__(self, config: ConfigT | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__()
        # Override config if provided
        if config is None:
            self.config = self.default_config(**kwargs)
        else:
            self.config = copy.deepcopy(config)
            self.config.merge(partial=kwargs)
        # Rng
        seed = getattr(self.config, 'seed', None)
        if seed is not None:
            self._seed = seed
            # Random engine key.
            self.rng = Variable(jax.random.PRNGKey(self._seed))
        # Input validation
        assert is_object_of_type(self.config.modules_specs, list[ModuleSpecs]), 'Invalid modules list'
        # Get specs and perform basic validation
        input_specs, output_specs = self._validate_modules(self.config.modules_specs)
        self._controller_input_specs = input_specs
        self._controller_output_specs = output_specs
        # Flat output mapping
        self._contoller_output_map: list[tuple[str, str, str]] = []
        for output_name, output_spec in self._controller_output_specs.items():
            self._contoller_output_map.append(
                (output_name, output_spec['map'].origin, output_spec['map'].port)
            )
        # Get modules map
        self._modules_inputs_map = {spec.name: spec.inputs for spec in self.config.modules_specs}
        self._modules_effects_map = {spec.name: spec.effects for spec in self.config.modules_specs if spec.effects != {}}
        #self._modules_output_map = {spec.name: list(spec.module_cls._get_output_specs().keys()) for spec in self.config.modules_specs}
        self._modules_output_map = {}
        for spec in self.config.modules_specs:
            if issubclass(spec.module_cls, Controller):
                key_list = list(spec.module_cls._get_controller_output_specs(spec.config.modules_specs).keys())
            else:
                key_list = list(spec.module_cls._get_output_specs().keys())
            self._modules_output_map[spec.name] = key_list
        # Save original specs
        self._modules_specs = copy.deepcopy(self.config.modules_specs)
        self._modules_names = tuple([spec.name for spec in self.config.modules_specs])
        # Create modules.
        for spec in self.config.modules_specs:
            from spark.nn.controllers.neuron import Neuron
            if issubclass(spec.module_cls, Neuron):
                setattr(self, spec.name, REGISTRY.NEURONS.get(spec.module_cls.__name__).class_ref(config=spec.config))
            else:
                setattr(self, spec.name, REGISTRY.MODULES.get(spec.module_cls.__name__).class_ref(config=spec.config))



    @classmethod
    def get_properties(cls,) -> tuple[str, ...]:
        """
            Returns all the attributes names wrapped by the spark_property wrapper.
        """
        return tuple(
            [name  for name, attr in inspect.getmembers(cls) if isinstance(attr, spark_property)]
        )



    @classmethod
    def _get_controller_property_specs(cls,) -> dict[str, PortSpecs]:
        """
            Dynamically constructs the property specs of a controller.
        """
        return sig_parser.get_property_specs(cls)



    @classmethod
    def _get_controller_input_specs(cls, modules_specs: list[ModuleSpecs]) -> dict[str, PortSpecs]:
        """
            Dynamically constructs the input specs of a controller from a list of ModuleSpecs.
            Module level validation is applied to ensure that input ports exist.
        """
        controller_input_specs = {}
        # Iterate over every module.
        for module_specs in modules_specs:
            if issubclass(module_specs.module_cls, Controller):
                module_input_specs = module_specs.module_cls._get_controller_input_specs(module_specs.config.modules_specs)
            else:
                module_input_specs = module_specs.module_cls._get_input_specs()
            # Validate that input names are well defined.
            module_input_ports = set(module_input_specs.keys())
            module_defined_input_ports = set(module_specs.inputs.keys())
            # Missing ports
            missing_ports = module_input_ports.difference(module_defined_input_ports)
            if len(missing_ports) > 0:
                raise ValueError(
                    f'Missing input port names "{missing_ports}" in module "{module_specs.name}" specification. '
                    f'Module "{module_specs.name}" requires the following input ports: {module_input_ports}'
                )     
            # Invalid ports
            invalid_ports = module_defined_input_ports.difference(module_input_ports)
            if len(invalid_ports) > 0:
                raise ValueError(
                    f'Invalid input port names "{invalid_ports}" in module "{module_specs.name}". '
                    f'Module "{module_specs.name}" only defines the following input ports: {module_input_ports}'
                )  
            # Get __call__ ports. 
            # TODO: Validation for multiple modules refering to the same __call__ input is missing.
            for input_name, port_spec_list in module_specs.inputs.items():
                for port_spec in port_spec_list:
                    if port_spec.origin == '__call__':
                        controller_input_specs[port_spec.port] = module_input_specs[input_name]
        return controller_input_specs



    @classmethod
    def _get_controller_output_specs(cls, modules_specs: list[ModuleSpecs]) -> dict[str, dict]:
        """
            Dynamically constructs the output specs of a controller from a list of ModuleSpecs.
            Module level validation is applied to ensure that defined output ports exist and do not overlap.
        """
        controller_output_specs = {}
        # Iterate over every module.
        for module_specs in modules_specs:
            # Iterate over every defined output. 
            module_output_ports = None
            module_property_ports = None
            for out_name, port_name in module_specs.outputs.items():
                # Check that the output name is not already registered.
                if out_name in controller_output_specs:
                    other = controller_output_specs[out_name]['map'].origin
                    raise ValueError(
                        f'Repeated output names are not supported. '
                        f'Modules "{other}" and "{module_specs.name}"define the same output variable name: "{out_name}".'
                    )
                # Check that the modules defines a named port port_name
                if module_output_ports is None:
                    if issubclass(module_specs.module_cls, Controller):
                        module_output_ports = {k:v['spec'] for k, v in module_specs.module_cls._get_controller_output_specs(module_specs.config.modules_specs).items()}
                        module_property_ports = module_specs.module_cls._get_controller_property_specs()
                    else:
                        module_output_ports = module_specs.module_cls._get_output_specs()
                        module_property_ports = module_specs.module_cls._get_property_specs()
                if not port_name in module_output_ports and not port_name in module_property_ports:
                    raise ValueError(
                        f'Invalid output port name "{port_name}" in module "{module_specs.name}". '
                        f'Module "{module_specs.name}" only defines the following ports: '
                        f'\nOutput ports: {list(module_output_ports.keys())}. '
                        f'\nProperty ports: {list(module_property_ports.keys())}. '
                    )
                # Register output
                is_property = True if port_name in module_property_ports else False 
                module_out_spec = module_property_ports[port_name] if is_property else module_output_ports[port_name]
                controller_output_specs[out_name] = {
                    'map': PortMap(origin=module_specs.name, port=port_name, is_property=is_property),
                    'spec': module_out_spec
                }
        return controller_output_specs



    def recurrent_contract(
            self, 
        ) -> tuple[dict[str, tp.Any], dict[str, PortSpecs]]:
        """
            Returns the expected specs for the outputs and properties of the module.

            This function is a binding contract that allows the modules to accept self connections.
        """
        raise RuntimeError(
            f'Recurrent contract not implemented.'
        )



    @classmethod
    def has_recurrent_contract(cls) -> bool:
        """
            Returns True if the modules defines a recurrent contract, False otherwise.
        """
        return cls.recurrent_contract is not Controller.recurrent_contract



    @classmethod 
    def get_config_spec(cls) -> type[ConfigT]:
        """
            Returns the default configuration class associated with this module.
        """
        type_hints = tp.get_type_hints(cls)
        return type_hints['config']



    # TODO: Validate port shapes (broadcastable). I think this can only be done after all the modules are built.
    @classmethod
    def _validate_modules(cls, modules_specs: list[ModuleSpecs]) -> tuple[dict[str, PortSpecs], dict[str, dict]]:
        
        # Get controller specs for validation.
        # NOTE: The following two methods perform port name validations and must be called in order to fully validate the model. 
        controller_input_specs = cls._get_controller_input_specs(modules_specs)
        controller_output_specs = cls._get_controller_output_specs(modules_specs)
        controller_property_specs = cls._get_controller_property_specs()

        # Gather modules specs.
        modules_origins_specs = {
            '__call__': controller_input_specs,
            '__self__': controller_property_specs,
        }
        for spec in modules_specs:
            if issubclass(spec.module_cls, Controller):
                output_specs = {k:v['spec'] for k, v in spec.module_cls._get_controller_output_specs(spec.config.modules_specs).items()}
                property_specs = spec.module_cls._get_controller_property_specs()
            else:
                output_specs = spec.module_cls._get_output_specs()
                property_specs = spec.module_cls._get_property_specs()
            name_intersection = list(set(output_specs.keys()).intersection(set(property_specs.keys())))
            if len(name_intersection) > 0:
                raise ValueError(
                    f'Name overlap between one our more output variables and properties. '
                    f'Module "{spec.name}" defines the names "{name_intersection}" for output '
                    f'variables and a properties, which is not supported.'
                )
            modules_origins_specs[spec.name] = {
                **output_specs,
                **property_specs,
            }
                
        # Validate module port expected specs.
        for spec in modules_specs:  
            # Call ports
            if issubclass(spec.module_cls, Controller):
                module_input_specs = {
                    **spec.module_cls._get_controller_input_specs(spec.config.modules_specs),
                    **spec.module_cls._get_controller_property_specs(),
                }
            else:
                module_input_specs = {
                    **spec.module_cls._get_input_specs(),
                    **spec.module_cls._get_property_specs(),
                }
            for input_name, port_spec_list in spec.inputs.items():
                # Get module port specs
                expected_port_specs = module_input_specs[input_name]
                for port_map in port_spec_list:
                    # Validate all input ports are well connected.
                    if not port_map.port in modules_origins_specs[port_map.origin]:
                        raise ValueError(
                            f'Output port "{port_map.port}" in module "{port_map.origin}" is being '
                            f'requested by input port "{input_name}" of module "{spec.name}" but '
                            f'module "{port_map.origin}" only defines the following output ports: '
                            f'{list(modules_origins_specs[port_map.origin].keys())}'
                        )    
                    # Validate port types
                    other_port_specs = modules_origins_specs[port_map.origin][port_map.port]
                    if not other_port_specs.payload_type == expected_port_specs.payload_type:
                        raise ValueError(
                            f'Payload type "{other_port_specs.payload_type}" at output port "{port_map.port}" '
                            f'in module "{port_map.origin}" does not match the expected payload type "{expected_port_specs.payload_type}" '
                            f'from input port "{input_name}" of module "{spec.name}".'
                        )    
            # Effects (property ports)
            if issubclass(spec.module_cls, Controller):
                module_property_specs = spec.module_cls._get_controller_property_specs()
            else:
                module_property_specs = spec.module_cls._get_property_specs()
            for property_name, port_spec_list in spec.effects.items():
                # Get module port specs
                expected_port_specs = module_property_specs[property_name]
                for port_map in port_spec_list:
                    # Validate all input ports are well connected.
                    if not port_map.port in modules_origins_specs[port_map.origin]:
                        raise ValueError(
                            f'Output port "{port_map.port}" in module "{port_map.origin}" is being '
                            f'requested by property port "{property_name}" of module "{spec.name}" but '
                            f'module "{port_map.origin}" only defines the following output ports: '
                            f'{list(modules_origins_specs[port_map.origin].keys())}'
                        )    
                    # Validate port types
                    other_port_specs = modules_origins_specs[port_map.origin][port_map.port]
                    if not other_port_specs.payload_type == expected_port_specs.payload_type:
                        raise ValueError(
                            f'Payload type "{other_port_specs.payload_type}" at output port "{port_map.port}" '
                            f'in module "{port_map.origin}" does not match the expected payload type "{expected_port_specs.payload_type}" '
                            f'from property port "{property_name}" of module "{spec.name}".'
                        )    

        return controller_input_specs, controller_output_specs



    # TODO: Compare forward (from __call__ to outputs) and backward (from outputs to __call__) induced orders
    # I think the unfolding may be meaningful for the neuron controller since it does not rely on the cache system.
    @classmethod
    def _execution_order(
            cls, 
            modules_specs: list[ModuleSpecs],
            ignore_output_contracts: bool = False,
            skip_validation: bool = False,
        ) -> list[list[str]]:
        """
            Computes the order of module execution.
        """
        # Validate
        if not skip_validation:
            cls._validate_modules(modules_specs)
        # Gather initial inputs
        available_inputs = set(['__call__', '__self__'])
        # Gather modules
        remaining_modules = set([spec.name for spec in modules_specs])
        output_contracts = {spec.name: spec.module_cls.has_recurrent_contract() for spec in modules_specs}
        inputs_maps = {spec.name: spec.inputs for spec in modules_specs}
        # Compute execution order
        execution_order = []
        # Runtime error pass
        may_skip_error = True
        while len(remaining_modules) > 0:
            next_modules = []
            for name in remaining_modules:
                # Get module inputs
                module_required_inputs = set([pm for inputs_pm in inputs_maps[name].values() for pm in inputs_pm])
                non_instantiated_inputs = 0
                # Check that all inputs are available
                may_execute = True
                for port_map in module_required_inputs:
                    if not port_map.origin in available_inputs:
                        # Does input module defines an output policy?
                        if (not ignore_output_contracts) and output_contracts[port_map.origin]:
                            non_instantiated_inputs += 1
                            continue
                        # Input is not available nor can be obtained via contract, module cannot be executed yet.
                        may_execute = False
                        break
                # May execute?
                if may_execute:
                    # Skip module if all inputs are not initialized and we may skip the error.
                    if not (may_skip_error and non_instantiated_inputs == len(module_required_inputs)):
                        next_modules.append(name)
            # Progress check
            if len(next_modules) == 0:
                if may_skip_error:
                    # Give execution a pass, it found modules that would be better to initialize later if possible.
                    may_skip_error = False
                else:
                    # Model is invalid, some modules are not viable.
                    raise RuntimeError(
                        f'Cannot compute execution order. The following modules cannot be safely initialized: "{remaining_modules}".'
                    )
            # Reset skip
            if len(next_modules) > 0:
                may_skip_error = True    
            # Update execution order
            execution_order.append(next_modules)
            for name in next_modules:
                available_inputs.add(name)
                remaining_modules.remove(name)
        return execution_order



    def _instantiate_modules(
            self, 
            input_specs: dict[str, PortSpecs], 
            execution_order: list[list[str]]
        ) -> tuple[utils.TwoKeyDict, utils.TwoKeyDict]:
        # Compute controller property specs. Shapes for properties should be known by this point.
        property_specs = self._get_controller_property_specs()
        for property_name in property_specs.keys():
            property_specs[property_name].shape = getattr(self, property_name).shape
            property_specs[property_name].dtype = getattr(self, property_name).dtype
        # Set initial specs
        modules_output_specs = utils.TwoKeyDict()
        modules_output_specs['__call__'] = input_specs
        modules_property_specs = utils.TwoKeyDict()
        modules_property_specs['__self__'] = property_specs
        # Build modules. 
        for group in execution_order:
            for module_name in group:
                # Skip __call__
                if module_name == '__call__':
                    continue
                # Collect module specs and construct a mock input
                mock_input = {}
                validate_async = True
                for port_name, port_map_list in self._modules_inputs_map[module_name].items():
                    portspecs_list = []
                    for port_map in port_map_list:
                        if port_map.is_property and port_map.origin in modules_property_specs:
                            # Module was already built grab, get the spec.
                            spec = modules_property_specs[port_map.origin, port_map.port]
                            portspecs_list.append(spec)
                        elif port_map.origin in modules_output_specs:
                            # Module was already built grab, get the spec.
                            spec = modules_output_specs[port_map.origin, port_map.port]
                            portspecs_list.append(spec)
                        elif port_map.origin in group:
                            # Module is not built yet, it must define a recurrent spec to be part of a cyclic dependency.
                            origin_module: SparkModule = getattr(self, port_map.origin)
                            output_spec, property_spec = origin_module.recurrent_contract()
                            spec = property_spec[port_map.port] if port_map.is_property else output_spec[port_map.port]
                            portspecs_list.append(spec)
                            # Turn off async validation.
                            validate_async = False
                        else:
                            # Something weird happend. The constructor is trying to get something from a module that should have been called later.
                            raise RuntimeError(
                                f'Trying to get port from module "{port_map.origin}" for "{module_name}"... '
                                f'418 I\'m a teapot.'
                            )
                    mock_port_spec = PortSpecs.from_portspecs_list(portspecs_list, validate_async=validate_async)
                    mock_input[port_name] = mock_port_spec._create_mock_input()
                # Initialize module
                module: SparkModule | Controller = getattr(self, module_name)
                abc_output = module(**mock_input)
                # Add output specs to list
                if isinstance(module, Controller):
                    # TODO: Several comparisons like this one are required throught the entire code to deal with Neuron controllers
                    # A good equivalent of get_X_specs from the instance is missing.
                    modules_output_specs[module_name] = {k:v['spec'] for k,v in module._get_controller_output_specs(module.config.modules_specs).items()}
                    for output_name in modules_output_specs[module_name].keys():
                        modules_output_specs[module_name][output_name].shape = abc_output[output_name].shape
                        modules_output_specs[module_name][output_name].dtype = abc_output[output_name].dtype
                        if issubclass(modules_output_specs[module_name][output_name].payload_type, SpikeArray):
                            modules_output_specs[module_name][output_name].inhibition_mask = abc_output[output_name].inhibition_mask
                            modules_output_specs[module_name][output_name].async_spikes = abc_output[output_name].async_spikes
                    modules_property_specs[module_name] = module._get_controller_property_specs()
                    for property_name in modules_property_specs[module_name].keys():
                        modules_property_specs[module_name][property_name].shape = getattr(module, property_name).shape
                        modules_property_specs[module_name][property_name].dtype = getattr(module, property_name).dtype
                else:
                    modules_output_specs[module_name] = module.get_output_specs()
                    modules_property_specs[module_name] = module.get_property_specs()

        return modules_output_specs, modules_property_specs


    def _build(self, **abc_args: SparkPayload) -> None:
        """
            Triggers the shape inference and parameter initialization cascade.
        """
        # Bind arguments to avoid parameter mixing.
        call_signature = inspect.signature(self.__call__)
        try:
            bound_args = call_signature.bind(**abc_args)
            bound_args.apply_defaults()
        except TypeError as error:
            raise TypeError(f'Error binding arguments for "{self}": {error}') from error

        # Update shapes on controller specs
        for key, value in abc_args.items():
            self._controller_input_specs[key].shape = value.shape
        # Build model.
        self.build(self._controller_input_specs)
        self.__built__ = True
        # Construct mock input to compute the module output shapes.
        from spark.core.payloads import ValueSparkPayload
        mock_input = {}
        for key, value in self._controller_input_specs.items():
            mock_input[key] = value._create_mock_input()

        # TODO: See _build in spark.Module. 
        abc_output = self.__call__(**mock_input)
        self.reset()

        # Update shapes on controller specs
        for key, value in abc_output.items():
            self._controller_output_specs[key]['spec'].shape = value.shape



    #@abc.abstractmethod
    def build(self, input_specs: dict[str, PortSpecs]):
        self._call_build(input_specs)



    def get_controller_inputs(self,) -> tuple[str]:
        """
            Returns the names of the controller's input variables
        """
        return tuple(self._controller_input_specs.keys())



    def get_controller_outputs(self,) -> tuple[str]:
        """
            Returns the names of the controller's output variables
        """
        return tuple(self._controller_output_specs.keys())



    def refresh_seeds(self, seed: int | None = None) -> None:
        """
            Utility method to recompute all seed variables within the SparkConfig.
            Useful when creating several populations from the same config.

            NOTE: This method has no effect after the model has been built.
        """
        self.config.with_new_seeds()



    def reset(self) -> None:
        """
            Resets all the modules to its initial state.
        """
        for name in self._modules_names:
            module: SparkModule = getattr(self, name)
            module.reset()



    def _concatenate_payloads(self, args: list[SparkPayload]) -> SparkPayload:
        if len(args) == 1:
            return args[0]
        payload_type = type(args[0])
        if issubclass(payload_type, SpikeArray):
            return payload_type._from_encoding(jnp.concatenate([x._encoding.reshape(-1) for x in args]))
        else:
            return payload_type(jnp.concatenate([x.value.reshape(-1) for x in args]))



    @abc.abstractmethod
    def __call__(self, **inputs: SparkPayload) -> dict[str, SparkPayload]:
        """
            Update controller's states.
        """
        pass



    @abc.abstractmethod
    def read_state(self, port_list: list[PortMap]) -> dict:
        """
            Utility function to read internal controller's variables.
        """
        pass



    def _parse_tree_structure(self, current_depth: int = 0, name: str | None = None) -> str:
        """
            Parses the tree with to produce a string with the appropiate format for the ascii_tree method.
        """
        if name:
            rep = current_depth * ' ' + f'{name} ({self.__class__.__name__})\n'
        else:
            rep = current_depth * ' ' + f'{self.__class__.__name__}\n'
        for module_name in self._modules_names:
            module: SparkModule = getattr(self, module_name)
            rep += module._parse_tree_structure(current_depth+1, name=module_name)
        return rep



    def get_rng_keys(self, num_keys: int) -> jax.Array | list[jax.Array]:
        """
            Generates a new collection of random keys for the JAX's random engine.
        """
        if not hasattr(self, 'rng'):
            raise RuntimeError(
                f"Module '{self.__class__.__name__}' does not have a random number generator initialized. "
                "Ensure its configuration defines a 'seed' attribute."
            )
        self.rng.value, *keys = jax.random.split(self.rng.value, num_keys+1)
        if num_keys == 1:
            return keys[0]
        return keys

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################