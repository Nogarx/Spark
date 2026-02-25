#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import os
import abc
import jax
import jax.numpy as jnp
import typing as tp
import dataclasses as dc
from math import prod
import spark.core.utils as utils
from spark.core.module import SparkModule, SparkMeta
from spark.core.specs import PortSpecs, PortMap, ModuleSpecs
from spark.core.payloads import SparkPayload, SpikeArray
from spark.core.registry import register_config, REGISTRY
from spark.core.config import BaseSparkConfig
from spark.core.flax_imports import data
from spark.core.config_validation import TypeValidator

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ControllerMeta(SparkMeta):
    """
        Controller metaclass.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: We need to alliviate the need to define input/output maps. Maybe build them at build time with the modules_map plus in/out info?
@register_config
class ControllerConfig(BaseSparkConfig):
    """
        Configuration class for Controller's.
    """
    input_map: dict[str, PortSpecs] = dc.field(
        metadata = {
            'description': 'Input map configuration.',
        })
    output_map: dict[str, dict] = dc.field(
        metadata = {
            'description': 'Output map configuration.',
        })
    modules_map: dict[str, ModuleSpecs] = dc.field(
        metadata = {
            'description': 'Modules map configuration.',
        })
    seed: int = dc.field(
        default_factory=lambda: int.from_bytes(os.urandom(4), 'little'), 
        metadata={
            'validators': [
                TypeValidator,
            ], 
            'description': 'Seed for internal random processes.',
        })



    def _validate_maps(self,):
        """
            Basic validation of the configuration maps to ensure that controller can properly read the data.
        """
        # Input map validation
        if not isinstance(self.input_map, dict):
            raise TypeError(
                f'\"input_map\" must be a dictionary, but got \"{type(self.input_map).__name__}\".'
            )
        for key in self.input_map.keys():
            if not isinstance(key, str):
                raise TypeError(
                    f'All keys in \"input_map\" must be strings, but found key \"{key}\" of type {type(key).__name__}.'
                )
            if not isinstance(self.input_map[key], PortSpecs):
                raise TypeError(
                    f'All values in \"input_map\" must be PortSpecs, but found value \"{self.input_map[key]}\" '
                    f'of type {type(self.input_map[key]).__name__}.'
                )
        # Output map validation
        if not isinstance(self.output_map, dict):
            raise TypeError(
                f'\"output_map\" must be a dictionary, but got \"{type(self.output_map).__name__}\".'
            )
        for output_name, output_details in self.output_map.items():
            if not isinstance(output_name, str):
                raise TypeError(
                    f'All keys in \"output_map\" must be strings, but found key \"{output_name}\" of type {type(output_name).__name__}.'
                )
            if not isinstance(output_details, dict):
                raise TypeError(
                    f'All values in \"output_map\" must be dict, but found value \"{output_details}\" '
                    f'of type {type(output_details).__name__}.'
                )		
            _input = output_details.get('input', None)
            if not isinstance(_input, PortMap):
                raise TypeError(
                    f'Expected \"output_map[\"{output_name}\"][\"input\"]\" to be of type PortMap, but got \"{_input}\".'
                )
            spec = output_details.get('spec', None)
            if not isinstance(spec, PortSpecs):
                raise TypeError(
                    f'Expected \"output_map[\"{output_name}\"][\"spec\"]\" to be of type PortSpecs, but got \"{spec}\".'
                )
        # Modules map validation
        if not isinstance(self.modules_map, dict):
            raise TypeError(
                f'\"modules_map\" must be a dictionary, but got \"{type(self.modules_map).__name__}\".'
            )
        for key in self.modules_map.keys():
            if not isinstance(key, str):
                raise TypeError(
                    f'All keys in \"modules_map\" must be strings, but found key \"{key}\" of type \"{type(key).__name__}\".'
                )
            if not isinstance(self.modules_map[key], ModuleSpecs):
                raise TypeError(
                    f'All values in \"modules_map\" must be type \"ModuleSpecs\", but found value \"{self.modules_map[key]}\" '
                    f'of type \"{type(self.modules_map[key]).__name__}\".'
                )
        # Validate sources.
        valid_sources = set(self.input_map.keys()) | set(self.modules_map.keys())
        # Does the origin exist in the set of valid sources?
        for module_name, module_specs in self.modules_map.items():
            for port_name, connections in module_specs.inputs.items():
                for port_map in connections:
                    if port_map.origin not in valid_sources and port_map.origin != '__call__':
                        raise ValueError(
                            f'In module \"{module_name}\", connection for port \"{port_name}\" refers to an '
                            f'unknown origin: \"{port_map.origin}\". Valid sources are: {valid_sources}.'
                        )



    def validate(self, is_partial: bool = False, errors: list | None = None, current_path: list[str] = ['controller']) -> dict[str] | None:
        # Controller specific validation.
        if not is_partial:
            try:
                self._validate_maps()
            except Exception as e:
                if errors is not None:
                    errors.append((current_path[0], e))
                else:
                    raise
        # Standard config validation.
        for module_spec in self.modules_map.values():
            try:
                module_spec.config.validate(is_partial=is_partial, errors=errors, current_path=current_path+[module_spec.name])
            except Exception as e:
                raise ValueError(
                    f'Error on module \"{module_spec.name}\": {e}'
                )



    def _parse_tree_structure(self, current_depth: int, simplified: bool = False, header: str | None= None) -> str:
        """
            Parses the tree to produce a string with the appropiate format for the ascii_tree method.
        """
        level_header = f'{header}: ' if header else ''
        rep = current_depth * ' ' + f'{level_header}{self.__class__.__name__}\n'

        # Expand inputs specs
        rep += (current_depth + 1) * ' ' + f'Input Map:\n'
        for spec_name, spec in self.input_map.items():
            rep += (current_depth + 2) * ' ' + f'{spec_name} <- {spec}\n'
        # Expand outputs specs
        rep += (current_depth + 1) * ' ' + f'Output Map:\n'
        for spec_name, spec in self.output_map.items():
            rep += (current_depth + 2) * ' ' + f'{spec_name} <- {spec['input']} | {spec['spec']}\n'
        # Expand module specs
        rep += (current_depth + 1) * ' ' + f'Modules Map:\n'
        for spec_name, module_spec in self.modules_map.items():
            if not simplified:
                rep += (current_depth + 2) * ' ' + f'{spec_name}: {module_spec.module_cls.__name__}\n'
                rep += (current_depth + 3) * ' ' + f'Inputs:\n'
                for input_name, port_spec_list in module_spec.inputs.items():
                    rep += (current_depth + 4) * ' ' + f'{input_name}:\n'
                    for port in port_spec_list:
                        rep += (current_depth + 5) * ' ' + f'{port}\n'
                rep += module_spec.config._parse_tree_structure(current_depth+3, simplified=simplified)
            else:
                rep += module_spec.config._parse_tree_structure(current_depth+2, simplified=simplified)
        return rep



    def refresh_seeds(self):
        """
            Utility method to recompute all seed variables within the SparkConfig.
            Useful when creating several populations from the same config.
        """
        for spec in self.modules_map.values():
            spec.config = spec.config.with_new_seeds()



#-----------------------------------------------------------------------------------------------------------------------------------------------#



class Controller(SparkModule, metaclass=ControllerMeta):
    """
        Controller model.

        A controller is a pipeline object used to represent and coordinate a collection of Spark modules.
    """
    config: ControllerConfig

    # TODO: There is no need to have of all _modules_X_specs, we can access them directly from the module.
    # Typing annotations.
    _modules_list: list[str]
    _modules_input_map: dict[str, dict[str, list[PortMap]]]
    _modules_input_specs: dict[str, dict[str, PortSpecs]]
    _modules_output_specs: dict[str, dict[str, PortSpecs]]
    _modules_property_specs: dict[str, dict[str, PortSpecs]]

    def __init__(self, config: ControllerConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Build modules
        self._build_modules()



    def _build_modules(self,):
        """
            Construct the modules defined in the modules_map of the configuration class.
        """
        # Construc all modules.
        for module_name, module_specs in self.config.modules_map.items():
            setattr(self, module_name, REGISTRY.MODULES.get(module_specs.module_cls.__name__).class_ref(config=module_specs.config))
        self._modules_list = list(self.config.modules_map.keys())
        # Information flow map.
        self._modules_input_map = {}
        for module_name, module_specs in self.config.modules_map.items():
            self._modules_input_map[module_name] = module_specs.inputs



    # TODO: I think most of this validation can be moved to ControllerConfig
    def _validate_connections(self,):
        """
            Prevalidates that all modules are reachable.
        """
        # Collect all modules input/output specs.
        self._modules_input_specs = {}
        self._modules_output_specs = {}
        self._modules_property_specs = {}
        for module_name in self._modules_list:
            module: SparkModule = getattr(self, module_name)
            self._modules_input_specs[module_name] = module.get_input_specs()
            self._modules_output_specs[module_name] = module.get_output_specs()
            self._modules_property_specs[module_name] = module.get_property_specs()

        # Validate that each module can be connected.
        for module_name in self._modules_list:
            
            # Validate that every non-optional input port exists
            module_inputs: dict[str, list[PortMap]] = self.config.modules_map[module_name].inputs
            required_ports = set(self._modules_input_specs[module_name].keys()).difference(module_inputs.keys())
            for port_name in required_ports:
                class_error = REGISTRY.MODULES.get(self.config.modules_map[module_name].module_cls).class_ref
                raise ValueError(
                    f'Port \"{port_name}\" for module "{module_name}\" is not defined and '
                    f'is a mandatory input of \"{class_error.__name__}\".'
                )
            
            # Validate that no extra input port were defined
            extra_ports = set(module_inputs.keys()).difference(set(self._modules_input_specs[module_name].keys()))
            if len(extra_ports) > 0:
                raise ValueError(
                    f'Port \"{extra_ports[0]}\" for module \"{module_name}\" is not part of the specifications '
                    f'of \"{self.config.modules_map[module_name].module_cls}\".'
                )
            
            # Validate the integrity of the ports 
            for port_name, ports_list in module_inputs.items():
                
                # Check input module defines the output port
                for port_map in ports_list:
                    # Inspect __call__ map
                    if port_map.origin == '__call__':
                        if not port_map.port in self.config.input_map:
                            raise ValueError(
                                f'Input port \"{port_name}\" for module \"{module_name}\" request port \"{port_map.port}\" '
                                f'from \"{port_map.origin}\", but inputs_map does not define such port.'
                            )
                        continue
                    # Inspect module outputs/properties
                    origin_module: SparkModule = getattr(self, port_map.origin)
                    if port_map.is_property and not port_map.port in self._modules_property_specs[port_map.origin]:
                        raise ValueError(
                            f'Input port \"{port_name}\" for module \"{module_name}\" request spark property \"{port_map.port}\" from '
                            f'\"{port_map.origin}\", but \"{self.config.modules_map[port_map.origin].module_cls}\" does not define such spark property.'
                        )
                    elif not port_map.port in self._modules_output_specs[port_map.origin]:
                        raise ValueError(
                            f'Input port \"{port_name}\" for module \"{module_name}\" request port \"{port_map.port}\" from '
                            f'\"{port_map.origin}\", but \"{self.config.modules_map[port_map.origin].module_cls}\" does not define such port.'
                        )
                
                # Validate ports can be safely merged.
                if len(ports_list) > 1:
                    # Grab first port as proxy
                    if ports_list[0].is_property:
                        payload_type = self._modules_property_specs[ports_list[0].origin][ports_list[0].port].payload_type
                    else:
                        payload_type = self._modules_output_specs[ports_list[0].origin][ports_list[0].port].payload_type
                    # Test all other ports against proxy
                    for i in range(1,len(ports_list)):
                        if ports_list[i].is_property:
                            target_payload_type = self._modules_property_specs[ports_list[i].origin][ports_list[i].port].payload_type
                        else:
                            target_payload_type = self._modules_output_specs[ports_list[i].origin][ports_list[i].port].payload_type
                        if not payload_type == target_payload_type:
                            raise TypeError(
                                f'Input port \"{port_name}\" for module \"{module_name}\" request port \"{ports_list[i].port}\" '
                                f'from \"{ports_list[i].origin}\", but payload type is not compatible, expected \"{payload_type}\" '
                                f'and got \"{target_payload_type}\".'
                            ) 
                
                # Validate shapes
                expected_shape = 0
                if len(ports_list) > 1:
                    # Many-to-one input-output. Inputs need to be merged, default merged behaviour is to flat everything.
                    for port_map in ports_list:
                        if port_map.origin == '__call__':
                            expected_shape += prod(self.config.input_map[port_map.port].shape)
                        elif port_map.is_property:
                            expected_shape += prod(self._modules_property_specs[port_map.origin][port_map.port].shape)
                        else:
                            expected_shape += prod(self._modules_output_specs[port_map.origin][port_map.port].shape)
                else:
                    # One-to-one input-output.
                    if port_map.origin == '__call__':
                        expected_shape = self.config.input_map[port_map.port].shape
                    elif port_map.is_property:
                        expected_shape = self._modules_property_specs[port_map.origin][port_map.port].shape
                    else:
                        expected_shape = self._modules_output_specs[port_map.origin][port_map.port].shape
                # Normalize and compare shapes.
                expected_shape = utils.validate_shape(expected_shape)
                shape = self._modules_input_specs[module_name][port_name].shape
                if expected_shape != shape:
                    raise ValueError(
                        f'Input port \"{port_name}\" for module \"{module_name}\" expected input shape '
                        f'{expected_shape} but got shape \"{shape}\".'
                    ) 



    def execution_order(
            self, 
            ignore_output_contracts: bool = False,
        ) -> list[list[str]]:
        # Gather initial inputs
        available_inputs = set(['__call__'])
        # Gather modules
        remaining_modules = set(self._modules_list)
        # Compute execution order
        execution_order = []
        # Runtime error pass
        may_skip_error = True
        while len(remaining_modules) > 0:
            next_modules = []
            for module_name in remaining_modules:
                # Get module inputs
                module_required_inputs = set([pm for inputs_pm in self._modules_input_map[module_name].values() for pm in inputs_pm])
                non_instantiated_inputs = 0
                # Check that all inputs are available
                may_execute = True
                for port_map in module_required_inputs:
                    if not port_map.origin in available_inputs:
                        # Does input module defines an output policy?
                        if (not ignore_output_contracts) and getattr(self, port_map.origin).has_output_contract():
                            non_instantiated_inputs += 1
                            continue
                        # Input is not available nor can be obtained via contract, module cannot be executed yet.
                        may_execute = False
                        break
                # May execute?
                if may_execute:
                    # Skip module if all inputs are not initialized and we may skip the error.
                    if not (may_skip_error and non_instantiated_inputs == len(module_required_inputs)):
                        next_modules.append(module_name)
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
            for module_name in next_modules:
                available_inputs.add(module_name)
                remaining_modules.remove(module_name)
        return execution_order



    @abc.abstractmethod
    def build(self, input_specs: dict[str, PortSpecs]):
        pass



    # NOTE: We need to override _construct_input_specs since sig_parser.get_input_specs will lead to an incorrect 
    # signature because Controller can have an arbitrary number of inputs all under the key "inputs".
    def _construct_input_specs(self, abc_args: dict[str, dict[str, SparkPayload]]) -> dict[str, PortSpecs]:
        # Extract the real inputs from abc_args, they are all under the key 'inputs'
        abc_args = abc_args['inputs']
        # Validate specs and abc_args match.
        expected_input = self.config.input_map
        if expected_input.keys() != abc_args.keys():
            # Check if missing key is optional.
            set_diff = set(self.config.input_map.keys()).difference(abc_args.keys())
            for key in set_diff:
                raise ValueError(
                    f'Module \"{self.name}\" expects variable \"{key}\" but it was not provided.'
                )
            # Check if extra keys are provided.
            set_diff = set(abc_args.keys()).difference(expected_input.keys())
            for key in set_diff:
                raise ValueError(
                    f'Module \"{self.name}\" received an extra variable \"{key}\" but it is not part of the specification.'
                )
        # Finish specs, use abc_args to skip optional missing keys.
        input_specs = {}
        for key, payload in abc_args.items():
            # Sanity check
            if not isinstance(payload, SparkPayload):
                raise TypeError(
                    f'Expected non-optional input payload \"{key}\" of module \"{self.name}\" to be '
                    f'of type \"{SparkPayload.__name__}\" but got type {type(payload)}'
                )
            # PortSpecs are immutable, we need to create a new one.
            input_specs[key] = PortSpecs(
                payload_type=payload.__class__,
                shape=payload.shape,
                dtype=payload.dtype,
                description=f'Auto-generated input spec for input \"{key}\" of module \"{self.name}\".',
                # Payload specific build metadata
                async_spikes=payload.async_spikes if isinstance(payload, SpikeArray) else None,
                inhibition_mask=payload.inhibition_mask if isinstance(payload, SpikeArray) else None
            )
        self._input_specs = data(input_specs)
    


    # NOTE: We need to override _construct_output_specs since sig_parser.get_output_specs will lead to an incorrect 
    # signature because Controller can have an arbitrary number of outputs all under the key "outputs".
    def _construct_output_specs(self, abc_args: dict[str, SparkPayload]) -> None:
        # Output is constructed dynamically from the output map, there is no ground truth.
        expected_output = self.config.output_map
        output_specs = {}
        for output_name in expected_output.keys():
            port_map: PortMap = expected_output[output_name]['input']
            port_spec: PortSpecs = expected_output[output_name]['spec']
            output_specs[output_name] = PortSpecs(
                payload_type=port_spec.payload_type,
                shape=abc_args[output_name].shape,
                dtype=abc_args[output_name].dtype,
                description=f'Auto-generated output spec for output \"{output_name}\".',
                # Payload specific build metadata
                async_spikes=abc_args[output_name].async_spikes if isinstance(abc_args[output_name], SpikeArray) else None,
                inhibition_mask=abc_args[output_name].inhibition_mask if isinstance(abc_args[output_name], SpikeArray) else None
            )
        self._output_specs = data(output_specs)



    def _construct_property_specs(self,) -> None:
        self._property_specs = data({})



    def reset(self):
        """
            Resets all the modules to its initial state.
        """
        for module_name in self._modules_list:
            module: SparkModule = getattr(self, module_name)
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



    def _parse_tree_structure(self, current_depth: int = 0, name: str | None = None) -> str:
        """
            Parses the tree with to produce a string with the appropiate format for the ascii_tree method.
        """
        if name:
            rep = current_depth * ' ' + f'{name} ({self.__class__.__name__})\n'
        else:
            rep = current_depth * ' ' + f'{self.__class__.__name__}\n'
        for module_name in self._modules_list:
            module: SparkModule = getattr(self, module_name)
            rep += module._parse_tree_structure(current_depth+1, name=module_name)
        return rep

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################