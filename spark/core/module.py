#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import OutputSpec, InputSpec, InputArgSpec, VarSpec
    from spark.core.payloads import SparkPayload
    from spark.core.shape import Shape
from spark.core.specs import InputSpec, VarSpec
from spark.core.payloads import DummyArray

import os
import abc
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import inspect
from jax.typing import DTypeLike
from typing import Any, TypedDict, Type, get_type_hints
from spark.core.wrappers import HookingMeta
from functools import wraps
from dataclasses import dataclass, fields, MISSING, asdict, field
import spark.core.signature_parser as sig_parser
import spark.core.validation as validation
from spark.core.configuration import SparkConfig

# TODO: Support for list[SparkPayloads] was implemented in a wacky manner and 
# may have damage several parts of the module. This needs to be further validated. 
# Some checks that assumed that shape was a tuple may now not work properly.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Meta module to resolve metaclass conflicts
class SparkMeta(nnx.module.ModuleMeta, HookingMeta):

    def __new__(mcs, name, bases, dct):
        # Instantiate object
        cls = super().__new__(mcs, name, bases, dct)

        # Get __call__ method
        original_call = getattr(cls, '__call__', None)

        # Check if __call__ is valid.
        if original_call is None or original_call is object.__call__:
            # Skip abstract classes.
            is_base_model = dct.get('_is_base_model', False)
            if not is_base_model:
                 raise TypeError(f'Class "{name}" must implement or inherit a __call__ method to be used with {mcs.__name__}.')
            return cls


        # We check for a marker on the function to avoid wrapping it again.
        if getattr(original_call, '__is_wrapped__', False):
            return cls

        # Wrapper function.
        @wraps(original_call)
        def wrapped_call(self, *args, **kwargs):
            """
                Wrapper around __call__ to trigger lazy init.
            """
            # Use getattr for a safe check on the instance's `__built__` flag.
            if not getattr(self, '__built__', False):
                self._build(*args, **kwargs)
                #try:
                #    # Call the _build method.
                #    self._build(*args, **kwargs)
                #except Exception as e:
                #    # Provide a more informative error message upon failure.
                #    raise RuntimeError(f'An error was encountered while trying to build module "{self.name}": {e}.') from e
            # After potentially building, execute the original __call__ logic.
            return original_call(self, *args, **kwargs)

        # Mark our new wrapper function. This is for the safeguard check above.
        wrapped_call.__is_wrapped__ = True

        # Replace the original __call__ method on the class with our new wrapped version.
        setattr(cls, '__call__', wrapped_call)

        return cls
            
#-----------------------------------------------------------------------------------------------------------------------------------------------#
            
# TODO: We need a reliable way to infer the shape/type for inputs and outputs.
class SparkModule(nnx.Module, abc.ABC, metaclass=SparkMeta):

    name: str = 'name'
    config: SparkConfig
    default_config: type[SparkConfig] = SparkConfig

    # NOTE: This is a workaround to require all childs of SparkModule to define a, 
    # default_config while at the same time allow for a lazy definition of the property. 
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        print(cls.__name__)
        # Special cases and abstract classes dont need config but yet they are still SparkModules ¯\_(ツ)_/¯
        is_abc = inspect.isabstract(cls) and len(getattr(cls, '__abstractmethods__', set())) == 0
        if cls.__name__ in ['Brain'] or is_abc:
            cls.default_config = None
            return
        # Check if defines config
        resolved_hints = get_type_hints(cls)
        config_type = resolved_hints.get('config')
        if not config_type and not getattr(config_type, '__is_spark_config__', None):
            raise AttributeError('SparkModules must define a valid config: type[SparkConfig] attribute.')
        cls.default_config = config_type

    def __init__(self, *, config: SparkConfig = None, **kwargs):
        # Initialize super.
        super().__init__()
        # Override config if provided
        self.config = self.default_config.create(**kwargs) if config is None else config.merge(**kwargs)
        # Define default parameters.
        self._dtype = self.config.dtype
        self._dt = self.config.dt
        self._seed = self.config.seed
        # Random engine key.
        self.rng = nnx.Variable(jax.random.PRNGKey(self._seed))
        #self.rng = nnx.Rngs(default=42)
        # Specs
        self._input_specs: dict[str, InputSpec] = None
        self._output_specs: dict[str, OutputSpec] = None
        # Flags
        self.__built__: bool = False
        self.__allow_cycles__: bool = False
        self._default_payload_type = DummyArray

    @property
    @abc.abstractmethod
    def default_config(self) -> type[SparkConfig]:
        """
            Returns the default configuration dataclass for this module.
        """
        pass
    
    @classmethod
    def get_config_spec(cls):
        """
            Expose safe-to-inspect configuration metadata.
        """
        if not hasattr(cls, 'config'):
            return {}
        return {
            field.name: {
                'type': field.type.__name__,
                'default': field.default if field.default is not MISSING else None,
                'description': field.metadata.get('description', '')
            }
            for field in fields(cls.config)
        }

    def initialize(self, shape: Shape = None, dtype: DTypeLike = None, specs: dict[str, VarSpec] = None) -> None:
        # Skip if already compiled
        if self.__built__:
            return

        # Check if got shape and dtype
        if shape and validation.is_shape(shape) and dtype and validation.is_dtype(dtype):
            # Define mock input, all using the same shape and dtype.
            input_specs = sig_parser.get_input_specs(type(self))
            mock_input = {}
            for key, value in input_specs.items():
                payload_type = value.payload_type
                if payload_type.__name__ == 'SparkPayload' and self._default_payload_type:
                    payload_type = self._default_payload_type
                mock_input[key] = payload_type(jnp.zeros(shape, dtype=dtype))

        # Check if got list[shape] and dtype
        # This scenario is only for variadic arguments.
        elif shape and validation.is_list_shape(shape):
            # Define mock input, all using the same shape and dtype.
            input_specs = sig_parser.get_input_specs(type(self))
            mock_input = {}
            for key, value in input_specs.items():
                payload_type = value.payload_type
                if payload_type.__name__ == 'SparkPayload' and self._default_payload_type:
                    payload_type = self._default_payload_type
                mock_input[key] = [payload_type(jnp.zeros(s, dtype=dtype)) for s in shape]

        # Check if got dict
        elif specs and validation.is_dict_of(specs, VarSpec):
            input_specs = sig_parser.get_input_specs(type(self))
            # Make sure the keys match. To prevent errors and avoid missunderstadings.
            for key in input_specs.keys():
                if not key in specs:
                    raise ValueError(f'Expected VarSpec for key "{key}" but key is not defined.')
            for key in specs.keys():
                if not key in input_specs:
                    raise ValueError(f'VarSpec defined for key "{key}" but key is not part of the __call__ method.')
            # Define mock input
            mock_input = {}
            for key, value in input_specs.items():
                payload_type = value.payload_type
                if payload_type.__name__ == 'SparkPayload' and self._default_payload_type:
                    payload_type = self._default_payload_type
                mock_input[key] = payload_type(jnp.zeros(specs[key].shape, dtype=specs[key].dtype))

        # Raise error, cannot build mock input safely.
        else:
            raise RuntimeError(f'Expected either shape and dtype or specs to be not None but got'
                               f'shape: {shape}, dtype: {dtype} and specs:{specs}')

        # Use call with the mock input.
        self.__call__(**mock_input)

        # Reset stateful modules 
        self.reset()



    def _build(self, *abc_args: SparkPayload, **kwargs) -> None:
        """
            Triggers the shape inference and parameter initialization cascade.
        """

        # Bind arguments to avoid parameter mixing.
        call_signature = inspect.signature(self.__call__)
        try:
            bound_args = call_signature.bind(*abc_args, **kwargs)
            bound_args.apply_defaults()
        except TypeError as error:
            raise TypeError(f'Error binding arguments for "{self.name}": {error}') from error

        # Construct input specs.
        input_specs = self._construct_input_specs(bound_args.arguments)

        # Build model.
        self.build(input_specs)
        self.__built__ = True

        # TODO: The correct approach to build the model is through eval_shape. 
        # However, Constant and Variable are clashing with JAX tracer
        # For some reason, the tracer thinks that Constant produces a side effect 
        # when interacting with other arrays and Variable leaks.
        # This probably requires extending both classes to tell them what to do with ShapeDtypeStruct 
        if False:
            # Replace arrays with spec-backed shape proxies.
            abc_inputs = {}
            for key, value in bound_args.arguments.items():
                if value and validation._is_payload_instance(value):
                    abc_inputs[key] = input_specs[key].payload_type(
                        jax.ShapeDtypeStruct(shape=input_specs[key].shape, dtype=input_specs[key].dtype))
            # Evaluate.
            abc_output = nnx.eval_shape(self.__call__, **abc_inputs)

        # Evaluate workaround, just use the __call__ directly with some mock input. 
        mock_input = {}
        for key, value in self._input_specs.items():
            if not isinstance(value.shape, list):
                mock_input[key] = value.payload_type(jnp.zeros(value.shape, dtype=value.dtype))
            else:
                mock_input[key] = [value.payload_type(jnp.zeros(s, dtype=value.dtype)) for s in value.shape]
        abc_output = self.__call__(**mock_input)

        # Contruct output sepcs.
        self._construct_output_specs(abc_output)



    def build(self, input_specs: dict[str, InputSpec]) -> None:
        """
            Build method.
        """
        pass



    def reset(self,):
        """
            Reset module to its default state.
        """
        pass



    def set_output_shapes(self, output_specs: Shape | dict[str, Shape | OutputSpec | SparkPayload]) -> None:
        """
            Auxiliary function prematurely defines the shape of the output_specs.

            This is function is a binding contract that allows the modules to accept self connections.
        """

        # Set output specs to the default
        self._output_specs = sig_parser.get_output_specs(type(self))

        # Check if is a Shape
        if validation.is_shape(output_specs):
            # Set the same shape for all outputs
            for k in self._output_specs.keys():
                object.__setattr__(self._output_specs[k], 'shape', output_specs)

        # Check if is dict[str, Shape]
        elif validation.is_dict_of(output_specs, Shape):
            for k in self._output_specs.keys():
                object.__setattr__(self._output_specs[k], 'shape', output_specs[k])

        # Check if is dict[str, OutputSpec] or dict[str, SparkPayload]
        elif validation.is_dict_of(output_specs, OutputSpec) or validation.is_dict_of(output_specs, SparkPayload):
            for k in self._output_specs.keys():
                object.__setattr__(self._output_specs[k], 'shape', output_specs[k].shape)

        # Raise error, output_spec is invalid
        else:
            raise TypeError(f'Expected "output_spec" to be of type "Shape | dict[str, Shape | OutputSpec | SparkPayload]" '
                            f'but got "{type(output_specs).__name__}".')

        # Enable recurrence flag.
        self.__allow_cycles__ = True



    def _construct_input_specs(self, abc_args: dict[str, SparkPayload | list[SparkPayload]]) -> dict[str, InputSpec]:

        # Get default spec from signature. Default specs helps validate the user didn't make a mistake.
        self._input_specs = sig_parser.get_input_specs(type(self))

        # Validate specs and abc_args match
        if len(self._input_specs) != len(abc_args):
            raise ValueError(f'Default Input Specs encounter {len(self._input_specs)} variables ' 
                             f'but only {len(abc_args)} were passed to the compiler.')
        
        # Finish specs
        for key in self._input_specs.keys():

            # Unpack
            value = self._input_specs[key]
            abc_paylaod = abc_args[key]

            # Sanity check
            if not (abc_paylaod is None) \
                and value.payload_type.__name__ != 'SparkPayload' \
                and not isinstance(abc_paylaod, value.payload_type):
                raise TypeError(f'Expected non-optional input payload "{key}" of module "{self.name}" to be '
                                f'of type "{value.payload_type}" but got type {type(abc_paylaod)}')
            
            # Replace default spec (Specs are frozen)
            if value.payload_type.__name__ == 'SparkPayload' and self._default_payload_type:
                # NOTE: This is a particular case for modules that expect abstract variadic positional arguments. 
                # I am looking at you Merger (╯°□°）╯︵ ┻━┻.
                object.__setattr__(self._input_specs[key], 'payload_type', self._default_payload_type)
            shape, dtype = None, None
            if abc_paylaod:
                shape = abc_paylaod.shape if not isinstance(abc_paylaod, list) else [p.shape for p in abc_paylaod]
                dtype = abc_paylaod.dtype if not isinstance(abc_paylaod, list) else abc_paylaod[0].dtype
            object.__setattr__(self._input_specs[key], 'shape', shape)
            object.__setattr__(self._input_specs[key], 'dtype', dtype)
            object.__setattr__(self._input_specs[key], 'description', 
                               f'Auto-generated input spec for input "{key}" of module "{self.name}".')

        return self._input_specs



    def _construct_output_specs(self, abc_args: dict[str, SparkPayload]) -> dict[str, OutputSpec]:

        # Get default spec from signature. Default specs helps validate the user didn't make a mistake.
        self._output_specs = sig_parser.get_output_specs(type(self))

        # Validate specs and abc_args match
        if len(self._output_specs) != len(abc_args):
            raise ValueError(f'Default Output Specs encounter {len(self._output_specs)} variables ' 
                            f'but only {len(abc_args)} were produced at the end of __call__.')
        
        # Finish specs
        for key in self._output_specs.keys():
            # Unpack
            value = self._output_specs[key]
            abc_paylaod = abc_args[key]

            # Sanity check
            if not (abc_paylaod is None) and not isinstance(abc_paylaod, value.payload_type):
                raise TypeError(f'Expected non-optional output payload "{key}" of module "{self.name}" to be '
                                f'of type "{value.payload_type}" but got type {type(abc_paylaod)}')
            
            # Replace default spec (Specs are frozen)
            object.__setattr__(self._output_specs[key], 'shape', abc_paylaod.shape if abc_paylaod else None)
            object.__setattr__(self._output_specs[key], 'dtype', abc_paylaod.dtype if abc_paylaod else None)
            object.__setattr__(self._output_specs[key], 'description', 
                            f'Auto-generated output spec for input "{key}" of module "{self.name}".')



    def set_fallback_payload_type(self, payload_type: type[SparkPayload]) -> None:
        self._default_payload_type = payload_type



    def get_input_specs(self) -> dict[str, InputSpec]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        if not self._input_specs:
            raise RuntimeError('Module requires to be initialized first.')
        return self._input_specs



    def get_output_specs(self) -> dict[str, OutputSpec]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        # Build the spec.
        if not self._output_specs:
            raise RuntimeError('Module requires to be initialized first.')
        return self._output_specs



    @classmethod
    def _get_input_specs(cls) -> dict[str, InputSpec]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        return sig_parser.get_input_specs(cls)



    @classmethod
    def _get_output_specs(cls) -> dict[str, OutputSpec]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        return sig_parser.get_output_specs(cls)



    @classmethod
    def _get_init_signature(cls) -> dict[str, InputArgSpec]:
        """
            Returns a dictionary mapping logical output port names to their OutputSpec.
        """
        return sig_parser.get_method_signature(cls.__init__)  



    def get_rng_keys(self, num_keys: int) -> jax.Array | list[jax.Array]:
        self.rng.value, *keys = jax.random.split(self.rng.value, num_keys+1)
        if num_keys == 1:
            return keys[0]
        return keys



    @abc.abstractmethod
    def __call__(self, *args: SparkPayload, **kwargs) -> dict[str, SparkPayload]:
        pass
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################