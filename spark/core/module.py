#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spark.core.payloads import SparkPayload

import abc
import jax
import inspect
import jax.numpy as jnp
import flax.nnx as nnx
import typing as tp
import dataclasses as dc
from jax.typing import DTypeLike
from functools import wraps

import spark.core.signature_parser as sig_parser
import spark.core.validation as validation
import spark.core.utils as utils
from spark.core.specs import OutputSpec, InputSpec
from spark.core.config import BaseSparkConfig
from spark.core.variables import Variable

# TODO: Support for list[SparkPayloads] was implemented in a wacky manner and 
# may have damage several parts of the module. This needs to be further validated. 
# Some checks that assumed that shape was a tuple may now not work properly.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

ConfigT = tp.TypeVar("ConfigT", bound=BaseSparkConfig)
InputT = tp.TypeVar("InputT")

class ModuleOutput(tp.TypedDict):
    """
        Spark module output template
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkMeta(nnx.module.ModuleMeta):
    """
        Metaclass for Spark Modules.
    """

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
            # Check if already built.
            if not getattr(self, '__built__', False):
                self._build(*args, **kwargs)
            # After potentially building, execute the original __call__ logic.
            return original_call(self, *args, **kwargs)

        # Mark our new wrapper function. This is for the safeguard check above.
        setattr(wrapped_call, '__is_wrapped__', True)

        # Replace the original __call__ method on the class with our new wrapped version.
        setattr(cls, '__call__', wrapped_call)

        return cls
            
#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkModule(nnx.Module, abc.ABC, tp.Generic[ConfigT, InputT], metaclass=SparkMeta):
    """
        Base class for Spark Modules
    """

    name: str = 'name'
    config: ConfigT
    default_config: type[ConfigT]

    # NOTE: This is a workaround to require all childs of SparkModule to define a, 
    # default_config while at the same time allow for a lazy definition of the property. 
    def __init_subclass__(cls, **kwargs):
        from spark.core.config import BaseSparkConfig
        super().__init_subclass__(**kwargs)
        # Special cases and abstract classes dont need config yet they are SparkModules ¯\_(ツ)_/¯
        is_abc = inspect.isabstract(cls) and len(getattr(cls, '__abstractmethods__', set())) == 0
        if is_abc:
            return
        # Check if defines config
        resolved_hints = tp.get_type_hints(cls)
        config_type = resolved_hints.get('config', None)
        if not config_type or config_type == ConfigT or not issubclass(config_type, BaseSparkConfig):
            raise AttributeError('SparkModules must define a valid config: type[BaseSparkConfig] attribute.')
        cls.default_config = tp.cast(type[ConfigT], config_type)



    def __init__(self, *, config: ConfigT | None = None, name: str | None = None, **kwargs):
        # Initialize super.
        super().__init__()
        # Override config if provided
        if config is None:
            self.config = self.default_config(**kwargs)
        else:
            import copy
            self.config = copy.deepcopy(config)
            self.config.merge(partial=kwargs)
        # Define default parameters.
        self.name = name if name else self.__class__.__name__
        dtype = getattr(self.config, 'dtype', None)
        if dtype:
            self._dtype = dtype
        dt = getattr(self.config, 'dt', None)
        if dt:
            self._dt = dt
        seed = getattr(self.config, 'seed', None)
        if seed:
            self._seed = seed
            # Random engine key.
            self.rng = Variable(jax.random.PRNGKey(self._seed))
        # Specs
        self._input_specs: dict[str, InputSpec] | None = None
        self._output_specs: dict[str, OutputSpec] | None = None
        # Flags
        self.__built__: bool = False
        self.__allow_cycles__: bool = False
        self._recurrent_shape_contract: dict[str, tuple[int, ...]] | None = None



    @classmethod 
    def get_config_spec(cls) -> type[BaseSparkConfig]:
        """
            Returns the default configuratio class associated with this module.
        """
        type_hints = tp.get_type_hints(cls)
        return type_hints['config']



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
        self._input_specs = self._construct_input_specs(bound_args.arguments)

        # Build model.
        self.build(self._input_specs)
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
        from spark.core.payloads import ValueSparkPayload
        mock_input = {}
        for key, value in self._input_specs.items():
            if value.payload_type and issubclass(value.payload_type, ValueSparkPayload):
                if not isinstance(value.shape, list):
                        mock_input[key] = value.payload_type(jnp.zeros(value.shape, dtype=value.dtype))
                else:
                        mock_input[key] = [value.payload_type(jnp.zeros(s, dtype=value.dtype)) for s in value.shape]
        abc_output = self.__call__(**mock_input)

        # Contruct output sepcs.
        self._output_specs = self._construct_output_specs(abc_output)



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



    def set_recurrent_shape_contract(
            self, 
            shape: tuple[int, ...] | None = None, 
            output_shapes: dict[str, tuple[int, ...]] | None = None
        ) -> None:
        """
            Recurrent shape policy pre-defines expected shapes for the output specs.

            This is function is a binding contract that allows the modules to accept self connections.

            Input:
                shape: tuple[int, ...], A common shape for all the outputs.
                output_shapes: dict[str, tuple[int, ...]], A specific policy for every single output variable.

            NOTE: If both, shape and output_specs, are provided, output_specs takes preference over shape.
        """

        # Validate shapes.
        output_specs = sig_parser.get_output_specs(type(self))
        if not output_shapes is None:
            # Validate specs and output_shapes match.
            if output_specs.keys() != output_shapes.keys():
                raise ValueError(
                    f'Keys missmatch between expected outputs and provided outputs. '
                    f'Expected: {list(output_specs.keys())}. '
                    f'Provided: {list(output_shapes.keys())},'
                )
            # Construct recurrent contract.
            self._recurrent_shape_contract = {}
            for key, value in output_shapes.items():
                try:
                    self._recurrent_shape_contract[key] = utils.validate_shape(value)
                except Exception as e:
                    raise TypeError(
                        f'Arguemnt \"output_shapes[\"{key}\"]\" is not a valid shape. Error: {e}'
                    )
        elif not shape is None:
            # Construct recurrent contract.
            self._recurrent_shape_contract = {}
            try:
                for key in output_specs.keys():
                    self._recurrent_shape_contract[key] = utils.validate_shape(shape)
            except Exception as e:
                raise TypeError(
                    f'Arguemnt \"shape\" is not a valid shape. Error: {e}'
                )
        else:
            raise ValueError(
                f'Expected \"output_spec\" or \"shape\" to be provided but both are None.'
            )
        # Enable recurrence flag.
        self.__allow_cycles__ = True
        


    def get_recurrent_shape_contract(self,):
        """
            Retrieve the recurrent shape policy of the module.
        """
        if self.__allow_cycles__:
            return self._recurrent_shape_contract
        raise RuntimeError(
            f'Trying to access the recurrent shape contract of module \"{self.name}\", but \"set_recurrent_shape_contract\" '
            f'has not been called with the appropiate shapes. If you are trying to use \"{self.name}\" within a recurrent context '
            f'set the recurrent shape contract inside the __init__ method of \"{self.__class__.__name__}\".'
        )



    def _construct_input_specs(self, abc_args: dict[str, SparkPayload | list[SparkPayload]]) -> dict[str, InputSpec]:
        """
            Input spec constructor.
        """
        # Get default spec from signature. Default specs helps validate the user didn't make a mistake.
        input_specs = sig_parser.get_input_specs(type(self))
        # Validate specs and abc_args match.
        if input_specs.keys() != abc_args.keys():
            # Check if missing key is optional.
            set_diff = set(input_specs.keys()).difference(abc_args.keys())
            for key in set_diff:
                raise ValueError(
                    f'Module \"{self.name}\" expects variable \"{key}\" but it was not provided.'
                )
            # Check if extra keys are provided.
            set_diff = set(abc_args.keys()).difference(input_specs.keys())
            for key in set_diff:
                raise ValueError(
                    f'Module \"{self.name}\" received an extra variable \"{key}\" but it is not part of the specification.'
                )
        # Finish specs, use abc_args to skip optional missing keys.
        from spark.core.payloads import SpikeArray
        for key, payload in abc_args.items():
            # NOTE: This is a particular case for modules that expect abstract variadic positional arguments. 
            # I am looking at you Concat (╯°□°）╯︵ ┻━┻.
            payload_type = input_specs[key].payload_type
            if payload_type and payload_type.__name__ == 'SparkPayload':
                payload_type = payload.__class__ if not isinstance(payload, list) else payload[0].__class__
            else:
                payload_type = input_specs[key].payload_type
            shape = payload.shape if not isinstance(payload, list) else [p.shape for p in payload]
            dtype = payload.dtype if not isinstance(payload, list) else payload[0].dtype
            # InputSpec are immutable, we need to create a new one.
            input_specs[key] = InputSpec(
                payload_type=payload_type,
                shape=shape,
                dtype=dtype,
                description=f'Auto-generated input spec for input \"{key}\" of module \"{self.name}\".',
                # Payload specific build metadata
                async_spikes=payload.async_spikes if isinstance(payload, SpikeArray) else None
            )
        return input_specs



    def _construct_output_specs(self, abc_args: ModuleOutput) -> dict[str, OutputSpec]:
        """
            Output spec constructor.
        """
        # Get default spec from signature.
        output_specs = sig_parser.get_output_specs(type(self))
        # Validate specs and abc_args match.
        if output_specs.keys() != abc_args.keys():
            raise ValueError(
                f'Keys missmatch between expected outputs (TypedDict) and outputs (__call__). '
                f'Expected: {list(output_specs.keys())}. '
                f'__call__: {list(abc_args.keys())},'
            )
        # Finish specs.
        for key in output_specs.keys():
            output_specs[key] = OutputSpec(
                payload_type=output_specs[key].payload_type,
                shape=abc_args[key].shape,
                dtype=abc_args[key].dtype,
                description=f'Auto-generated output spec for input \"{key}\" of module \"{self.name}\".',
            )
        return output_specs



    def get_input_specs(self) -> dict[str, InputSpec]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        if not self._input_specs:
            raise RuntimeError('Module not yet built.')
        return self._input_specs



    def get_output_specs(self) -> dict[str, OutputSpec]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        if not self._output_specs:
            raise RuntimeError('Module not yet built.')
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



    def get_rng_keys(self, num_keys: int) -> jax.Array | list[jax.Array]:
        """
            Generates a new collection of random keys for the JAX's random engine.
        """
        self.rng.value, *keys = jax.random.split(self.rng.value, num_keys+1)
        if num_keys == 1:
            return keys[0]
        return keys



    @abc.abstractmethod
    def __call__(self, **kwargs: InputT) -> ModuleOutput:
        """
            Execution method.
        """
        pass



    def __repr__(self,):
        return f'{self.__class__.__name__}(...)'



    def inspect(self,) -> str:
        """
            Returns a formated string of the datastructure.
        """
        print(utils.ascii_tree(self._parse_tree_structure()))



    def _parse_tree_structure(self, current_depth: int = 0, name: str | None = None) -> str:
        """
            Parses the tree with to produce a string with the appropiate format for the ascii_tree method.
        """
        if name:
            rep = current_depth * ' ' + f'{name} ({self.__class__.__name__})\n'
        else:
            rep = current_depth * ' ' + f'{self.__class__.__name__}\n'
        for name, value in self.__dict__.items():
            if isinstance(value, SparkModule):
                rep += value._parse_tree_structure(current_depth+1, name=name)
        return rep


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################