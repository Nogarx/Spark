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
import copy
import spark.core.signature_parser as sig_parser
import spark.core.validation as validation
import spark.core.utils as utils
from spark.core.specs import PortSpecs
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
        self._input_specs: dict[str, PortSpecs] | None = None
        self._output_specs: dict[str, PortSpecs] | None = None
        # Flags
        self.__built__: bool = False
        self.__allow_cycles__: bool = False
        self._contract_specs: dict[str, PortSpecs] | None = None



    @classmethod 
    def get_config_spec(cls) -> type[BaseSparkConfig]:
        """
            Returns the default configuration class associated with this module.
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
        self._input_specs = nnx.data(self._construct_input_specs(bound_args.arguments))

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
                    abc_inputs[key] = self._input_specs[key].payload_type(
                        jax.ShapeDtypeStruct(shape=self._input_specs[key].shape, dtype=self._input_specs[key].dtype))
            # Evaluate.
            abc_output = nnx.eval_shape(self.__call__, **abc_inputs)

        # Evaluate workaround, just use the __call__ directly with some mock input. 
        from spark.core.payloads import ValueSparkPayload
        mock_input = {}
        for key, value in self._input_specs.items():
            mock_input[key] = value._create_mock_input()
        abc_output = self.__call__(**mock_input)
        # Contruct output sepcs.
        self._output_specs = nnx.data(self._construct_output_specs(abc_output))



    def build(self, input_specs: dict[str, PortSpecs]) -> None:
        """
            Build method.
        """
        pass



    def reset(self,):
        """
            Reset module to its default state.
        """
        pass



    def set_contract_output_specs(self, contract_specs: dict[str, PortSpecs]) -> None:
        """
            Recurrent shape policy pre-defines expected shapes for the output specs.

            This is function is a binding contract that allows the modules to accept self connections.

            Input:
                contract_specs: dict[str, PortSpecs], A dictionary with a contract for the output specs.

            NOTE: If both, shape and output_specs, are provided, output_specs takes preference over shape.
        """

        # Validate contract.
        output_specs = sig_parser.get_output_specs(type(self))
        for port in output_specs.keys():
            if port not in contract_specs:
                raise ValueError(
                    f'PortSpecs for port {port} is not defined.'
                )
            if not isinstance(contract_specs[port], PortSpecs):
                raise ValueError(
                    f'PortSpecs for port got {contract_specs[port].__class__.__name__}, which is not a valid PortSpecs..'
                )   
        # Set contract
        self._contract_specs = contract_specs
        # Enable recurrence flag.
        self.__allow_cycles__ = True
        


    def get_contract_output_specs(self,) -> dict[str, PortSpecs] | None:
        """
            Retrieve the recurrent spec policy of the module.
        """
        if self.__allow_cycles__:
            return self._contract_specs
        raise RuntimeError(
            f'Trying to access the contract output specs of module \"{self.name}\", but \"set_contract_output_specs\" '
            f'has not been called. If you are trying to use \"{self.name}\" within a recurrent context '
            f'set the recurrent shape contract inside the __init__ method of \"{self.__class__.__name__}\".'
        )



    def _construct_input_specs(self, abc_args: dict[str, SparkPayload | list[SparkPayload]]) -> dict[str, PortSpecs]:
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
            # PortSpecs are immutable, we need to create a new one.
            input_specs[key] = PortSpecs(
                payload_type=payload_type,
                shape=shape,
                dtype=dtype,
                description=f'Auto-generated input spec for input \"{key}\" of module \"{self.name}\".',
                # Payload specific build metadata
                async_spikes=payload.async_spikes if isinstance(payload, SpikeArray) else None,
                inhibition_mask=payload.inhibition_mask if isinstance(payload, SpikeArray) else None
            )
        return input_specs



    def _construct_output_specs(self, abc_args: ModuleOutput) -> dict[str, PortSpecs]:
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
        from spark.core.payloads import SpikeArray
        for key in output_specs.keys():
            output_specs[key] = PortSpecs(
                payload_type=output_specs[key].payload_type,
                shape=abc_args[key].shape,
                dtype=abc_args[key].dtype,
                description=f'Auto-generated output spec for input \"{key}\" of module \"{self.name}\".',
                # Payload specific build metadata
                async_spikes=abc_args[key].async_spikes if isinstance(abc_args[key], SpikeArray) else None,
                inhibition_mask=abc_args[key].inhibition_mask if isinstance(abc_args[key], SpikeArray) else None
            )
        return output_specs



    def get_input_specs(self) -> dict[str, PortSpecs]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        if not self._input_specs:
            raise RuntimeError('Module not yet built.')
        return self._input_specs



    def get_output_specs(self) -> dict[str, PortSpecs]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        if not self._output_specs:
            raise RuntimeError('Module not yet built.')
        return self._output_specs



    @classmethod
    def _get_input_specs(cls) -> dict[str, PortSpecs]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        return sig_parser.get_input_specs(cls)



    @classmethod
    def _get_output_specs(cls) -> dict[str, PortSpecs]:
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



    def checkpoint(self, path, overwrite=False) -> None:

        import os
        import tarfile
        import shutil
        import pathlib
        import datetime
        import orbax.checkpoint as ocp
        from spark.core.flax_imports import split

        try:
            # Check if file exists
            tar_path = pathlib.Path(path).with_suffix('.spark')
            if tar_path.exists() and not overwrite:
                raise RuntimeError(
                    f'Attempting to overwrite file with {tar_path}. Set \"overwrite=True\" if this was intended.'
                )
            # Create temporary directory
            timestamp = datetime.datetime.now().timestamp()
            temp_dir = pathlib.Path(
                os.path.join('/'.join(path.split('/')[:-1]), f'tmp_{timestamp}')
            )
            temp_dir.mkdir(exist_ok=True)
            # Save config
            config_path = pathlib.Path(os.path.join(temp_dir, 'model.scfg'))
            config: BaseSparkConfig = self.config
            config.__metadata__['input_specs'] = self.get_input_specs()
            config.to_file(config_path.absolute())
            # Save state
            _, state = split((self))
            ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
            state_path = pathlib.Path(os.path.join(temp_dir, 'state'))
            ckptr.save(state_path.absolute(), state)
            ckptr.wait_until_finished()
            # Compress into tar
            with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(temp_dir, arcname=".")
            # Remove temporary files
            shutil.rmtree(temp_dir)
            # Message
            print(f'Checkpoint for model {self.__class__.__name__} successfully saved to path: {tar_path}.')
        except Exception as e:
            raise RuntimeError(f'Unable to generate checkpoint for model {self.__class__.__name__}: {e}')


    # TODO: This is a potentially dangerous operation. We need to add some safe guards
    #  to prevent malicious software to enter a computer unintentionally.
    @classmethod
    def from_checkpoint(cls, path, safe=True) -> SparkModule:

        import os
        import tarfile
        import shutil
        import pathlib
        import datetime
        import orbax.checkpoint as ocp
        from spark.core.config import BaseSparkConfig
        from spark.core.flax_imports import split, merge

        def is_child(member_name: str, parent: str) -> bool:
            m = pathlib.PurePosixPath(member_name)
            p = pathlib.PurePosixPath(parent)
            # Must be relative
            if m.is_absolute():
                return False
            # Normalize and ensure containment
            try:
                m.relative_to(p)
                return True
            except ValueError:
                return False

        def safe_member(m: tarfile.TarInfo) -> bool:
            return not (m.issym() or m.islnk())
        
        # Open tar file
        timestamp = datetime.datetime.now().timestamp()
        temp_dir = pathlib.Path(
            os.path.join('/'.join(path.split('/')[:-1]), f'tmp_{timestamp}')
        )
        try:
            with tarfile.open(path, "r:gz") as tar:
                config_file = tar.getmember('./model.scfg')
                state_dir = tar.getmember('./state')
                if safe and not config_file.isfile():
                    raise RuntimeError(
                        f'Unable to validate model.scfg. If you still want to try to extract the file set \"safe=False\"'
                    )
                if safe and not state_dir.isdir():
                    raise RuntimeError(
                        f'Unable to validate state. If you still want to try to extract the file set \"safe=False\"'
                    )
                tar.extract(config_file, path=temp_dir, filter='data') 
                state_files = []
                for member in tar.getmembers():
                    if is_child(member.name, state_dir.name):
                        if safe_member(member):
                            state_files.append(member)
                        else:
                            raise RuntimeError(
                                f'A strange possibly malicious file was detected: \"{member.name}\". If you still want to try to extract the file set \"safe=False\"'
                            )
                tar.extractall(members=state_files, path=temp_dir, filter='data') 
            # Restore model
            config_path = pathlib.Path(os.path.join(temp_dir, 'model.scfg'))
            config = BaseSparkConfig.from_file(config_path.absolute())
            # Get model class
            model_cls: type[SparkModule] = config.class_ref
            # Initialize the module.
            model = model_cls(config=config)
            dummy_input = {
                port_name: port_spec._create_mock_input() for port_name, port_spec in config.__metadata__['input_specs'].items()
            }
            model(**dummy_input)
            graph, template_state = split((model))
            # Restore state
            ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
            state_path = pathlib.Path(os.path.join(temp_dir, 'state'))
            state = ckptr.restore(state_path.absolute(), template_state)
            ckptr.wait_until_finished()
            #Assemble
            model = merge(graph, state)
            # Remove temporary files
            shutil.rmtree(temp_dir)
            # Message
            print(f'Model {model.__class__.__name__} loaded successfully from path: {path}.')
            return model
        except Exception as e:
            # Remove temporary files
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise RuntimeError(f'Unable to restore checkpoint for model {path}: {e}')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################