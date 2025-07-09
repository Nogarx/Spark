#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import OutputSpecs, InputSpecs, InputArgSpec
    from spark.core.payloads import SparkPayload
    from spark.core.shape import Shape
    
import os
import abc
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Any, Optional, Dict, List
from spark.core.wrappers import HookingMeta
import spark.core.signature_parser as sig_parser

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Meta module to resolve metaclass conflicts
class SparkMeta(nnx.module.ModuleMeta, HookingMeta):
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: We need a reliable way to infer the shape/type for inputs and outputs.
class SparkModule(nnx.Module, abc.ABC, metaclass=SparkMeta):
    name: str

    # TODO: Implement lazy shape inference. The Metaclass HookingMeta implements a hook system that can be used to
    # execute pre and post functions for any method specified based on the content of _hooks. The plan is to use setup + init
    # when input and output shapes are not specified.
    #_hooks = {
    #    '__call__': {
    #        'pre': 'setup'
    #    }
    #}
    

    def __init__(self, 
                seed: Optional[int] = None, 
                dtype: Optional[Any] = jnp.float16, 
                dt: Optional[float] = 1.0,
                **kwargs):
        # Sanity checks.
        if not isinstance(dt, float) and dt >= 0:
            raise ValueError(f'"dt" must be a positive float, got {dt}')
        # Initialize super.
        super().__init__()
        # Define default parameters.
        self._dtype = dtype
        self._dt = dt
        # Random engine key.
        self._seed = int.from_bytes(os.urandom(4), 'little') if seed is None else seed
        self.rng = nnx.Variable(jax.random.PRNGKey(self._seed))
        # Built flag
        self.__built__ = None

    @property
    @abc.abstractmethod   
    def input_shapes(self,) -> List[Shape]:
        pass

    @property
    @abc.abstractmethod   
    def output_shapes(self,) -> List[Shape]:
        pass

    def setup(self, *args: SparkPayload, **kwargs) -> None:
        if self.__built__ is None:
            self.init(*args, **kwargs)
            self.__built__ = True

    #@abc.abstractmethod   
    def init(self, *args) -> None:
        pass

    def __post_init__(self,):
        # More sanity checks. Validate specs works.
        self.get_input_specs()
        self.get_output_specs()

    def get_rng_keys(self, num_keys: int):
        self.rng.value, *keys = jax.random.split(self.rng.value, num_keys+1)
        if num_keys == 1:
            return keys[0]
        return keys

    def get_input_specs(self) -> Dict[str, InputSpecs]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        if not hasattr(self, '_input_specs'):
            # Build the specs.
            self._input_specs = sig_parser.get_input_specs(self)
        return self._input_specs
    
    @classmethod
    def _get_input_specs(cls) -> Dict[str, InputSpecs]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        return sig_parser.get_input_specs(cls)

    def get_output_specs(self) -> Dict[str, OutputSpecs]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        # Build the spec.
        if not hasattr(self, '_output_specs'):
            # Build the specs.
            self._output_specs = sig_parser.get_output_specs(self)
        return self._output_specs

    @classmethod
    def _get_output_specs(cls) -> Dict[str, OutputSpecs]:
        """
            Returns a dictionary of the SparkModule's input port specifications.
        """
        return sig_parser.get_output_specs(cls)

    @classmethod
    def _get_init_signature(cls) -> Dict[str, InputArgSpec]:
        """
            Returns a dictionary mapping logical output port names to their OutputSpecs.
        """
        return sig_parser.get_method_signature(cls.__init__)  

    @classmethod
    def from_graph(cls, graph_serializer: Dict[str, Any]) -> SparkModule:
        """
            Instantiate the module from a configuration dictionary.
        """
        return cls(**graph_serializer['init'])

    @abc.abstractmethod
    def __call__(self, *args: SparkPayload) -> Dict[str, SparkPayload]:
        pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################