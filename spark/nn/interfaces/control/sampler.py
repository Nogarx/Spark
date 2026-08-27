#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp
import dataclasses as dc
import spark.core.utils as utils
from math import prod
from spark.core.specs import PortSpecs
from spark.core.backend import Constant
from spark.core.registry import register_interface, register_config
from spark.core.payloads import SparkPayload
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.control.base import ControlInterface, ControlInterfaceConfig, ControlInterfaceOutput, _build_signature_from_inputs

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class SamplerConfig(ControlInterfaceConfig):
    """
        Configuration for `Sampler`.

        Parameters
        ----------
        sample_size : tuple of int
            Shape of the result. May be larger than the input, in which case entries are drawn
            more than once.
    """
    
    sample_size: int = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Sample size to drawn from the population. May be larger than the population.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class Sampler(ControlInterface):
    """
        Draws a fixed set of entries from its inputs.

        The inputs are flattened, concatenated and indexed by a set of indices drawn once at build
        time and held for the lifetime of the module. The same entries are read on every step, so
        this is a fixed projection rather than a fresh sample per step.

        Indices are drawn with replacement, so ``sample_size`` may exceed the size of the input
        and an entry may appear more than once.

        Parameters
        ----------
        config : SamplerConfig
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        **inputs : SparkPayload
            Any number of inputs, all of the same payload type. Named by the graph.

        Output Ports
        ------------
        output : SparkPayload
            The drawn entries, of shape ``sample_size`` and of the same payload type as the inputs.

        Properties
        ----------
        indices : jax.Array
            Flat indices drawn at build time and read on every step. Read only.
    """
    config: SamplerConfig

    def __init__(self, config: SamplerConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.sample_size = self.config.sample_size

    def build(self, **abc_args: SparkPayload) -> None:
        # Validate payloads types.
        payload_type = None
        for key, value in abc_args.items():
            payload_type = type(value) if payload_type is None else payload_type
            if payload_type != type(value):
                raise TypeError(
                    f'Expected all payload types to be of same type \"{payload_type}\" '
                    f'but input spec \"{key}\" is of type "{type(value)}".'
                )
        self._payload_type = payload_type
        # Initialize shapes
        input_shape = utils.merge_shape_list([spec.shape for spec in abc_args.values()])
        # Initialize variables
        self._indices = Constant(
            jax.random.randint(
                self.get_rng_keys(1), 
                self.sample_size, 
                minval=0, 
                maxval=prod(input_shape)
            ), 
            dtype=jnp.uint32
        )

    @property
    def indices(self,) -> jax.Array:
        return self._indices.value

    def _overwrite_call_signature(self, raw_kwargs: dict[str, SparkPayload]) -> None:
        # Create the new Signature object and assign it to the __call__ method
        self.__call__.__func__.__signature__ = _build_signature_from_inputs(raw_kwargs)

    def __call__(self, **inputs: SparkPayload) -> ControlInterfaceOutput:
        """
            Reads the drawn entries out of the inputs.

            Parameters
            ----------
            **inputs : SparkPayload
                Any number of inputs, all of the same payload type.

            Returns
            -------
            ControlInterfaceOutput
                Dictionary with one entry, ``output``, of shape ``sample_size`` and of the payload
                type of the inputs.
        """
        # Control flow operation
        return {
            'output': self._payload_type(
                jnp.concatenate([x.value.reshape(-1) for x in inputs.values()])[self.indices]
            )
        }
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################