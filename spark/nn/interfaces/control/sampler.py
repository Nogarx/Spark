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
from spark.nn.interfaces.control.base import ControlInterface, ControlInterfaceConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class SamplerConfig(ControlInterfaceConfig):
    """
        Configuration for `Sampler`.

        Parameters
        ----------
        sample_size : int
            Number of entries in each output. May be larger than the input, in which case entries
            are drawn more than once.
        num_outputs : int, default 1
            Number of outputs, ``output_0`` to ``output_{num_outputs - 1}``.
        disjoint : bool, default False
            Draws the outputs from separate entries. While the input holds ``num_outputs *
            sample_size`` entries or more, no entry is drawn twice; past that, every entry is drawn
            as evenly as possible, and an output holds an entry twice only if ``sample_size``
            exceeds the input. Otherwise every output is drawn on its own, with replacement.
    """
    
    sample_size: int = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Number of entries in each output. May be larger than the input.',
        })
    num_outputs: int = dc.field(
        default = 1,
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Number of outputs, output_0 to output_{num_outputs - 1}.',
        })
    disjoint: bool = dc.field(
        default = False,
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Draws the outputs from separate entries, as far as the input holds enough of them.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class Sampler(ControlInterface):
    """
        Draws fixed sets of entries from its inputs, one per output.

        The inputs are flattened, concatenated and indexed by sets of indices drawn once at build
        time and held for the lifetime of the module. The same entries are read on every step, so
        each output is a fixed projection rather than a fresh sample per step.

        By default the indices of every output are drawn on their own, with replacement, so
        ``sample_size`` may exceed the size of the input and an entry may appear more than once.
        With ``disjoint``, the outputs are drawn from separate entries, which splits the input into
        ``num_outputs`` populations.

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
        output_0, ..., output_{num_outputs - 1} : SparkPayload
            The drawn entries, of shape ``sample_size`` and of the same payload type as the inputs.

        Properties
        ----------
        indices : jax.Array
            Flat indices drawn at build time and read on every step, of shape
            ``(num_outputs, sample_size)``. Read only.
    """
    config: SamplerConfig

    def __init__(self, config: SamplerConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.sample_size = self.config.sample_size
        self.num_outputs = self.config.num_outputs
        self.disjoint = self.config.disjoint

    @classmethod
    def _get_output_specs(cls, config: SamplerConfig | None = None) -> dict[str, PortSpecs]:
        """
            Returns the output port specifications, one per output of ``config``.

            Parameters
            ----------
            config : SamplerConfig, optional
                Configuration of the module. Without it, or without ``num_outputs``, one output.

            Returns
            -------
            dict of str to PortSpecs
                ``output_0`` to ``output_{num_outputs - 1}``, whose type follows the inputs.
        """
        count = max(1, int(getattr(config, 'num_outputs', None) or 1))
        return {
            f'output_{k}': PortSpecs(payload_type=SparkPayload, shape=None, dtype=None, description=f'Output port for output_{k}')
            for k in range(count)
        }

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
        self._indices = Constant(self._draw_indices(prod(input_shape)), dtype=jnp.uint32)

    def _draw_indices(self, size: int) -> jax.Array:
        """
            Draws the flat indices read by each output, one row per output.
        """
        key = self.get_rng_keys(1)
        shape = (self.num_outputs, self.sample_size)
        if not self.disjoint:
            return jax.random.randint(key, shape, minval=0, maxval=size)
        # One permutation of the entries, read as a cycle: each output takes the next sample_size entries along it, so
        # no entry is drawn again before every other entry was.
        order = jax.random.permutation(key, size)
        return order[jnp.arange(self.num_outputs * self.sample_size) % size].reshape(shape)

    @property
    def indices(self,) -> jax.Array:
        return self._indices.value

    def __call__(self, **inputs: SparkPayload) -> dict[str, SparkPayload]:
        """
            Reads the drawn entries out of the inputs.

            Parameters
            ----------
            **inputs : SparkPayload
                Any number of inputs, all of the same payload type.

            Returns
            -------
            dict of str to SparkPayload
                One entry per output, ``output_0`` to ``output_{num_outputs - 1}``, each of shape
                ``sample_size`` and of the payload type of the inputs.
        """
        # Control flow operation
        entries = jnp.concatenate([x.value.reshape(-1) for x in inputs.values()])
        return {f'output_{k}': self._payload_type(entries[self.indices[k]]) for k in range(self.num_outputs)}
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################