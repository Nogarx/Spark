#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax.numpy as jnp
import dataclasses as dc
import spark.core.utils as utils
from spark.core.specs import PortSpecs
from spark.core.registry import register_interface, register_config
from spark.core.payloads import SparkPayload
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.control.base import ControlInterface, ControlInterfaceConfig, ControlInterfaceOutput, _build_signature_from_inputs

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ConcatConfig(ControlInterfaceConfig):
    """
        Configuration for `Concat`.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class Concat(ControlInterface):
    """
        Joins several inputs into one flat payload.

        Every input is flattened and concatenated in the order the ports were declared. All inputs
        must carry the same payload type, which is also the type of the result.

        Parameters
        ----------
        config : ConcatConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        **inputs : SparkPayload
            Any number of inputs, all of the same payload type. Named by the graph.

        Output Ports
        ------------
        output : SparkPayload
            The joined inputs, one dimensional and of the same payload type as the inputs.

        See Also
        --------
        ConcatReshape : The same join, followed by a reshape.
    """
    config: ConcatConfig

    def __init__(self, config: ConcatConfig | None = None, **kwargs):
		# Initialize super.
        super().__init__(config=config, **kwargs)

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

    def _overwrite_call_signature(self, raw_kwargs: dict[str, SparkPayload]) -> None:
        # Create the new Signature object and assign it to the __call__ method
        self.__call__.__func__.__signature__ = _build_signature_from_inputs(raw_kwargs)

    def __call__(self, **inputs: SparkPayload) -> ControlInterfaceOutput:
        """
            Flattens and concatenates every input.

            Parameters
            ----------
            **inputs : SparkPayload
                Any number of inputs, all of the same payload type.

            Returns
            -------
            ControlInterfaceOutput
                Dictionary with one entry, ``output``, one dimensional and of the payload type of the
                inputs.
        """
        # Control flow operation
        return {
            'output': self._payload_type(jnp.concatenate([x.value.reshape(-1) for x in inputs.values()]))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ConcatReshapeConfig(ConcatConfig):
    """
        Configuration for `ConcatReshape`.

        Parameters
        ----------
        reshape : tuple of int
            Shape of the result. Its size must match the total size of the inputs.
    """

    reshape: tuple[int, ...] = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Target shape after the merge operation.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class ConcatReshape(ControlInterface):
    """
        Joins several inputs into one payload of a given shape.

        `Concat` followed by a reshape to ``reshape``, which is what lets several one dimensional
        sources feed a module that expects a pool of a particular shape.

        Parameters
        ----------
        config : ConcatReshapeConfig
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        **inputs : SparkPayload
            Any number of inputs, all of the same payload type. Named by the graph.

        Output Ports
        ------------
        output : SparkPayload
            The joined inputs, of shape ``reshape`` and of the same payload type as the inputs.

        See Also
        --------
        Concat : The join, without the reshape.
    """
    config: ConcatReshapeConfig

    def __init__(self, config: ConcatReshapeConfig | None = None, **kwargs):
		# Initialize super.
        super().__init__(config=config, **kwargs)
        # Intialize variables.
        self.reshape = utils.validate_shape(self.config.reshape)

    def build(self, **abc_args: SparkPayload) -> None:
        # Validate payloads types.
        payload_type = None
        for key, value in abc_args.items():
            payload_type = type(value) if payload_type is None else payload_type
            if payload_type !=type(value):
                raise TypeError(
                    f'Expected all payload types to be of same type \"{payload_type}\" '
                    f'but input spec \"{key}\" is of type "{type(value)}".'
                )
        self.payload_type = payload_type
        # Validate final shape.
        try:
            jnp.concatenate([jnp.zeros(s.shape).reshape(-1) for s in abc_args.values()]).reshape(self.reshape)
        except:
            raise ValueError(f'Shapes {[s.shape for s in abc_args.values()]} are not broadcastable to {self.reshape}')

    def _overwrite_call_signature(self, raw_kwargs: dict[str, SparkPayload]) -> None:
        # Create the new Signature object and assign it to the __call__ method
        self.__call__.__func__.__signature__ = _build_signature_from_inputs(raw_kwargs)

    def __call__(self, **inputs: SparkPayload) -> ControlInterfaceOutput:
        """
            Flattens, concatenates and reshapes every input.

            Parameters
            ----------
            **inputs : SparkPayload
                Any number of inputs, all of the same payload type.

            Returns
            -------
            ControlInterfaceOutput
                Dictionary with one entry, ``output``, of shape ``reshape`` and of the payload type of
                the inputs.
        """
        # Control flow operation
        return {
            'output': self.payload_type(jnp.concatenate([x.value.reshape(-1) for x in inputs.values()]).reshape(self.reshape))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################