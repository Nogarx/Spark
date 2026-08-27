#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import typing as tp
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from spark.core.specs import PortSpecs
from spark.core.backend import Constant
from spark.core.utils import get_einsum_dot_exp_string
from spark.core.payloads import FloatArray, IntegerMask, SpikeArray, SparkPayload
from spark.nn.components.base import Component, ComponentConfig
from spark.nn.initializers import MaskedInitializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

PlasticityParamLike = float  | tuple[float, float, float, float] | jax.Array #| MaskedInitializer

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class PlasticityOutput(tp.TypedDict):
    """
        Output ports of a plasticity rule.

        Attributes
        ----------
        kernel : FloatArray
            The updated synaptic weights, to be written back onto the synapse.
    """
    kernel: FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class PlasticityConfig(ComponentConfig):
    """
        Base configuration for plasticity rules.

        Carries no field of its own. Concrete rules declare their own parameters.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=PlasticityConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Plasticity(Component, tp.Generic[ConfigT]):
    r"""
        Base class for plasticity rules.

        A plasticity rule reads the presynaptic spikes, the postsynaptic spikes and the current
        weights, and returns the weights for the next step. It does not own the weights: the
        result is written back onto the synapse through an effect declared in the enclosing
        controller.

        Parameters
        ----------
        config : PlasticityConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        pre_spikes : SpikeArray
            Presynaptic spikes, after any conduction delay.
        post_spikes : SpikeArray
            Postsynaptic spikes emitted on this step.
        kernel : FloatArray
            Current synaptic weights, read from the synapse.

        Output Ports
        ------------
        kernel : FloatArray
            The updated weights, written back onto the synapse as an effect.

        Notes
        -----
        A rule parameter may be given per connection type rather than as a single value. Passing
        a 4-tuple selects one value for each of the excitatory-excitatory, excitatory-inhibitory,
        inhibitory-excitatory and inhibitory-inhibitory pairs, in the order ``(EE, EI, IE, II)``.
        The pairing is read from the inhibition masks the spikes carry and is available as the
        ``synaptic_mask`` property, with ``synaptic_mask_map`` giving the names.

        Spikes are reshaped to broadcast against the kernel before the rule is evaluated: the
        presynaptic axes are placed last and the postsynaptic axes first. A rule is therefore
        written as an elementwise expression over the kernel shape.

        See Also
        --------
        HebbianRule : Pair-based potentiation.
        OjaRule : Hebbian potentiation with multiplicative normalization.
        QuadrupletRule : Four-term rule driven by an external signal.
        ThreeFactorHebbianRule : Hebbian rule gated by an external signal.
        ZenkeRule : Triplet rule with heterosynaptic and transmitter-induced terms.
        RUShortTermPlasticity : Short term depression of the current, not of the weights.
    """

    def __init__(self, config: ConfigT | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config = config, **kwargs)

    @property
    def synaptic_mask(self,) -> IntegerMask:
        if getattr(self, '_synaptic_mask', None) is None:
            return IntegerMask(jnp.array(0))
        else:
            return IntegerMask(self._synaptic_mask.value)
        
    @property
    def synaptic_mask_map(self,) -> dict[int, str]:
        return {
            0: 'EE',
            1: 'EI',
            2: 'IE',
            3: 'II',
        }

    def _initialize_synaptic_mask(self, **abc_args: SparkPayload) -> None:
        # Generate mask once
        if getattr(self, '_synaptic_mask', None) is None:
            pre_spikes: SpikeArray = abc_args['pre_spikes']
            post_spikes: SpikeArray = abc_args['post_spikes']
            # Get shapes
            pre_shape = pre_spikes.shape
            post_shape = post_spikes.shape
            # Reshape spikes to match kernel shape
            # NOTE: Post spikes should never be async
            if not pre_spikes.async_spikes:
                pre_shape = (1,) * len(post_spikes.shape) + pre_spikes.shape
                post_shape = post_spikes.shape + (1,) * len(pre_spikes.shape)
                pre_inhibition_mask = pre_spikes.inhibition_mask.reshape(pre_shape)
                post_inhibition_mask = post_spikes.inhibition_mask.reshape(post_shape)
            else:
                post_shape = post_spikes.shape + (1,) * (len(pre_spikes.shape) - len(post_spikes.shape))
                pre_inhibition_mask = pre_spikes.inhibition_mask
                post_inhibition_mask = post_spikes.inhibition_mask.reshape(post_shape)
            synaptic_mask = (
                + 0 * (1-post_inhibition_mask) * (1-pre_inhibition_mask)    # EE
                + 1 * (1-post_inhibition_mask) *    pre_inhibition_mask     # EI
                + 2 *    post_inhibition_mask  * (1-pre_inhibition_mask)    # IE
                + 3 *    post_inhibition_mask  *    pre_inhibition_mask     # II
            )
            self._synaptic_mask = Constant(synaptic_mask, dtype=jnp.uint8)

    def _initialize_variable(self, variable: PlasticityParamLike, shape: tuple, dtype: jnp.dtype) -> jax.Array:
        """
            Parses a config parameter into a JAX array matching the synapse configuration
        """
        # Resolve variable raw value
        if callable(variable):
            _var = variable(key=self.get_rng_keys(1), shape=shape, dtype=dtype)
        else:
            _var = variable
        # Construct jax.Array
        # Case 1) variable is a tuple of connection type specific values  
        if isinstance(_var, (tuple, list)) and len(_var) == 4:
            # Handle Tensor (EE, EI, IE, II) resolution
            _val = jnp.zeros_like(self._synaptic_mask.value, dtype=dtype)
            for i in range(4):
                _val = jnp.where(self._synaptic_mask.value == i, _var[i], _val)
            return _val
        # Case 2) variable is a scalar or is already an array
        elif isinstance(_var, ArrayLike):
            return jnp.array(_var, dtype=dtype)
        # Case 3) variable is who knows what 
        else:
            raise TypeError(
                f'Cannot resolve "variable" to and ArrayLike or an iterable of size four. Got: {_var}'
            )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################