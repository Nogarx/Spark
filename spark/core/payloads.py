#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import abc
import numpy as np
import jax
import jax.numpy as jnp
import typing as tp
import dataclasses as dc
from jax.typing import DTypeLike
from spark.core.registry import register_payload

# NOTE: Direct jax.Array subclassing is not supported. Currently the best approach is to define __jax_array__ 
# (https://docs.jax.dev/en/latest/jep/28661-jax-array-protocol.html). However it is probable that such approach will be deprecated
# in the future (https://github.com/jax-ml/jax/issues/26764#event-16480127978). This makes creating a full fledge section of variables
# quite complicated and not worth it at this moment.
# The best we can do right now is go for a Level 1 approach (polymorphic inputs) plus a manual override of relevant operations.
# The goal eventually is get rid of all the .value and other things that may be annoying for the user in a more graceful and robust manner.

# Payloads are incredibly useful to support a GUI and several other handshakes with the user 
# at basically zero overhead cost to the framework since the default approach is to JIT compile the model.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_pytree_node_class
@dc.dataclass
class SparkPayload(abc.ABC):
    """
        Abstract payload definition to validate exchanges between SparkModule's.
    """

    def tree_flatten(self) -> tuple[tuple[jax.Array], None]:
        pass

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> tp.Self:
        pass

    @property
    def shape(self) -> tp.Any:
        pass

    @property
    def dtype(self) -> tp.Any:
        pass

    @classmethod
    @abc.abstractmethod
    def _from_spec(cls, spec: PortSpecs) -> tp.Self:
        pass

    #@abc.abstractmethod
    #def _update(self, payload: SparkPayload) -> None:
    #    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_pytree_node_class
@dc.dataclass(init=False)
class SpikeArray(SparkPayload):
    """
        Representation of a collection of spike events.

        Init:
            spikes: jax.Array[bool], True if neuron spiked, False otherwise
            inhibition_mask: jax.Array[bool], True if neuron is inhibitory, False otherwise

        The async_spikes flag is automatically set True by delay mechanisms that perform neuron-to-neuron specific delays.
        Note that when async_spikes is True the shape of the spikes changes from (origin_units,) to (origin_units, target_units).
        This is important when implementing new synaptic models, since fully valid synaptic models should be able to handle both cases. 
    """
    _encoding: jax.Array 
    async_spikes: bool = False

    # Encoding schema
    # (Spike bit, Inhibition bit)
    # 0: (False, False) ->  0
    # 1: (True,  False) ->  1
    # 2: (False, True)  -> -0
    # 3: (True,  True)  -> -1

    def __init__(self, spikes: jax.Array, inhibition_mask: jax.Array = False, async_spikes: bool = False) -> None:
        spikes = jnp.array(spikes, dtype=jnp.uint8)
        inhibition_mask = jnp.array(inhibition_mask, dtype=jnp.uint8)
        self._encoding = spikes | (inhibition_mask << 1)
        self.async_spikes = async_spikes

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self._encoding,)
        aux_data = (self.__class__,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> tp.Self:
        obj = cls.__new__(cls)
        obj._encoding = children[0]
        return obj

    def __jax_array__(self) -> jnp.ndarray: 
        return self.value
    
    def __array__(self, dtype=None) -> np.ndarray: 
        return np.array(self.value).astype(dtype if dtype else jnp.float16)

    @property
    def spikes(self) -> jax.Array:
        # Extract Bit 0
        return (self._encoding & 1).astype(bool)

    @property
    def inhibition_mask(self) -> jax.Array:
        # Extract Bit 1
        return ((self._encoding >> 1) & 1).astype(bool)

    @property
    def value(self) -> jax.Array:
        return jnp.where(self.inhibition_mask, jnp.float16(-1), jnp.float16(1)) * self.spikes

    @property
    def shape(self) -> tuple[int, ...]:
        return self._encoding.shape

    @property
    def dtype(self) -> DTypeLike:
        return jnp.float16

    @classmethod
    def _from_spec(cls, spec: PortSpecs) -> tp.Self:
        obj = cls.__new__(cls)
        spikes = jnp.zeros(spec.shape, dtype=jnp.uint8)
        if isinstance(spec.inhibition_mask, jax.Array):
            inhibition_mask = jnp.array(spec.inhibition_mask, dtype=jnp.uint8)
        elif isinstance(spec.inhibition_mask, bool):
            inhibition_mask = spec.inhibition_mask * jnp.ones(spec.shape, dtype=jnp.uint8)
        else:
            inhibition_mask = jnp.zeros(spec.shape, dtype=jnp.uint8)
        obj._encoding = spikes | (inhibition_mask << 1)
        obj.async_spikes = spec.async_spikes
        return obj 

    @classmethod
    def _from_encoding(cls, encoding: jax.Array) -> tp.Self:
        obj = cls.__new__(cls)
        obj._encoding = jnp.array(encoding, dtype=jnp.uint8)
        return obj 
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dc.dataclass
class ValueSparkPayload(SparkPayload, abc.ABC):
    """
        Abstract payload definition to single value payloads.
    """
    value: jnp.ndarray

    def __jax_array__(self) -> jnp.ndarray: 
        return self.value
    
    def __array__(self, dtype=None) -> np.ndarray: 
        return np.array(self.value).astype(dtype if dtype else self.value.dtype)

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.value,)
        aux_data = (self.__class__,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> tp.Self:
        obj = cls(children[0])
        return obj

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.value.dtype

    @classmethod
    def _from_spec(cls, spec: PortSpecs) -> tp.Self:
        return cls(jnp.zeros(spec.shape, dtype=spec.dtype))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_pytree_node_class
@dc.dataclass
class CurrentArray(ValueSparkPayload):
    """
        Representation of a collection of currents.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_pytree_node_class
@dc.dataclass
class PotentialArray(ValueSparkPayload):
    """
        Representation of a collection of membrane potentials.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_pytree_node_class
@dc.dataclass
class BooleanMask(ValueSparkPayload):
    """
        Representation of an inhibitory boolean mask.
    """
    pass
        
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_pytree_node_class
@dc.dataclass
class IntegerMask(ValueSparkPayload):
    """
        Representation of an integer mask.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_pytree_node_class
@dc.dataclass
class FloatArray(ValueSparkPayload):
    """
        Representation of a float array.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_pytree_node_class
@dc.dataclass
class IntegerArray(ValueSparkPayload):
    """
        Representation of an integer array.
    """
    pass
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

