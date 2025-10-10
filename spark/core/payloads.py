#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

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

@jax.tree_util.register_dataclass
@dc.dataclass
class SparkPayload(abc.ABC):
    """
        Abstract payload definition to validate exchanges between SparkModule's.
    """
    pass

    @property
    def shape(self) -> tp.Any:
        pass

    @property
    def dtype(self) -> tp.Any:
        pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_dataclass
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

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.value.dtype

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_dataclass
@dc.dataclass
class SpikeArray(ValueSparkPayload):
    """
        Representation of a collection of spike events.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_dataclass
@dc.dataclass
class CurrentArray(ValueSparkPayload):
    """
        Representation of a collection of currents.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_dataclass
@dc.dataclass
class PotentialArray(ValueSparkPayload):
    """
        Representation of a collection of membrane potentials.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_dataclass
@dc.dataclass
class BooleanMask(ValueSparkPayload):
    """
        Representation of an inhibitory boolean mask.
    """
    pass
        
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_dataclass
@dc.dataclass
class IntegerMask(ValueSparkPayload):
    """
        Representation of an integer mask.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_dataclass
@dc.dataclass
class FloatArray(ValueSparkPayload):
    """
        Representation of a float array.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_payload
@jax.tree_util.register_dataclass
@dc.dataclass
class IntegerArray(ValueSparkPayload):
    """
        Representation of an integer array.
    """
    pass
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

