#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import abc
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Any, Optional
from spark.core.shape import bShape, normalize_shape
from spark.core.variables import Variable, Constant

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class BaseTracer(nnx.Module, abc.ABC):
	name: str

	def __init__(self, 
				shape: bShape, 
				seed: Optional[int] = None, 
				dtype: Optional[Any] = jnp.float16, 
				dt: Optional[float] = 1.0,
				**kwargs):
		# Sanity checks
		if not isinstance(dt, float) and dt >= 0:
			raise ValueError(f'"dt" must be a positive float, got {dt}')
		# Initialize super.
		super().__init__(**kwargs)
		# Main attributes
		self.shape = normalize_shape(shape)
		self._seed = int.from_bytes(os.urandom(4), 'little') if seed is None else seed
		self.rng = Variable(jax.random.PRNGKey(self._seed))
		self._dtype = dtype
		self._dt = dt

	@abc.abstractmethod
	def reset(self,) -> None:
		pass

	@abc.abstractmethod
	def masked_reset(self, mask) -> None:
		pass

	@abc.abstractmethod
	def _update(self, x: jax.Array) -> jax.Array:
		pass

	@property
	@abc.abstractmethod
	def value(self, ) -> jax.Array:
		pass

	def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
		return  self._update(x)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class Tracer(BaseTracer):
	"""
		Multipurpose exponential tracer.
	"""

	def __init__(self, 
				shape: bShape, 
				tau: jax.Array, 
				scale: jax.Array = 1, 
				base: jax.Array = 0,
				**kwargs):
		# Initialize super.
		super().__init__(shape, **kwargs)
		# Main attributes
		self.scale = Constant(scale, dtype=self._dtype)
		self.base = Constant(base, dtype=self._dtype)
		self.decay = Constant(self._dt / tau, dtype=self._dtype)
		self.trace = Variable(base * jnp.ones(self.shape), dtype=self._dtype)

	def reset(self,) -> None:
		self.trace.value = self.base * jnp.ones(self.shape, dtype=self._dtype)

	def masked_reset(self, mask) -> None:
		self.trace.value = self.base * jnp.ones(self.shape, dtype=self._dtype) * mask + (1 - mask) * self.trace.value

	def _update(self, x: jax.Array) -> jax.Array:
		self.trace.value = self.trace.value - self.decay * (self.trace.value - self.base) +\
						self.scale * x.astype(self._dtype)
		return self.trace.value

	@property
	def value(self, ) -> jax.Array:
		return self.trace.value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class DoubleTracer(BaseTracer):
	"""
		Multipurpose double exponential tracer.
	"""

	def __init__(self, 
				shape: bShape, 
				tau_1: jax.Array, 
				tau_2: jax.Array,
				scale_1: jax.Array = 1, 
				scale_2: jax.Array = 1, 
				base_1: jax.Array = 0,
				base_2: jax.Array = 0,
				*args,
				**kwargs):
		# Initialize super.
		super().__init__(shape, **kwargs)
		# Main attributes
		self.scale_1 = Constant(scale_1, dtype=self._dtype)
		self.scale_2 = Constant(scale_2, dtype=self._dtype)
		self.base_1 = Constant(base_1, dtype=self._dtype)
		self.base_2 = Constant(base_2, dtype=self._dtype)
		self.decay_1 = Constant(self._dt / tau_1, dtype=self._dtype)
		self.decay_2 = Constant(self._dt / tau_2, dtype=self._dtype)
		self.trace_1 = Variable(jnp.zeros(self.shape, dtype=self._dtype), dtype=self._dtype)
		self.trace_2 = Variable(jnp.zeros(self.shape, dtype=self._dtype), dtype=self._dtype)

	def reset(self,) -> None:
		self.trace_1.value = self.base_1 * jnp.zeros(self.shape, dtype=self._dtype)
		self.trace_2.value = self.base_2 * jnp.zeros(self.shape, dtype=self._dtype)

	def masked_reset(self, mask) -> None:
		self.trace_1.value = self.base_1 * jnp.ones(self.shape, dtype=self._dtype) * mask + (1 - mask) * self.trace_1.value
		self.trace_2.value = self.base_2 * jnp.ones(self.shape, dtype=self._dtype) * mask + (1 - mask) * self.trace_2.value

	def _update(self, x: jax.Array) -> jax.Array:
		self.trace_1.value = self.trace_1.value - self.decay_1 * (self.trace_1.value - self.base_1) +\
							self.scale_1 * x.astype(self._dtype)
		self.trace_2.value = self.trace_2.value - self.decay_2 * (self.trace_2.value - self.base_2) +\
							self.scale_2 * self.trace_1.value
		return self.trace_2.value

	@property
	def value(self, ) -> jax.Array:
		return self.trace_2.value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SaturableTracer(Tracer):
	"""
		Multipurpose saturable exponential tracer.
	"""

	def _update(self, x: jax.Array) -> jax.Array:
		self.trace.value = self.trace.value - self.decay * (self.trace.value - self.base) +\
					self.scale * x.astype(self._dtype)
		self.trace.value = jnp.clip(self.trace.value, min=0, max=1)
		return self.trace.value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SaturableDoubleTracer(DoubleTracer):
	"""
		Multipurpose saturable double exponential tracer.
	"""

	def _update(self, x: jax.Array) -> jax.Array:
		self.trace_1.value = self.trace_1.value - self.decay_1 * (self.trace_1.value - self.base_1) +\
							self.scale_1 * x.astype(self._dtype)
		self.trace_2.value = self.trace_2.value - self.decay_2 * (self.trace_2.value - self.base_2) +\
							self.scale_2 * self.trace_1.value
		return self.trace_2.value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RUTracer(BaseTracer):
	"""
		Resource-Usage tracer for STP (Short Term Plasticity).
	"""
	
	def __init__(self, 
				shape: bShape, 
				R_tau: jax.Array, 
				U_tau: jax.Array, 
				scale_U: jax.Array, 
				**kwargs):
		# Initialize super.
		super().__init__(shape, **kwargs)
		# Main attributes
		self.scale_U = Constant(scale_U, dtype=self._dtype)
		self.decay_R = Constant(self._dt / R_tau, dtype=self._dtype)
		self.decay_U = Constant(self._dt / U_tau, dtype=self._dtype)
		self.trace_R = Variable(jnp.ones(self.shape), dtype=self._dtype)
		self.trace_U = Variable(jnp.zeros(self.shape), dtype=self._dtype)

	def reset(self,) -> None:
		self.trace_R.value = jnp.ones(self.shape, dtype=self._dtype)
		self.trace_U.value = jnp.zeros(self.shape, dtype=self._dtype)

	def masked_reset(self, mask) -> None:
		self.trace_R.value = jnp.ones(self.shape, dtype=self._dtype) * mask + (1 - mask) * self.trace_R.value
		self.trace_U.value = jnp.zeros(self.shape, dtype=self._dtype) * mask + (1 - mask) * self.trace_U.value

	def _update(self, x:jax.Array) -> jax.Array:
		# Update usage
		self.trace_U.value = self.trace_U.value - self.decay_U * self.trace_U.value \
				+ (1 - self.trace_U.value) * self.scale_U * x.astype(self._dtype)
		# Compute RU
		trace_RU = self.trace_U.value * self.trace_R.value
		# Update resources
		self.trace_R.value = self.trace_R.value + self.decay_R * (1 - self.trace_R.value) \
				- self.trace_U.value * (self.trace_R.value * x.astype(self._dtype))
		return trace_RU

	# NOTE: Technically this is not correct since RU is U(t) * R(t-1). 
	# This is implemented just to fullfill the specifications of a Tracer and is not intended to be used.
	# However this trace is so common in the literature that it is okay to break the rules.
	@property
	def value(self, ) -> jax.Array:
		return self.trace_R.value * self.trace_U.value
	
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################