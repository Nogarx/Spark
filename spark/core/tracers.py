#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import abc
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import typing as tp
from jax.typing import DTypeLike
from spark.core.variables import Variable, Constant

# TODO: Base constant for the rise-decay and the rise-fast-slow models are not properly set up.
# This is probably not important since practically every case is used with scale and base set to 
# one and zero, respectively. However, it would be ideal to make this tracers as general as possible.
# On the other hand, this may be important optimization for the RFSTracer, which may be used to implement
# semi-realistic synaptic models and currently uses more memory and operations that may be required.

# NOTE: Double trace (RSTracer) can be implemented as the difference between two exponentials, one fast and one slow.
# Org: (1−exp(−t/tau_rise)​) * exp(−t/tau_decay)​
# Diff: exp(−t/tau_decay)​−exp(−t/((tau_rise * tau_decay) / (tau_rise + tau_decay)))
# Simiarly the RFSTracer can be implemented as the sum of two RSTracers.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class BaseTracer(nnx.Module, abc.ABC):
	"""
		Base Tracer class
	"""

	def __init__(
			self, 
			shape: tuple[int, ...], 
			seed: int | None = None, 
			dtype: DTypeLike = jnp.float16, 
			dt: float = 1.0,
			**kwargs
		):
		# Sanity checks
		if not isinstance(dt, float) or dt < 0:
			raise ValueError(f'"dt" must be a positive float, got {dt}')
		# Initialize super.
		super().__init__(**kwargs)
		# Main attributes
		self.shape = shape
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

	def __init__(
			self, 
			shape: tuple[int, ...], 
			tau: jax.Array | float, 
			scale: jax.Array | float = 1, 
			base: jax.Array | float = 0,
			**kwargs
		) -> None:
		# Initialize super.
		super().__init__(shape, **kwargs)
		# Main attributes
		self.scale = Constant(scale, dtype=self._dtype)
		self.base = Constant(base, dtype=self._dtype)
		self.decay = Constant(jnp.exp(-self._dt / tau), dtype=self._dtype)
		self.trace = Variable(base * jnp.ones(self.shape), dtype=self._dtype)

	def reset(self,) -> None:
		self.trace.value = self.base * jnp.ones(self.shape, dtype=self._dtype)

	def masked_reset(self, mask) -> None:
		self.trace.value = self.base * jnp.ones(self.shape, dtype=self._dtype) * mask + (1 - mask) * self.trace.value

	def _update(self, x: jax.Array) -> jax.Array:
		self.trace.value = self.base + self.decay * (self.trace.value - self.base) + self.scale * x.astype(self._dtype)
		return self.trace.value

	@property
	def value(self, ) -> jax.Array:
		return self.trace.value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RDTracer(BaseTracer):
	"""
		Rise-Decay Tracer.

		Multipurpose double exponential tracer.
	"""

	def __init__(
			self, 
			shape: tuple[int, ...], 
			tau_rise: jax.Array | float, 
			tau_decay: jax.Array | float,
			scale_rise: jax.Array | float = 1, 
			scale_decay: jax.Array | float = 1, 
			base_rise: jax.Array | float = 0,
			base_decay: jax.Array | float = 0,
			**kwargs
		) -> None:
		# Initialize super.
		super().__init__(shape, **kwargs)
		# Tau's coupling factor
		tau_rise = (tau_rise * tau_decay) / (tau_rise + tau_decay)
		# Main attributes
		self.tracer_rise = Tracer(
			shape=shape, tau=tau_rise, scale=scale_rise, base=base_rise, **kwargs
		)
		self.tracer_decay = Tracer(
			shape=shape, tau=tau_decay, scale=scale_decay, base=base_decay, **kwargs
		)

	def reset(self,) -> None:
		self.tracer_rise.reset()
		self.tracer_decay.reset()

	def masked_reset(self, mask) -> None:
		self.tracer_rise.masked_reset(mask)
		self.tracer_decay.masked_reset(mask)

	def _update(self, x: jax.Array) -> jax.Array:
		trace_rise = self.tracer_rise(x)
		trace_decay = self.tracer_decay(x)
		return trace_decay - trace_rise

	@property
	def value(self, ) -> jax.Array:
		return self.tracer_decay.value - self.tracer_rise.value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RFSTracer(BaseTracer):
	"""
		Rise-Fast-Slow Tracer

		Multipurpose triple exponential tracer.
	"""

	def __init__(
			self, 
			shape: tuple[int, ...], 
			alpha: jax.Array | float,
			tau_rise: jax.Array | float, 
			tau_fast_decay: jax.Array | float,
			tau_slow_decay: jax.Array | float,
			scale_rise: jax.Array | float = 1, 
			scale_fast_decay: jax.Array | float = 1, 
			scale_slow_decay: jax.Array | float = 1, 
			base_rise: jax.Array | float = 0,
			base_fast_decay: jax.Array | float = 0,
			base_slow_decay: jax.Array | float = 0,
			**kwargs
		) -> None:
		# Initialize super.
		super().__init__(shape, **kwargs)
		# TODO: The easiest way to implement the RFS tracer is by means of a difference of two RDTracer's.
		# However this is likely to not be optimal and may consume a large amount of memory with large arrays.
		self.tracer_rise_fast = RDTracer(
			shape=shape, 
			tau_rise=tau_rise, 
			tau_decay=tau_fast_decay,
			scale_rise=scale_rise, 
			scale_decay=scale_fast_decay, 
			base_rise=base_rise,
			base_decay=base_fast_decay,
			**kwargs
		)
		self.tracer_rise_slow = RDTracer(
			shape=shape, 
			tau_rise=tau_rise, 
			tau_decay=tau_slow_decay,
			scale_rise=scale_rise, 
			scale_decay=scale_slow_decay, 
			base_rise=base_rise,
			base_decay=base_slow_decay,
			**kwargs
		)
		self.alpha = Constant(alpha, dtype=self._dtype)

	def reset(self,) -> None:
		self.tracer_rise_fast.reset()
		self.tracer_rise_slow.reset()

	def masked_reset(self, mask) -> None:
		self.tracer_rise_fast.masked_reset(mask)
		self.tracer_rise_slow.masked_reset(mask)

	def _update(self, x: jax.Array) -> jax.Array:
		tracer_rise_slow = self.tracer_rise_slow(x)
		tracer_rise_fast = self.tracer_rise_fast(x)
		return self.alpha.value * tracer_rise_fast + (1 - self.alpha.value) * tracer_rise_slow

	@property
	def value(self, ) -> jax.Array:
		return self.alpha.value * self.tracer_rise_fast.value + (1 - self.alpha.value) * self.tracer_rise_slow.value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: Validate tracer
class RUTracer(BaseTracer):
	"""
		Resource-Usage tracer for STP (Short Term Plasticity).
	"""
	
	def __init__(
			self, 
			shape: tuple[int, ...], 
			r_tau: jax.Array | float, 
			u_tau: jax.Array | float, 
			u_scale: jax.Array | float, 
			**kwargs
		):
		# Initialize super.
		super().__init__(shape, **kwargs)
		# Main attributes
		self.r_tracer = Tracer(shape=shape, tau=r_tau, scale=-1.0, base=1.0, **kwargs)
		self.u_tracer = Tracer(shape=shape, tau=u_tau, scale=u_scale, base=0.0, **kwargs)

	def reset(self,) -> None:
		self.r_tracer.reset()
		self.u_tracer.reset()

	def masked_reset(self, mask) -> None:
		self.r_tracer.masked_reset(mask)
		self.u_tracer.masked_reset(mask)

	def _update(self, x:jax.Array) -> jax.Array:
		# Update usage
		u_trace = self.u_tracer(x)
		# Compute RU
		trace_RU = u_trace * self.r_tracer.value
		# Update resources
		self.r_tracer.update(u_trace * self.r_tracer.value * x)
		return trace_RU

	# NOTE: Technically, this is not correct since RU is U(t) * R(t-1). 
	# This is implemented just to fullfill the specifications of a Tracer and is not intended to be used.
	# However this trace is so common in the literature that it is okay to break the rules.
	@property
	def value(self, ) -> jax.Array:
		return self.r_tracer.value * self.u_tracer.value
	
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################