#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import abc
import jax
import jax.numpy as jnp
import typing as tp
from math import prod
from jax.typing import DTypeLike
import spark.core.utils as utils
from spark.core.backend import Variable, Constant
from spark.core.backend import Module

# TODO: Base constant for the rise-decay and the rise-fast-slow models are not properly set up.
# This is probably not important since practically every case is used with scale and base set to 
# one and zero, respectively. However, it would be ideal to make these tracers as general as possible.
# On the other hand, this may be important optimization for the RFSTracer, which may be used to implement
# semi-realistic synaptic models and currently uses more memory and operations that may be required.

# NOTE: Double trace (RSTracer) can be implemented as the difference between two exponentials, one fast and one slow.
# Org: (1−exp(−t/tau_rise)​) * exp(−t/tau_decay)​
# Diff: exp(−t/tau_decay)​−exp(−t/((tau_rise * tau_decay) / (tau_rise + tau_decay)))
# Simiarly the RFSTracer can be implemented as the sum of two RSTracers.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def contract_tracer_args(axes: tuple[int, ...], shape: tuple[int, ...], **values: tp.Any) -> tuple[dict[str, tp.Any], bool]:
	"""
		Reduces the arguments of a tracer over the axes its output is summed on.

		A tracer whose arguments are constant along the summed axes gives the same result when it
		is applied to the summed value instead of to each entry, which holds far less state.

		Parameters
		----------
		axes : tuple of int
			Axes the trace is summed over.
		shape : tuple of int
			Shape of the trace before the sum.
		**values : Any
			Arguments of the tracer.

		Returns
		-------
		reduced : dict of str to Any
			The arguments, contracted when possible.
		contracted : bool
			True when every argument was contracted, so the caller may build the smaller tracer.
	"""
	count = prod(shape[axis] for axis in axes)
	contracted_values = {}
	for name, value in values.items():
		value, is_constant = utils.contract_axes(value, axes, shape)
		if not is_constant:
			return dict(values), False
		contracted_values[name] = value * count if name.startswith('base') else value
	return contracted_values, True

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class BaseTracer(Module, abc.ABC):
	"""
		Base class for exponential traces.

		A tracer holds a value that decays towards a base between calls and is driven by whatever
		is passed in. Subclasses provide `_update`, which advances the trace one step.

		Parameters
		----------
		shape : tuple of int
			Shape of the trace.
		dt : float, default 1.0
			Integration step, in ms.
		dtype : DTypeLike, optional
			Dtype of the trace.
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
	r"""
		Single exponential trace.

		Parameters
		----------
		shape : tuple of int
			Shape of the trace.
		tau : jax.Array or float
			Decay constant, in ms.
		scale : jax.Array or float, default 1
			Factor applied to the incoming value.
		base : jax.Array or float, default 0
			Value the trace decays towards.

		Notes
		-----
		With :math:`\lambda = 1 - \exp(-\Delta t / \tau)`,

		.. math::
			T \leftarrow T + \lambda (T_{\mathrm{base}} - T) + c \, x

		See Also
		--------
		RDTracer : Difference of two exponentials.
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
		rate_dtype = jnp.promote_types(self._dtype, jnp.float32)
		self.decay_rate = Constant(
			-jnp.expm1(-self._dt / jnp.asarray(tau, dtype=rate_dtype)), dtype=self._dtype,
		)
		self.trace = Variable(base * jnp.ones(self.shape), dtype=self._dtype)

	def reset(self,) -> None:
		self.trace.value = self.base.value * jnp.ones(self.shape, dtype=self._dtype)

	def masked_reset(self, mask) -> None:
		self.trace.value = self.base.value * jnp.ones(self.shape, dtype=self._dtype) * mask + (1 - mask) * self.trace.value

	def _update(self, x: jax.Array) -> jax.Array:
		trace = self.trace.value
		self.trace.value = trace + self.decay_rate.value * (self.base.value - trace) + self.scale.value * x.astype(self._dtype)
		return self.trace.value

	@property
	def value(self, ) -> jax.Array:
		return self.trace.value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RDTracer(BaseTracer):
	r"""
		Rise-and-decay trace.

		The difference of two single exponentials, which rises over ``tau_rise`` and falls over
		``tau_decay`` instead of jumping on the step a value arrives.

		Parameters
		----------
		shape : tuple of int
			Shape of the trace.
		tau_rise : jax.Array or float
			Rise constant, in ms.
		tau_decay : jax.Array or float
			Decay constant, in ms.
		scale_rise, scale_decay : jax.Array or float, default 1
			Factors applied to the incoming value in each component.
		base_rise, base_decay : jax.Array or float, default 0
			Values each component decays towards.

		Notes
		-----
		The rise constant is coupled to the decay constant as

		.. math::
			\tau_r' = \frac{\tau_r \tau_d}{\tau_r + \tau_d}

		which keeps the peak at the intended height as the two constants approach each other. The
		trace is the decay component minus the rise component.

		See Also
		--------
		Tracer : Single exponential.
		RFSTracer : Rise with a fast and a slow decay.
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
	r"""
		Rise, fast decay and slow decay trace.

		A blend of two `RDTracer` traces that share a rise constant and differ in their decay
		constants, so one input leaves both a fast transient and a slow tail.

		Parameters
		----------
		shape : tuple of int
			Shape of the trace.
		alpha : jax.Array or float
			Weight of the fast component. The slow component takes ``1 - alpha``.
		tau_rise : jax.Array or float
			Rise constant shared by both components, in ms.
		tau_fast_decay, tau_slow_decay : jax.Array or float
			Decay constants of the two components, in ms.
		scale_rise, scale_fast_decay, scale_slow_decay : jax.Array or float, default 1
			Factors applied to the incoming value.
		base_rise, base_fast_decay, base_slow_decay : jax.Array or float, default 0
			Values each component decays towards.

		Notes
		-----
		.. math::
			T = \alpha T_{\mathrm{fast}} + (1 - \alpha) T_{\mathrm{slow}}

		See Also
		--------
		RDTracer : One rise and one decay constant.
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
	
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################