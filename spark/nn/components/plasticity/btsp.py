#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import jax
import jax.numpy as jnp
import typing as tp
import dataclasses as dc


from spark.nn.initializers.base import Initializer
from spark.core.tracers import Tracer
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.core.payloads import BooleanMask, SpikeArray, FloatArray
from spark.core.backend import Constant
from spark.core.registry import register_module, register_config
from spark.nn.components.plasticity.base import (
    Plasticity, PlasticityConfig, PlasticityOutput, PlasticityParamLike,
)
from spark.nn.components.plasticity.modulated import ModulatedPlasticity, ModulatedPlasticityConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class BTSPRuleConfig(PlasticityConfig):
    """
        Configuration for `BTSPRule`.

        Every parameter may be given as a single value or as a 4-tuple holding one value per
        connection type, in the order ``(EE, EI, IE, II)``.

        Parameters
        ----------
        et_tau : PlasticityParamLike or Initializer, default 863.91
            Decay constant of the eligibility trace, in ms.
        et_scale : PlasticityParamLike or Initializer, default 0.05
            Increment added to the eligibility trace by each presynaptic spike. Pick it so the
            trace peaks near 1 at the presynaptic rates in use: a rate ``f`` sustained for
            longer than ``et_tau`` settles near ``et_scale * f * et_tau / 1000``.
        is_tau : PlasticityParamLike or Initializer, default 542.76
            Decay constant of the instructive signal, in ms.
        is_scale : PlasticityParamLike or Initializer, default 2.36
            Gain on the plateau entering the instructive signal. The default normalizes a
            300 ms plateau to a peak of 1, which is the plateau duration the rule was fitted
            on. For a plateau of duration ``D`` use ``1 / (1 - exp(-D / is_tau))``.
        w_max : PlasticityParamLike or Initializer, default 4.02
            Upper bound on the weights, and the target the potentiation term pulls towards. The
            fitted value is normalized, in multiples of the initial weight, while a synaptic
            kernel is in pA. Scale it into the units of the kernel it bounds, or the first step
            clips the whole kernel down to it.
        k_pot : PlasticityParamLike or Initializer, default 2.27
            Potentiation rate, per second.
        k_dep : PlasticityParamLike or Initializer, default 0.33
            Depression rate, per second.
        k_neg : PlasticityParamLike or Initializer, default 2.27
            Depression rate under a negative instructive sign, per second. Read only when the
            ``sign`` port is wired. Defaults to the potentiation rate, so that one negative
            event unlearns as much as one positive event learns.
        alpha_pot : PlasticityParamLike or Initializer, default 0.24
            Half maximum of the potentiation sigmoid, in units of the overlap.
        beta_pot : PlasticityParamLike or Initializer, default 30.32
            Slope of the potentiation sigmoid.
        alpha_dep : PlasticityParamLike or Initializer, default 0.09
            Half maximum of the depression sigmoid, in units of the overlap.
        beta_dep : PlasticityParamLike or Initializer, default 2260.61
            Slope of the depression sigmoid. The fitted value is steep enough that depression
            is effectively switched on at ``alpha_dep``.
        eta : float or jax.Array, default 1.0
            Overall gain on the update. The rates are carried by ``k_pot`` and ``k_dep``, so
            this is left at one unless the whole rule is being scaled.
    """

    et_tau: PlasticityParamLike | Initializer = dc.field(
        default = 863.91,
        metadata = {
            'units': 'ms',
            'validators': [TypeValidator, PositiveValidator],
            'description': 'Decay constant of the eligibility trace.',
        })
    et_scale: PlasticityParamLike | Initializer = dc.field(
        default = 0.05,
        metadata = {
            'validators': [TypeValidator],
            'description': 'Eligibility trace increment per presynaptic spike.',
        })
    is_tau: PlasticityParamLike | Initializer = dc.field(
        default = 542.76,
        metadata = {
            'units': 'ms',
            'validators': [TypeValidator, PositiveValidator],
            'description': 'Decay constant of the instructive signal.',
        })
    is_scale: PlasticityParamLike | Initializer = dc.field(
        default = 2.36,
        metadata = {
            'validators': [TypeValidator],
            'description': 'Gain on the plateau entering the instructive signal.',
        })
    w_max: PlasticityParamLike | Initializer = dc.field(
        default = 4.02,
        metadata = {
            'validators': [TypeValidator, PositiveValidator],
            'description': 'Maximum synaptic weight.',
        })
    k_pot: PlasticityParamLike | Initializer = dc.field(
        default = 2.27,
        metadata = {
            'units': '1/s',
            'validators': [TypeValidator],
            'description': 'Potentiation rate.',
        })
    k_dep: PlasticityParamLike | Initializer = dc.field(
        default = 0.33,
        metadata = {
            'units': '1/s',
            'validators': [TypeValidator],
            'description': 'Depression rate.',
        })
    k_neg: PlasticityParamLike | Initializer = dc.field(
        default = 2.27,
        metadata = {
            'units': '1/s',
            'validators': [TypeValidator],
            'description': 'Depression rate under a negative instructive sign.',
        })
    et_post_tau: float = dc.field(
        default = 0.0,
        metadata = {
            'units': 'ms',
            'validators': [TypeValidator],
            'description': 'Window of the pre-post coincidence laid into the eligibility trace; 0 lays it down from presynaptic spikes alone, as fitted.',
        })
    alpha_pot: PlasticityParamLike | Initializer = dc.field(
        default = 0.24,
        metadata = {
            'validators': [TypeValidator],
            'description': 'Half maximum of the potentiation sigmoid.',
        })
    beta_pot: PlasticityParamLike | Initializer = dc.field(
        default = 30.32,
        metadata = {
            'validators': [TypeValidator],
            'description': 'Slope of the potentiation sigmoid.',
        })
    alpha_dep: PlasticityParamLike | Initializer = dc.field(
        default = 0.09,
        metadata = {
            'validators': [TypeValidator],
            'description': 'Half maximum of the depression sigmoid.',
        })
    beta_dep: PlasticityParamLike | Initializer = dc.field(
        default = 2260.61,
        metadata = {
            'validators': [TypeValidator],
            'description': 'Slope of the depression sigmoid.',
        })
    eta: float | jax.Array = dc.field(
        default = 1.0,
        metadata = {
            'validators': [TypeValidator],
            'description': 'Learning rate.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class BTSPRule(Plasticity):
    r"""
        Behavioral timescale synaptic plasticity.

        Weights are moved by the overlap of two slow signals, both held by the rule: an
        eligibility trace laid down locally by presynaptic activity, and an instructive signal
        laid down by the dendritic plateau. Both last on the order of a second, so an input
        active seconds before or after a plateau is still modified by it.

        The update is bidirectional and depends on the weight rather than on postsynaptic
        firing: weak synapses potentiate towards ``w_max`` and strong ones depress towards
        zero, which makes the rule stable without a separate homeostatic term.

        The plateau carries no sign. An optional ``sign`` port gates the rule with the sign
        of the instructive event: positive runs the fitted rule, negative depresses every
        eligible input, zero leaves the weights alone.

        Parameters
        ----------
        config : BTSPRuleConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        modulation : FloatArray
            The third factor, in ``[0, 1]``. Wire the ``plateau`` port of a dendrite onto it.
            It drives the instructive signal.
        pre_spikes : SpikeArray
            Presynaptic spikes, after any conduction delay. They drive the eligibility trace.
        kernel : FloatArray
            Current synaptic weights, read from the synapse.
        sign : FloatArray, optional
            Sign of the instructive event, in ``[-1, 1]``, one entry per postsynaptic unit or
            a scalar. Values in between scale the gated terms, so pass the sign alone for a
            hard gate. Without the port the rule is unsigned, as fitted.
        inhibition_mask : BooleanMask, optional
            Marks the inhibitory units of the pool. Supplied by the enclosing `Neuron` without
            being wired, and read only to type the connections. The update never reads the
            postsynaptic state, which is what separates BTSP from a Hebbian rule.

        Output Ports
        ------------
        kernel : FloatArray
            The updated weights, written back onto the synapse as an effect.

        Notes
        -----
        The rule holds both traces: the eligibility one per presynaptic unit, the instructive
        one per dendrite.

        .. math::
            \tau_{ET} \frac{dET}{dt} = -ET + \lambda_{ET} R \\
            \tau_{IS} \frac{dIS}{dt} = -IS + \lambda_{IS} P

        The two are driven differently because their inputs are. Presynaptic spikes are events,
        so each one adds ``et_scale`` to the eligibility trace. The plateau is a sustained
        level, so the instructive signal relaxes towards ``is_scale`` times it.

        Their pointwise product is the overlap that drives both processes:

        .. math::
            \frac{dW}{dt} = (W_{\max} - W) \, k_+ \, q_+(ET \cdot IS)
                          - W \, k_- \, q_-(ET \cdot IS)

        with :math:`q_\pm` a sigmoid rescaled to pass through 0 at an overlap of 0 and 1 at an
        overlap of 1:

        .. math::
            q(x) = \frac{\sigma(\beta (x - \alpha)) - \sigma(-\beta \alpha)}
                        {\sigma(\beta (1 - \alpha)) - \sigma(-\beta \alpha)}

        The overlap is clipped to ``[0, 1]``, which is the range the sigmoids were fitted on.
        The rates are given per second and the weights are bounded to ``[0, w_max]``. Both
        ``lambda`` factors are free normalizations, chosen so neither trace exceeds 1.

        For a constant overlap :math:`x` the weight relaxes to the fixed point

        .. math::
            W^*(x) = W_{\max} \frac{k_+ q_+(x)}{k_+ q_+(x) + k_- q_-(x)}

        at rate :math:`k_+ q_+(x) + k_- q_-(x)`. The target rises with the overlap, so an
        input active close to the plateau is pulled towards ``w_max`` and an eligible input far
        from it towards a low weight, whatever the postsynaptic activity. This is what keeps
        the total drive bounded without a homeostatic term.

        With the ``sign`` port wired, the sign :math:`s` gates the terms:

        .. math::
            \frac{dW}{dt} = [s]_+ \left( (W_{\max} - W) k_+ q_+ - W k_- q_- \right)
                          - [s]_- \, W k_{\mathrm{neg}} \, q_+

        A negative event therefore depresses eligible inputs in proportion to their weight,
        following the potentiation profile so that the inputs closest to the event lose the
        most. A zero sign gates plasticity off.

        The reference implementation applies each induction lap as one step with the weight
        held fixed during the lap. Here the equation is integrated every step, so one plateau
        closes a fraction :math:`1 - \exp(-k \int q \, dt)` of the gap to the target rather
        than :math:`k \int q \, dt` of it, and cannot overshoot.

        The modulation and the sign are expected in a shape that broadcasts against the
        kernel: a scalar, the postsynaptic shape, or the kernel shape itself.

        References
        ----------
        .. [1] A. D. Milstein et al., "Bidirectional synaptic plasticity rapidly modifies
               hippocampal representations", eLife 10, e73046, 2021.
               https://doi.org/10.7554/eLife.73046
    """
    config: BTSPRuleConfig

    def __init__(self, config: BTSPRuleConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(
            self,
            modulation: FloatArray,
            pre_spikes: SpikeArray,
            kernel: FloatArray,
            sign: FloatArray | None = None,
            inhibition_mask: BooleanMask | None = None,
            post_spikes: SpikeArray | None = None,
        ) -> None:
        kernel_shape = kernel.shape
        pre_shape = pre_spikes.shape
        modulation_shape = modulation.shape
        sign_shape = modulation_shape if sign is None else sign.shape
        _init = lambda var: self._initialize_variable(var, shape=kernel_shape, dtype=kernel.dtype)
        _et_tau = _init(self.config.init.et_tau)
        _et_scale = _init(self.config.init.et_scale)
        _is_tau = _init(self.config.init.is_tau)
        _is_scale = _init(self.config.init.is_scale)
        # Broadcast shapes.
        self._pre_shape = pre_shape if pre_shape == kernel_shape else (1,) * (len(kernel_shape) - len(pre_shape)) + pre_shape
        self._modulation_shape = modulation_shape if modulation_shape == kernel_shape else modulation_shape + (1,) * (len(kernel_shape) - len(modulation_shape))
        self._sign_shape = sign_shape if sign_shape == kernel_shape else sign_shape + (1,) * (len(kernel_shape) - len(sign_shape))
        # Eligibility trace.
        self._coincidence = float(self.config.et_post_tau) > 0.0 and post_spikes is not None
        if self._coincidence:
            post_shape = post_spikes.shape
            self._post_shape = post_shape if post_shape == kernel_shape else post_shape + (1,) * (len(kernel_shape) - len(post_shape))
            self.pre_window = Tracer(self._pre_shape, tau=self.config.et_post_tau, scale=1.0, base=0.0, dt=self._dt, dtype=self._dtype)
            self.post_window = Tracer(self._post_shape, tau=self.config.et_post_tau, scale=1.0, base=0.0, dt=self._dt, dtype=self._dtype)
            self.eligibility = Tracer(kernel_shape, tau=_et_tau, scale=_et_scale, base=0.0, dt=self._dt, dtype=self._dtype)
        else:
            self.eligibility = Tracer(self._pre_shape, tau=_et_tau, scale=_et_scale, base=0.0,
                                      dt=self._dt, dtype=self._dtype)
        # Instructive signal.
        _is_gain = -jnp.expm1(-self._dt / jnp.asarray(_is_tau, dtype=jnp.float32))
        self.instructive = Tracer(self._modulation_shape, tau=_is_tau,
                                  scale=(_is_gain * _is_scale).astype(self._dtype), base=0.0,
                                  dt=self._dt, dtype=self._dtype)
        # Rates are given per second, the step is in ms.
        self.k_pot = Constant(_init(self.config.init.k_pot) / 1000.0, dtype=self._dtype)
        self.k_dep = Constant(_init(self.config.init.k_dep) / 1000.0, dtype=self._dtype)
        self.k_neg = Constant(_init(self.config.init.k_neg) / 1000.0, dtype=self._dtype)
        self.w_max = Constant(_init(self.config.init.w_max), dtype=self._dtype)
        self.eta = Constant(self.config.eta, dtype=self._dtype)
        # Saturating sigmoids, rescaled to pass through (0, 0) and (1, 1).
        for name in ('pot', 'dep'):
            alpha = _init(getattr(self.config.init, f'alpha_{name}'))
            beta = _init(getattr(self.config.init, f'beta_{name}'))
            low = jax.nn.sigmoid(beta * (0.0 - alpha))
            high = jax.nn.sigmoid(beta * (1.0 - alpha))
            setattr(self, f'alpha_{name}', Constant(alpha, dtype=self._dtype))
            setattr(self, f'beta_{name}', Constant(beta, dtype=self._dtype))
            setattr(self, f'{name}_low', Constant(low, dtype=self._dtype))
            setattr(self, f'{name}_span', Constant(high - low, dtype=self._dtype))

    def _initialize_synaptic_mask(self, **abc_args: FloatArray) -> None:
        """
            Types every connection as EE, EI, IE or II.

            NOTE: This rule has no postsynaptic spike port. The presynaptic side
            comes from the spikes, the postsynaptic side from the inhibition mask of the pool.
        """
        if getattr(self, '_synaptic_mask', None) is not None:
            return
        pre_spikes: SpikeArray = abc_args['pre_spikes']
        kernel: FloatArray = abc_args['kernel']
        mask: BooleanMask | None = abc_args.get('inhibition_mask', None)
        kernel_shape = kernel.shape
        pre_shape = pre_spikes.shape
        pre_shape = pre_shape if pre_shape == kernel_shape else (1,) * (len(kernel_shape) - len(pre_shape)) + pre_shape
        pre_inhibition_mask = pre_spikes.inhibition_mask.reshape(pre_shape).astype(jnp.uint8)
        if mask is None:
            post_inhibition_mask = jnp.zeros((), dtype=jnp.uint8)
        else:
            mask_shape = mask.shape
            mask_shape = mask_shape if mask_shape == kernel_shape else mask_shape + (1,) * (len(kernel_shape) - len(mask_shape))
            post_inhibition_mask = jnp.asarray(mask.value, dtype=jnp.uint8).reshape(mask_shape)
        synaptic_mask = (
            + 0 * (1 - post_inhibition_mask) * (1 - pre_inhibition_mask)    # EE
            + 1 * (1 - post_inhibition_mask) *      pre_inhibition_mask     # EI
            + 2 *      post_inhibition_mask  * (1 - pre_inhibition_mask)    # IE
            + 3 *      post_inhibition_mask  *      pre_inhibition_mask     # II
        )
        self._synaptic_mask = Constant(
            jnp.broadcast_to(synaptic_mask, kernel_shape), dtype=jnp.uint8)

    def reset(self) -> None:
        """
            Resets component state.
        """
        self.eligibility.reset()
        self.instructive.reset()
        if self._coincidence:
            self.pre_window.reset()
            self.post_window.reset()

    def _clip_kernel(self, kernel: jax.Array) -> jax.Array:
        """
            The bounds applied to the weights after the update.
        """
        return jnp.clip(kernel, min=0.0, max=self.w_max.value)

    def _saturating(self, overlap: jax.Array, name: str) -> jax.Array:
        """
            Sigmoid of the overlap, rescaled to pass through (0, 0) and (1, 1).
        """
        raw = jax.nn.sigmoid(getattr(self, f'beta_{name}').value * (overlap - getattr(self, f'alpha_{name}').value))
        return (raw - getattr(self, f'{name}_low').value) / getattr(self, f'{name}_span').value

    def _kernel_delta(
            self,
            modulation: FloatArray,
            pre_spikes: SpikeArray,
            kernel: FloatArray,
            sign: FloatArray | None = None,
            inhibition_mask: BooleanMask | None = None,
            post_spikes: SpikeArray | None = None,
        ) -> jax.Array:
        """
            Kernel delta computation.
        """
        _pre_spikes = pre_spikes.spikes.reshape(self._pre_shape).astype(self._dtype)
        _modulation = modulation.value.reshape(self._modulation_shape)
        _kernel = kernel.value
        if self._coincidence:
            _post_spikes = post_spikes.spikes.reshape(self._post_shape).astype(self._dtype)
            coincidence = _pre_spikes * self.post_window(_post_spikes) + _post_spikes * self.pre_window(_pre_spikes)
            eligibility = self.eligibility(coincidence)
        else:
            eligibility = self.eligibility(_pre_spikes)
        instructive = self.instructive(_modulation)
        overlap = jnp.clip(eligibility * instructive, 0.0, 1.0)
        near = self._saturating(overlap, 'pot')
        potentiation = (self.w_max.value - _kernel) * self.k_pot.value * near
        depression = _kernel * self.k_dep.value * self._saturating(overlap, 'dep')
        if sign is None:
            return potentiation - depression
        _sign = sign.value.reshape(self._sign_shape).astype(self._dtype)
        positive = jnp.clip(_sign, 0.0, 1.0)
        negative = jnp.clip(-_sign, 0.0, 1.0)
        return (
            + positive * (potentiation - depression)
            - negative * _kernel * self.k_neg.value * near
        )

    def __call__(
            self,
            modulation: FloatArray,
            pre_spikes: SpikeArray,
            kernel: FloatArray,
            sign: FloatArray | None = None,
            inhibition_mask: BooleanMask | None = None,
            post_spikes: SpikeArray | None = None,
        ) -> PlasticityOutput:
        """
            Computes the weights for the next step.

            Parameters
            ----------
            modulation : FloatArray
                The third factor, in ``[0, 1]``. Wire a dendritic ``plateau`` onto it.
            pre_spikes : SpikeArray
                Presynaptic spikes, after any conduction delay.
            kernel : FloatArray
                Current synaptic weights.
            sign : FloatArray, optional
                Sign of the instructive event, in ``[-1, 1]``. Gates the rule.
            inhibition_mask : BooleanMask, optional
                Marks the inhibitory units of the pool. Not read by the update.
            post_spikes : SpikeArray, optional
                Postsynaptic spikes. Read only when ``et_post_tau`` > 0, for the coincidence
                eligibility.

            Returns
            -------
            PlasticityOutput
                Dictionary with one entry, ``kernel``, the updated weights.
        """
        delta = self._kernel_delta(modulation, pre_spikes, kernel, sign, inhibition_mask, post_spikes)
        return self._apply_delta(kernel, self._learning_rate() * delta)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ModulatedBTSPRuleConfig(ModulatedPlasticityConfig, BTSPRuleConfig):
    """
        Configuration for `ModulatedBTSPRule`.

        Union of `BTSPRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of its
        own, so the defaults are those of `BTSPRuleConfig`.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ModulatedBTSPRule(ModulatedPlasticity, BTSPRule):
    r"""
        Three-factor BTSP rule.

        `BTSPRule` composed with `ModulatedPlasticity`.

        See Also
        --------
        BTSPRule : The same rule without the third factor.
        ModulatedPlasticity : The mixin supplying the third factor.
    """
    config: ModulatedBTSPRuleConfig

    def __init__(self, config: ModulatedBTSPRuleConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################