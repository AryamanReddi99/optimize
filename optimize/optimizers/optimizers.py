from typing import Any, Optional
from optax._src import base
from optax._src import combine
from optax._src import transform
import functools
from typing import NamedTuple, Optional, Union

import chex
import jax
from jax import nn
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax.transforms import _accumulation
from optax.transforms import _adding
import optax.tree


class ScaleByAdamState(NamedTuple):
    """State for the Adam algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates


class ScaleByDoubleAdamState(NamedTuple):
    """State for the Adam algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    mu2: base.Updates
    nu: base.Updates


def adam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformationExtraArgs:

    return combine.chain(
        transform.scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        transform.scale_by_learning_rate(learning_rate),
    )


def scale_by_cautious_adam(
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
    r"""Rescale updates according to the Adam algorithm.

    See :func:`optax.adam` for more details.

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
      mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.
      nesterov: Whether to use Nesterov momentum. The variant of Adam with
        Nesterov momentum is described in [Dozat 2016]

    Returns:
      A :class:`optax.GradientTransformation` object.
    """

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
        nu = optax.tree.zeros_like(params)  # Second moment
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = optax.tree.update_moment(updates, state.mu, b1, 1)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree.map(
                lambda m, g: b1 * m + (1 - b1) * g,
                optax.tree.bias_correction(mu, b1, numerics.safe_increment(count_inc)),
                optax.tree.bias_correction(updates, b1, count_inc),
            )
        else:
            mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
        # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
        # unclear why. Other Nadam implementations also omit the extra b2 factor.
        nu_hat = optax.tree.bias_correction(nu, b2, count_inc)

        updates = jax.tree.map(
            lambda g, m, v: (
                None
                if m is None
                else m
                / (jnp.sqrt(v + eps_root) + eps)
                * jnp.where(
                    (m * g) < 0,
                    jnp.zeros_like(g, dtype=g.dtype),
                    jnp.ones_like(g, dtype=g.dtype),
                )
            ),
            updates,
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )

        mu = optax.tree.cast(mu, mu_dtype)
        nu = optax.tree.cast_like(nu, state.nu)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def cautious_adam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformationExtraArgs:
    return combine.chain(
        scale_by_cautious_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        transform.scale_by_learning_rate(learning_rate),
    )


def scale_by_cautious_double_adam(
    b11: jax.typing.ArrayLike = 0.9,
    b12: jax.typing.ArrayLike = 0.5,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment (slow)
        mu2 = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment (fast)
        nu = optax.tree.zeros_like(params)  # Second moment
        return ScaleByDoubleAdamState(
            count=jnp.zeros([], jnp.int32), mu=mu, mu2=mu2, nu=nu
        )

    def update_fn(updates, state, params=None):
        del params
        mu = optax.tree.update_moment(updates, state.mu, b11, 1)
        mu2 = optax.tree.update_moment(updates, state.mu2, b12, 1)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_increment(state.count)
        # if nesterov:
        #     mu_hat = jax.tree.map(
        #         lambda m, g: b1 * m + (1 - b1) * g,
        #         optax.tree.bias_correction(mu, b1, numerics.safe_increment(count_inc)),
        #         optax.tree.bias_correction(updates, b1, count_inc),
        #     )
        # else:
        mu_hat = optax.tree.bias_correction(mu, b11, count_inc)
        mu2_hat = optax.tree.bias_correction(mu2, b12, count_inc)

        # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
        # unclear why. Other Nadam implementations also omit the extra b2 factor.

        nu_hat = optax.tree.bias_correction(nu, b2, count_inc)

        updates = jax.tree.map(
            lambda g, m, m2, v: (
                jnp.where(
                    (m * g) > 0,
                    m,
                    jnp.where(
                        (m2 * g) > 0,
                        m2,
                        jnp.zeros_like(g, dtype=g.dtype),
                    ),
                )
                / (jnp.sqrt(v + eps_root) + eps)
            ),
            updates,
            mu_hat,
            mu2_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = optax.tree.cast(mu, mu_dtype)
        mu2 = optax.tree.cast(mu2, mu_dtype)
        nu = optax.tree.cast_like(nu, state.nu)
        return updates, ScaleByDoubleAdamState(count=count_inc, mu=mu, mu2=mu2, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def cautious_double_adam(
    learning_rate: base.ScalarOrSchedule,
    b11: float = 0.9,
    b12: float = 0.5,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformationExtraArgs:
    return combine.chain(
        scale_by_cautious_double_adam(
            b11=b11,
            b12=b12,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        transform.scale_by_learning_rate(learning_rate),
    )


def yogi(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-3,
) -> base.GradientTransformationExtraArgs:
    # pylint: enable=line-too-long
    return combine.chain(
        transform.scale_by_yogi(b1=b1, b2=b2, eps=eps),
        transform.scale_by_learning_rate(learning_rate),
    )


class ScaleByAnoState(NamedTuple):
    count: chex.Array
    mu: base.Updates
    nu: base.Updates


def scale_by_myano(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    gamma: jax.typing.ArrayLike = 1.0,
) -> base.GradientTransformation:

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
        nu = optax.tree.zeros_like(params)  # Second moment
        return ScaleByAnoState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params

        # First moment m_k: identical to Adam.
        mu = optax.tree.update_moment(updates, state.mu, b1, 1)

        # Second moment v_k: ANO update.
        #
        #   v_k = beta_2 * v_{k-1}
        #         - (1 - beta_2) * sign(v_{k-1} - g_k^2) * g_k^2
        #

        nu = jax.tree.map(
            lambda g, v: v
            - (1 - b2) * jnp.sign(v - numerics.abs_sq(g)) * numerics.abs_sq(g),
            updates,
            state.nu,
        )

        count_inc = numerics.safe_increment(state.count)

        # Bias correction for v_k.
        mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
        nu_hat = optax.tree.bias_correction(nu, b2, count_inc)

        # Normalized update. Momentum–gradient disagreement uses (m * g) < 0 so
        # the mask matches array dtype/shape; `gamma` is a hyperparameter (not
        # `params` from optax, which is the third argument to update_fn).
        updates = jax.tree.map(
            lambda g, m, v: (
                None
                if g is None
                else jnp.abs(g)
                * jnp.sign(m)
                * jnp.where(
                    (m * g) < 0,
                    jnp.asarray(gamma, dtype=g.dtype),
                    jnp.ones_like(g),
                )
                / (jnp.sqrt(v + eps_root) + eps)
            ),
            updates,
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )

        mu = optax.tree.cast(mu, mu_dtype)
        return updates, ScaleByAnoState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def myano(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-3,
    gamma: jax.typing.ArrayLike = 1.0,
) -> base.GradientTransformationExtraArgs:
    """The Yogi optimizer.

    Yogi is an adaptive optimizer, which provides control in tuning the effective
    learning rate to prevent it from increasing. By doing so, it focuses on
    addressing the issues of convergence and generalization in exponential moving
    average-based adaptive methods (such as Adam and RMSprop). Yogi is a
    modification of Adam and uses the same parameters.

    Args:
      learning_rate: A global scaling factor, either fixed or evolving along
        iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
      b1: Exponential decay rate to track the first moment of past gradients.
      b2: Exponential decay rate to track the second moment of past gradients.
      eps: A small constant applied to denominator outside of the square root (as
        in the Adam paper) to avoid dividing by zero when rescaling.

    Returns:
      The corresponding :class:`optax.GradientTransformationExtraArgs`.

    Examples:
      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
      >>> solver = optax.yogi(learning_rate=0.002)
      >>> params = jnp.array([1., 2., 3.])
      >>> print('Objective function: ', f(params))
      Objective function:  14.0
      >>> opt_state = solver.init(params)
      >>> for _ in range(5):
      ...  grad = jax.grad(f)(params)
      ...  updates, opt_state = solver.update(grad, opt_state, params)
      ...  params = optax.apply_updates(params, updates)
      ...  print('Objective function: {:.2E}'.format(f(params)))
      Objective function: 1.40E+01
      Objective function: 1.40E+01
      Objective function: 1.39E+01
      Objective function: 1.39E+01
      Objective function: 1.39E+01

    References:
      Zaheer et al, `Adaptive Methods for Nonconvex Optimization
      <https://proceedings.neurips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf>`_,
      2018
    """
    # pylint: enable=line-too-long
    return combine.chain(
        scale_by_myano(b1=b1, b2=b2, eps=eps, gamma=gamma),
        transform.scale_by_learning_rate(learning_rate),
    )
