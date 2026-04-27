"""Ten Adam steps in a `lax.scan`; collect time series of diagnostics.

After `run_gradient_descent_scan()`, `history` concatenates both scan stages along
time; the leading axis length is ``NUM_STEPS`` from the first scan plus the length
of the second scan (e.g. ``2 * NUM_STEPS`` when both scans use ``length=NUM_STEPS``).

- ``loss``: shape ``(NUM_STEPS,)``
- ``bc1``, ``bc2``: bias-correction denominators ``1 - b1^t``, ``1 - b2^t``
- ``mu``, ``nu``, ``mu_hat``, ``nu_hat``, ``grads``, ``params``: same pytree
  structure as params; each leaf has shape ``(NUM_STEPS,) + leaf_shape`` (``params``
  stores state after each optimizer step)
"""

import jax
import jax.numpy as jnp
import optax

from optimize.utils.jax_utils import pytree_norm
from optimize.optimizers.optimizers import adam, yogi, myano
import matplotlib.pyplot as plt

B2 = 0.999
LR = 1.0
# Grid of β₁ values for the multi-panel figure in ``__main__``
BETA1_VALUES = (0.5, 0.7, 0.9, 0.99)


def bias_correction_denom(decay: float, count: jax.Array) -> jax.Array:
    """Scalar 1 - decay**t, matching Optax bias correction denominator."""
    c = count.astype(jnp.float32)
    return jnp.asarray(1.0, jnp.float32) - (jnp.asarray(decay, jnp.float32) ** c)


def loss_fun(params, m1: float = 1.0, m2: float = 1.0):
    return (params["a1"] * m1 + params["a2"] * m2).mean()


def _concat_history_stages(a: dict, b: dict) -> dict:
    """Merge two scan histories by concatenating each leaf along time axis 0."""
    if set(a) != set(b):
        raise ValueError("history dicts must have the same keys")

    def _cat(x, y):
        if isinstance(x, dict) and isinstance(y, dict) and set(x) == set(y):
            return {k: _cat(x[k], y[k]) for k in x}
        xa, ya = jnp.asarray(x), jnp.asarray(y)
        return jnp.concatenate([jnp.atleast_1d(xa), jnp.atleast_1d(ya)], axis=0)

    return {k: _cat(a[k], b[k]) for k in a}


def run_gradient_descent_scan(opt: int, beta1: float):
    """Choose optimizer and run scans.

    When this function is ``jax.jit``-ted, pass ``static_argnames=("opt", "beta1")``:
    ``opt`` must stay a plain Python :class:`int` so the branches below are resolved
    at compile time, and ``beta1`` must be static for Optax init and
    :func:`bias_correction_denom` with the chosen ``tx``."""
    params = {"a1": jnp.array(0.0), "a2": jnp.array(0.0)}
    if opt == 0:
        tx = optax.adam(learning_rate=LR, b1=beta1, b2=B2)
    elif opt == 1:
        tx = myano(learning_rate=LR, b1=beta1, b2=B2, gamma=1.0)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")
    opt_state = tx.init(params)

    def step(carry, _):
        params, opt_state, m1, m2 = carry
        loss, grads = jax.value_and_grad(loss_fun)(params, m1, m2)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        adam_state = new_opt_state[0]
        count = adam_state.count
        mu = adam_state.mu
        nu = adam_state.nu

        bc1 = bias_correction_denom(beta1, count)
        bc2 = bias_correction_denom(B2, count)
        mu_hat = jax.tree.map(lambda m: m / bc1.astype(m.dtype), mu)
        nu_hat = jax.tree.map(lambda v: v / bc2.astype(v.dtype), nu)

        record = {
            "loss": loss,
            "bc1": bc1,
            "bc2": bc2,
            "mu": mu,
            "nu": nu,
            "mu_hat": mu_hat,
            "nu_hat": nu_hat,
            "grads": grads,
            "grad_norm": pytree_norm(grads),
            "updates": updates,
            "update_norm": pytree_norm(updates),
            "params": new_params,
            "count": count,
        }
        return (new_params, new_opt_state, m1, m2), record

    (final_params, final_opt_state, _, _), history1 = jax.lax.scan(
        step, (params, opt_state, 1.0, 1.0), None, length=5
    )

    # gradient changes
    (final_params, final_opt_state, _, _), history2 = jax.lax.scan(
        step, (final_params, final_opt_state, 1.0, -1.0), None, length=10
    )

    history = _concat_history_stages(history1, history2)

    # `history` is a dict of arrays / pytrees with leading time axis
    # (length of first scan + length of second scan; here ``2 * NUM_STEPS``).
    return final_params, final_opt_state, history


def _plot_path_quivers(ax, history):
    """Draw (a1,a2) path with unit gradient and momentum directions on ``ax``."""
    a1 = jnp.asarray(history["params"]["a1"])
    a2 = jnp.asarray(history["params"]["a2"])
    g1 = jnp.asarray(history["grads"]["a1"])
    g2 = jnp.asarray(history["grads"]["a2"])
    gnorm = jnp.sqrt(g1 * g1 + g2 * g2 + 1e-12)
    u1 = g1 / gnorm
    u2 = g2 / gnorm

    m1 = jnp.asarray(history["mu"]["a1"])
    m2 = jnp.asarray(history["mu"]["a2"])
    mnorm = jnp.sqrt(m1 * m1 + m2 * m2 + 1e-12)
    v1 = m1 / mnorm
    v2 = m2 / mnorm

    _base = dict(angles="xy", scale_units="xy", scale=2.0, alpha=0.9)
    w_grad, w_mom = 0.0075, 0.0030

    ax.plot(a1, a2, "k.-", label="(a1, a2)", zorder=1)
    ax.quiver(
        a1,
        a2,
        u1,
        u2,
        color="C0",
        width=w_grad,
        zorder=2,
        label=r"$\nabla L$ (unit, thick)",
        **_base,
    )
    ax.quiver(
        a1,
        a2,
        v1,
        v2,
        color="C1",
        width=w_mom,
        zorder=3,
        label=r"$m$ (unit, thin)",
        **_base,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    run_scan = jax.jit(run_gradient_descent_scan, static_argnames=("opt", "beta1"))

    histories = []
    for beta1 in BETA1_VALUES:
        _fp, _os, history = run_scan(0, beta1)
        histories.append(history)

    n = len(BETA1_VALUES)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 4.5), sharex=True, sharey=True)
    if n == 1:
        axes = (axes,)
    for ax, beta1, history in zip(axes, BETA1_VALUES, histories):
        _plot_path_quivers(ax, history)
        ax.set_title(rf"$\beta_1 = {beta1}$")
    for ax in axes:
        ax.set_xlabel("a1")
    axes[0].set_ylabel("a2")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        r"Path: $\nabla L$ (thicker) and momentum $m$ (thinner), unit directions"
    )
    fig.tight_layout()
    plt.show()
