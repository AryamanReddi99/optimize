"""Ten Adam steps in a `lax.scan`; collect time series of diagnostics.

After `run_gradient_descent_scan()`, `history` contains JAX arrays with leading
axis ``NUM_STEPS + 1`` (scan length plus one post-scan step with ``multiplier=0.1``):

- ``loss``: shape ``(NUM_STEPS,)``
- ``bc1``, ``bc2``: bias-correction denominators ``1 - b1^t``, ``1 - b2^t``
- ``mu``, ``nu``, ``mu_hat``, ``nu_hat``, ``grads``: same pytree structure as
  params; each leaf has shape ``(NUM_STEPS,) + leaf_shape``
"""

import jax
import jax.numpy as jnp
import optax

from optimize.utils.jax_utils import pytree_norm
from optimize.optimizers.optimizers import adam, yogi, myano
import matplotlib.pyplot as plt

B1 = 0.9
B2 = 0.999
LR = 1.0
NUM_STEPS = 10


def bias_correction_denom(decay: float, count: jax.Array) -> jax.Array:
    """Scalar 1 - decay**t, matching Optax bias correction denominator."""
    c = count.astype(jnp.float32)
    return jnp.asarray(1.0, jnp.float32) - (jnp.asarray(decay, jnp.float32) ** c)


def loss_fun(params, multiplier: float = 1.0):
    return ((params["a1"] + params["a2"]) * multiplier).mean()


def run_gradient_descent_scan():
    params = {"a1": jnp.array(0.0), "a2": jnp.array(0.0)}
    # tx = yogi(learning_rate=LR, b1=B1, b2=B2)
    # tx = optax.adam(learning_rate=LR, b1=B1, b2=B2)
    tx = myano(learning_rate=LR, b1=B1, b2=B2)
    opt_state = tx.init(params)

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fun)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        adam_state = new_opt_state[0]
        count = adam_state.count
        mu = adam_state.mu
        nu = adam_state.nu

        bc1 = bias_correction_denom(B1, count)
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
            "updates": updates,
            "update_norm": pytree_norm(updates),
        }
        return (new_params, new_opt_state), record

    (final_params, final_opt_state), history = jax.lax.scan(
        step, (params, opt_state), None, length=NUM_STEPS
    )

    # gradient drops
    loss, grads = jax.value_and_grad(loss_fun)(final_params, multiplier=0.99)
    updates, new_opt_state = tx.update(grads, final_opt_state, final_params)
    final_params = optax.apply_updates(final_params, updates)

    adam_state = new_opt_state[0]
    count = adam_state.count
    mu = adam_state.mu
    nu = adam_state.nu

    bc1 = bias_correction_denom(B1, count)
    bc2 = bias_correction_denom(B2, count)
    mu_hat = jax.tree.map(lambda m: m / bc1.astype(m.dtype), mu)
    nu_hat = jax.tree.map(lambda v: v / bc2.astype(v.dtype), nu)

    post_record = {
        "loss": loss,
        "bc1": bc1,
        "bc2": bc2,
        "mu": mu,
        "nu": nu,
        "mu_hat": mu_hat,
        "nu_hat": nu_hat,
        "grads": grads,
        "updates": updates,
        "update_norm": pytree_norm(updates),
    }

    def _append_time(h, x):
        return jnp.concatenate([h, jnp.expand_dims(x, 0)], axis=0)

    history = {
        **history,
        **{
            k: (
                jax.tree.map(_append_time, history[k], post_record[k])
                if k
                in (
                    "mu",
                    "nu",
                    "mu_hat",
                    "nu_hat",
                    "grads",
                    "updates",
                )
                else _append_time(history[k], post_record[k])
            )
            for k in post_record
        },
    }

    # `history` is a dict of arrays / pytrees with leading time axis
    # (NUM_STEPS + 1, ...).
    return final_params, final_opt_state, history


if __name__ == "__main__":
    final_params, final_opt_state, history = jax.jit(run_gradient_descent_scan)()
    print("losses:", history["loss"])
    print("bc1:", history["bc1"])
    print("bc2:", history["bc2"])
    print("grads[a1]:", history["grads"]["a1"])
    print("mu[a1]:", history["mu"]["a1"])
    print("mu_hat[a1]:", history["mu_hat"]["a1"])
    print("nu[a1]:", history["nu"]["a1"])
    print("nu_hat[a1]:", history["nu_hat"]["a1"])
    print("updates[a1]:", history["updates"]["a1"])
    print("update_norm:", history["update_norm"])
    print("final_params:", final_params)

    # plt.plot(history["update_norm"])
    # plt.show()
