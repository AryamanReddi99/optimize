import jax
import jax.numpy as jnp
import optax


def fn(init_params):
    def _adam_scan(carry, step):
        params, opt_state, beta1 = carry

        # Create a new Adam optimizer with the current beta1
        adam = optax.adam(learning_rate=0.01, b1=beta1)

        # Compute dummy gradients (in real scenario, you'd compute actual gradients)
        grads = jnp.array([0.1])

        # Apply the optimizer
        updates, new_opt_state = adam.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        # Modify beta1 for next iteration (example: increase by 5%)
        new_beta1 = beta1 * 1.05
        # Clamp beta1 to reasonable range (0.1 to 0.999)
        new_beta1 = jnp.clip(new_beta1, 0.1, 0.999)

        return (new_params, new_opt_state, new_beta1), None

    # Initialize optimizer state with initial beta1
    initial_beta1 = 0.9
    adam = optax.adam(learning_rate=0.01, b1=initial_beta1)
    opt_state = adam.init(init_params)

    # Run scan with (params, opt_state, beta1) as carry
    final_carry, _ = jax.lax.scan(
        _adam_scan, (init_params, opt_state, initial_beta1), None, 10
    )
    final_params, final_opt_state, final_beta1 = final_carry

    return final_params, final_beta1


# Test the function
v = jnp.array([0.9])
jitted_fn = jax.jit(fn)
final_params, final_beta1 = jitted_fn(v)
print("Final parameters:", final_params)
print("Final beta1:", final_beta1)
