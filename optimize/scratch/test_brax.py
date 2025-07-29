import jax
import jax.numpy as jnp
import numpy as np
import imageio
from mujoco_playground import wrapper
from mujoco_playground import registry

env = registry.load("CheetahRun")
env_cfg = registry.get_default_config("CheetahRun")

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

state = jit_reset(jax.random.PRNGKey(0))
rollout = [state]

f = 0.5
for i in range(2000):
    action = []
    for j in range(env.action_size):
        action.append(
            jnp.sin(state.data.time * 2 * jnp.pi * f + j * 2 * jnp.pi / env.action_size)
        )
    action = jnp.array(action)
    state = jit_step(state, action)
    rollout.append(state)
    print(state.done)
