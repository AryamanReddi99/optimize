import jax
import jax.numpy as jnp
import gymnax
import pickle
import os
import argparse
from optimize.networks.mlp import ActorCriticDiscrete
import time


def load_model(model_path):
    """Load a trained model from file."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    print(f"Loaded model from: {model_path}")
    print(f"Environment: {model_data['env_name']}")
    print(f"Experiment ID: {model_data['exp_id']}")

    # Extract the 0th model parameters if they have an extra leading dimension
    params = model_data["params"]

    def extract_params(params):
        """Recursively traverse nested dictionary and extract first parameters."""
        if isinstance(params, dict):
            return {k: extract_params(v) for k, v in params.items()}
        elif isinstance(params, jnp.ndarray) and len(params.shape) > 0:
            return params[0]
        else:
            return params

    params = extract_params(params)
    model_data["params"] = params
    return model_data


def render_episode(params, network, env, env_params, max_steps=1000, render_delay=0.01):
    """Render a single episode with the trained agent."""
    # Reset environment
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, env_params)

    total_reward = 0.0
    step = 0

    print(f"Starting episode (max {max_steps} steps)...")

    while step < max_steps:
        # Get action from policy
        pi, value = network.apply(params, obs)
        action = pi.sample(seed=jax.random.PRNGKey(step))  # Deterministic for rendering

        # Step environment
        rng = jax.random.PRNGKey(step)
        new_obs, new_state, reward, done, info = env.step(rng, state, action)

        # Update state
        obs = new_obs
        state = new_state
        total_reward += reward

        # Print step info
        print(
            f"Step {step}: Action={action}, Reward={reward:.3f}, Total={total_reward:.3f}"
        )

        # Check if episode is done
        if done:
            print(
                f"Episode finished after {step + 1} steps with total reward: {total_reward:.3f}"
            )
            break

        step += 1
        time.sleep(render_delay)  # Add delay for visualization

    if step >= max_steps:
        print(
            f"Episode reached max steps ({max_steps}) with total reward: {total_reward:.3f}"
        )

    return total_reward


def render_multiple_episodes(
    params, network, env, env_params, num_episodes=5, max_steps=1000
):
    """Render multiple episodes and compute average performance."""
    rewards = []

    print(f"Rendering {num_episodes} episodes...")

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        # Reset environment
        rng = jax.random.PRNGKey(episode)
        obs, state = env.reset(rng, env_params)

        total_reward = 0.0
        step = 0

        while step < max_steps:
            # Get action from policy
            pi, value = network.apply(params, obs)
            action = pi.sample(seed=jax.random.PRNGKey(episode * 1000 + step))

            # Step environment
            rng = jax.random.PRNGKey(episode * 1000 + step)
            new_obs, new_state, reward, done, info = env.step(rng, state, action)

            # Update state
            obs = new_obs
            state = new_state
            total_reward += reward

            # Check if episode is done
            if done:
                break

            step += 1

        rewards.append(total_reward)
        print(f"Episode {episode + 1} finished with reward: {total_reward:.3f}")

    avg_reward = sum(rewards) / len(rewards)
    print(f"\n--- Summary ---")
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.3f}")
    print(f"Min reward: {min(rewards):.3f}")
    print(f"Max reward: {max(rewards):.3f}")

    return rewards, avg_reward


def main():
    model_path = "/home/aryaman/Desktop/project-optimize/optimize/optimize/experiments/gymnax/ppo/ppo_beta/models/ppo_beta_cosine_Reacher-misc_2025-08-05_12-09-48"

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    # Load model
    model_data = load_model(model_path)
    params = model_data["params"]
    config = model_data["config"]
    env_name = model_data["env_name"]

    # Create environment
    env, env_params = gymnax.make(env_name)

    # Create network
    network = ActorCriticDiscrete(
        action_dim=env.num_actions,
        activation=config["activation"],
    )

    print(f"Environment: {env_name}")
    print(f"Action space: {env.num_actions} actions")
    print(f"Network architecture: {config['activation']} activation")

    # Render episodes
    rewards, avg_reward = render_multiple_episodes(
        params,
        network,
        env,
        env_params,
        num_episodes=5,
        max_steps=1000,
    )

    print(f"\nModel performance summary:")
    print(f"Model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Training config: {config}")


if __name__ == "__main__":
    main()
