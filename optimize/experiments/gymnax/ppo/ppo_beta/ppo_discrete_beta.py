import jax
import jax.numpy as jnp
import hydra
from typing import NamedTuple, Dict
from jax._src.typing import Array
from omegaconf import OmegaConf
import datetime
from optimize.utils.wandb_multilogger import WandbMultiLogger
from optimize.networks.mlp import ActorCriticDiscrete
import numpy as np
import optax
import gymnax
from optimize.utils.jax_utils import pytree_norm
import pickle
import os


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    new_done: jnp.ndarray
    value: jnp.ndarray
    info: jnp.ndarray


class RunnerState(NamedTuple):
    params: jnp.ndarray  # Network parameters
    opt_state: jnp.ndarray  # Optimizer state
    running_grad: jnp.ndarray  # Running gradient for cosine similarity
    obs: jnp.ndarray
    state: jnp.ndarray
    done: jnp.ndarray
    cumulative_return: jnp.ndarray
    timesteps: jnp.ndarray
    update_step: int
    rng: Array


class Updatestate(NamedTuple):
    params: jnp.ndarray
    opt_state: jnp.ndarray
    running_grad: jnp.ndarray  # Running gradient for cosine similarity
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: Array


def save_model(
    params,
    config,
    exp_id,
    models_dir,
):
    """Save the trained model parameters."""
    os.makedirs(models_dir, exist_ok=True)

    # Create a unique filename based on config and experiment ID
    model_path = os.path.join(models_dir, exp_id)

    # Save model data
    model_data = {
        "params": params,
        "config": config,
        "env_name": config["env_name"],
        "exp_id": exp_id,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    return model_path


def make_train(config):
    # env
    env, env_params = gymnax.make(config["env_name"])

    # config
    config["num_updates"] = (
        config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    )
    config["minibatch_size"] = (
        config["num_envs"] * config["num_steps"] // config["num_minibatches"]
    )

    def train(rng, exp_id):
        def train_setup(rng):
            # env reset
            rng, _rng_reset = jax.random.split(rng)
            _rng_resets = jax.random.split(_rng_reset, config["num_envs"])
            obs, state = jax.vmap(env.reset, in_axes=(0, None))(_rng_resets, env_params)

            # network and optimizers
            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (config["num_minibatches"] * config["update_epochs"]))
                    / config["num_updates"]
                )
                return config["lr"] * frac

            if config["anneal_lr"]:
                lr_schedule = linear_schedule
            else:
                lr_schedule = config["lr"]
            network = ActorCriticDiscrete(
                action_dim=env.num_actions,
                activation=config["activation"],
            )
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(obs.shape)
            network_params = network.init(_rng, init_x)
            # Create initial optimizer
            if config["optimizer"] == "adam":
                tx = optax.chain(
                    optax.clip_by_global_norm(config["max_grad_norm"]),
                    optax.adam(
                        learning_rate=lr_schedule,
                        eps=1e-5,
                        b1=config["beta_1"],
                        b2=config["beta_2"],
                    ),
                )
            elif config["optimizer"] == "rmsprop":
                tx = optax.chain(
                    optax.clip_by_global_norm(config["max_grad_norm"]),
                    optax.rmsprop(learning_rate=lr_schedule, eps=1e-5),
                )
            elif config["optimizer"] == "sgd":
                tx = optax.chain(
                    optax.clip_by_global_norm(config["max_grad_norm"]),
                    optax.sgd(learning_rate=lr_schedule),
                )

            # Initialize optimizer state
            opt_state = tx.init(network_params)

            # Initialize running gradient (zero gradient)
            running_grad = jax.tree.map(jnp.zeros_like, network_params)

            return (
                obs,
                state,
                network_params,
                opt_state,
                running_grad,
                network,
                tx,
                lr_schedule,
            )

        rng, _rng_setup = jax.random.split(rng)
        obs, state, params, opt_state, running_grad, network, tx, lr_schedule = (
            train_setup(_rng_setup)
        )

        def _train_loop(runner_state, unused):
            initial_timesteps = runner_state.timesteps

            # collect transitions
            def _env_step(runner_state, unused):
                params = runner_state.params
                obs = runner_state.obs
                state = runner_state.state
                done = runner_state.done
                rng = runner_state.rng

                # reset env if needed
                def reset_if_done(obs, state, done, rng):
                    return jax.lax.cond(
                        done, lambda: env.reset(rng, env_params), lambda: (obs, state)
                    )

                rng, _rng_reset = jax.random.split(rng)
                rng_resets = jax.random.split(_rng_reset, config["num_envs"])
                obs, state = jax.vmap(reset_if_done)(obs, state, done, rng_resets)

                # sample actions
                rng, _rng_action = jax.random.split(rng)
                pi, value = network.apply(params, obs)
                action = pi.sample(seed=_rng_action)
                log_prob = pi.log_prob(action)

                # step the environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["num_envs"])
                new_obs, new_state, reward, new_done, info = jax.vmap(env.step)(
                    rng_step, state, action
                )

                # Update timesteps
                timesteps = runner_state.timesteps + 1
                timesteps = jnp.where(new_done, 0, timesteps)

                transition = Transition(
                    obs=obs,
                    action=action.squeeze(),
                    log_prob=log_prob,
                    reward=reward,
                    done=done,
                    new_done=new_done,
                    value=value.squeeze(),
                    info=info,
                )

                runner_state = RunnerState(
                    params=params,
                    opt_state=runner_state.opt_state,
                    running_grad=runner_state.running_grad,
                    obs=new_obs,
                    state=new_state,
                    done=new_done,
                    cumulative_return=runner_state.cumulative_return,
                    timesteps=timesteps,
                    update_step=runner_state.update_step,
                    rng=rng,
                )

                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                runner_state,
                None,
                config["num_steps"],
            )

            # advantages
            params = runner_state.params
            last_obs = runner_state.obs
            rng = runner_state.rng

            _, last_value = network.apply(params, last_obs)
            last_value = last_value.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.new_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["gamma"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_value)

            # update networks
            def _update_epoch(update_state, unused):
                params = update_state.params
                opt_state = update_state.opt_state
                running_grad = update_state.running_grad
                traj_batch = update_state.traj_batch
                advantages = update_state.advantages
                targets = update_state.targets
                rng = update_state.rng

                rng, _rng_permute = jax.random.split(rng)
                permutation = jax.random.permutation(_rng_permute, config["num_envs"])
                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                shuffled_batch_split = jax.tree.map(
                    lambda x: jnp.reshape(
                        x,
                        [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                    ),
                    shuffled_batch,
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(x, 0, 1),
                    shuffled_batch_split,
                )

                def _update_minibatch(carry, minibatch):
                    params, opt_state, running_grad = carry
                    traj_minibatch, advantages_minibatch, targets_minibatch = minibatch

                    def _loss(params, traj_minibatch, gae_minibatch, targets_minibatch):
                        # rerun network
                        pi, value = network.apply(params, traj_minibatch.obs)
                        log_prob = pi.log_prob(traj_minibatch.action)

                        # actor loss
                        logratio = log_prob - traj_minibatch.log_prob
                        ratio = jnp.exp(logratio)
                        gae_minibatch = (gae_minibatch - gae_minibatch.mean()) / (
                            gae_minibatch.std() + 1e-8
                        )
                        loss_actor_1 = ratio * gae_minibatch
                        loss_actor_2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae_minibatch
                        )
                        loss_actor = -jnp.minimum(loss_actor_1, loss_actor_2).mean()
                        entropy = pi.entropy().mean()

                        # critic loss
                        value_pred_clipped = traj_minibatch.value + (
                            value - traj_minibatch.value
                        ).clip(-config["clip_eps"], config["clip_eps"])
                        value_loss = jnp.square(value - targets_minibatch)
                        value_loss_clipped = jnp.square(
                            value_pred_clipped - targets_minibatch
                        )
                        value_loss = (
                            0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
                        )

                        # stats
                        approx_kl_backward = ((ratio - 1) - logratio).mean()
                        approx_kl_forward = (ratio * logratio - (ratio - 1)).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["clip_eps"])

                        total_loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss
                            - config["ent_coef"] * entropy
                        )

                        return total_loss, {
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio": ratio,
                            "approx_kl_backward": approx_kl_backward,
                            "approx_kl_forward": approx_kl_forward,
                            "clip_frac": clip_frac,
                            "gae_mean": gae_minibatch.mean(),
                            "gae_std": gae_minibatch.std(),
                            "gae_max": gae_minibatch.max(),
                        }

                    grad_fn = jax.value_and_grad(_loss, has_aux=True)
                    total_loss, grads = grad_fn(
                        params,
                        traj_minibatch,
                        advantages_minibatch,
                        targets_minibatch,
                    )

                    # Apply gradients
                    updates, updated_opt_state = tx.update(grads, opt_state)
                    new_params = optax.apply_updates(params, updates)

                    new_tx = optax.chain(
                        optax.clip_by_global_norm(config["max_grad_norm"]),
                        optax.adam(
                            learning_rate=lr_schedule,
                            eps=1e-5,
                            b1=config["beta_1"],
                            b2=config["beta_2"],
                        ),
                    )

                    # Reinitialize optimizer state with new beta1
                    new_opt_state = new_tx.init(new_params)
                    new_opt_state = (
                        new_opt_state[0],
                        (
                            optax.ScaleByAdamState(
                                count=updated_opt_state[1][0].count,
                                mu=updated_opt_state[1][0].mu,
                                nu=updated_opt_state[1][0].nu,
                            ),
                            updated_opt_state[1][1],
                        ),
                    )

                    # Calculate cosine similarity between current gradient and running gradient
                    def cosine_similarity(grad1, grad2):
                        # Flatten gradients for cosine similarity calculation
                        flat_grad1 = jax.tree.leaves(grad1)
                        flat_grad2 = jax.tree.leaves(grad2)

                        # Concatenate all gradients
                        vec1 = jnp.concatenate([jnp.ravel(x) for x in flat_grad1])
                        vec2 = jnp.concatenate([jnp.ravel(x) for x in flat_grad2])

                        # Calculate cosine similarity
                        dot_product = jnp.dot(vec1, vec2)
                        norm1 = jnp.linalg.norm(vec1)
                        norm2 = jnp.linalg.norm(vec2)

                        # Avoid division by zero
                        denominator = norm1 * norm2
                        cosine_sim = jnp.where(
                            denominator > 1e-8, dot_product / denominator, 0.0
                        )
                        return cosine_sim

                    cos_sim = cosine_similarity(grads, running_grad)
                    cos_sim_mu = cosine_similarity(grads, new_opt_state[1][0].mu)

                    # Calculate angle between gradient vectors (in degrees)
                    cos_sim_clamped = jnp.clip(cos_sim, -1.0, 1.0)
                    gradient_angle_rad = jnp.arccos(cos_sim_clamped)
                    gradient_angle_deg = gradient_angle_rad * 180.0 / jnp.pi

                    # Update running gradient with current gradient
                    new_running_grad = grads

                    total_loss[1]["grad_norm"] = pytree_norm(grads)
                    total_loss[1]["mu_norm"] = pytree_norm(new_opt_state[1][0].mu)
                    total_loss[1]["nu_norm"] = pytree_norm(new_opt_state[1][0].nu)
                    total_loss[1]["cosine_similarity"] = cos_sim
                    total_loss[1]["gradient_angle_deg"] = gradient_angle_deg
                    total_loss[1]["cosine_similarity_mu"] = cos_sim_mu
                    return (new_params, new_opt_state, new_running_grad), total_loss

                (final_params, final_opt_state, final_running_grad), total_loss = (
                    jax.lax.scan(
                        _update_minibatch,
                        (params, opt_state, running_grad),
                        minibatches,
                    )
                )

                update_state = Updatestate(
                    params=final_params,
                    opt_state=final_opt_state,
                    running_grad=final_running_grad,
                    traj_batch=traj_batch,
                    advantages=advantages,
                    targets=targets,
                    rng=rng,
                )

                return update_state, total_loss

            update_state = Updatestate(
                params=params,
                opt_state=runner_state.opt_state,
                running_grad=runner_state.running_grad,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=targets,
                rng=rng,
            )

            update_state, loss_info = jax.lax.scan(
                _update_epoch,
                update_state,
                None,
                config["update_epochs"],
            )

            # log returns
            reward = traj_batch.reward
            done = traj_batch.new_done
            cumulative_return = runner_state.cumulative_return

            def _returns(carry_return, inputs):
                reward, done = inputs
                cumulative_return = carry_return + reward
                reset_return = jnp.zeros(reward.shape[1:], dtype=float)
                carry_return = jnp.where(done, reset_return, cumulative_return)
                return carry_return, cumulative_return

            new_cumulative_return, returns = jax.lax.scan(
                _returns,
                cumulative_return,
                (reward, done),
            )
            only_returns = jnp.where(done, returns, 0)
            returns_avg = jnp.where(
                done.sum() > 0, only_returns.sum() / done.sum(), 0.0
            )

            # log episode lengths
            def _episode_lengths(carry_length, done):
                cumulative_length = carry_length + 1
                reset_length = jnp.zeros(done.shape[1:], dtype=jnp.int32)
                carry_length = jnp.where(done, reset_length, cumulative_length)
                return carry_length, cumulative_length

            _, episode_lengths = jax.lax.scan(_episode_lengths, initial_timesteps, done)

            only_episode_ends = jnp.where(done, episode_lengths, 0)
            episode_length_avg = jnp.where(
                done.sum() > 0, only_episode_ends.sum() / done.sum(), 0.0
            )

            # log info
            total_loss, loss_info = loss_info
            loss_info["total_loss"] = total_loss
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            # wandb
            metric = {}
            metric["update_step"] = runner_state.update_step
            metric["env_step"] = (
                runner_state.update_step * config["num_envs"] * config["num_steps"]
            )
            metric["return"] = returns_avg
            metric["episode_length"] = episode_length_avg
            metric.update(loss_info)

            def callback(exp_id, metric):
                np_log_dict = {k: np.array(v) for k, v in metric.items()}
                LOGGER.log(int(exp_id), np_log_dict)

            jax.experimental.io_callback(callback, None, exp_id, metric)

            runner_state = RunnerState(
                params=update_state.params,
                opt_state=update_state.opt_state,
                running_grad=update_state.running_grad,
                obs=runner_state.obs,
                state=runner_state.state,
                done=runner_state.done,
                cumulative_return=new_cumulative_return,
                timesteps=runner_state.timesteps,
                update_step=runner_state.update_step + 1,
                rng=runner_state.rng,
            )

            return runner_state, metric

        rng, _train_rng = jax.random.split(rng)
        done = jnp.zeros((config["num_envs"]), dtype=jnp.bool_)
        cumulative_return = jnp.zeros((config["num_envs"]), dtype=float)
        initial_runner_state = RunnerState(
            params=params,
            opt_state=opt_state,
            running_grad=running_grad,
            obs=obs,
            state=state,
            done=done,
            cumulative_return=cumulative_return,
            timesteps=jnp.zeros((config["num_envs"]), dtype=jnp.int32),
            update_step=0,
            rng=_train_rng,
        )

        final_runner_state, metrics_batch = jax.lax.scan(
            _train_loop,
            initial_runner_state,
            None,
            length=config["num_updates"],
        )
        return final_runner_state, metrics_batch

    return train


@hydra.main(version_base=None, config_path="./", config_name="config_ppo_beta")
def main(config):
    try:
        # vmap and compile
        config = OmegaConf.to_container(config)
        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Starting compile...")
        train_vmap = jax.vmap(make_train(config))
        train_vjit = jax.block_until_ready(jax.jit(train_vmap))
        print("Compile finished...")

        # wandb
        job_type = f"{config['job_type']}_{config['env_name']}"
        group = job_type + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        global LOGGER
        LOGGER = WandbMultiLogger(
            project=config["project"],
            group=group,
            job_type=job_type,
            config=config,
            mode=config["wandb_mode"],
            seed=config["seed"],
            num_seeds=config["num_seeds"],
        )

        # run
        print("Running...")
        out = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
    finally:
        LOGGER.finish()
        print("Finished.")

        # Save the final models
        if config["save_model"]:
            model_path = save_model(out[0].params, config, group, config["models_dir"])
            print(f"Models saved to: {model_path}")


if __name__ == "__main__":
    main()
