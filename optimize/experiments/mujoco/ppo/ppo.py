import jax
import jax.numpy as jnp
import hydra
from flax.training.train_state import TrainState
from typing import NamedTuple, Dict
from jax._src.typing import Array
from omegaconf import OmegaConf
import datetime
from optimize.utils.wandb_multilogger import WandbMultiLogger
from optimize.networks.mlp import ActorCritic
import numpy as np
import optax
from mujoco_playground import registry
from optimize.utils.jax_utils import pytree_norm


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
    train_state: TrainState
    state: jnp.ndarray
    done: jnp.ndarray
    cumulative_return: jnp.ndarray
    timesteps: jnp.ndarray  # Track timesteps for each environment
    update_step: int
    rng: Array


class Updatestate(NamedTuple):
    train_state: TrainState
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: Array


def make_train(config):
    # env
    env = registry.load("CheetahRun")

    # config
    config["num_updates"] = (
        config["total_timesteps"] // config["num_steps"] // config["num_envs"]
    )
    config["minibatch_size"] = (
        config["num_envs"] * config["num_steps"] // config["num_minibatches"]
    )

    def train(rng, exp_id):
        jax.debug.print("Compile Finished. Running...")

        def train_setup(rng):
            # env reset
            rng, _rng_reset = jax.random.split(rng)
            _rng_resets = jax.random.split(_rng_reset, config["num_envs"])
            state = jax.vmap(env.reset)(_rng_resets)

            # network and optimizers
            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (config["num_minibatches"] * config["update_epochs"]))
                    / config["num_updates"]
                )
                return config["lr"] * frac

            network = ActorCritic(
                action_dim=env.action_size,
                activation=config["activation"],
            )
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(state.obs.shape)
            network_params = network.init(_rng, init_x)
            if config["optimizer"] == "adam":
                optimizer = optax.adam
            elif config["optimizer"] == "rmsprop":
                optimizer = optax.rmsprop
            if config["anneal_lr"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["max_grad_norm"]),
                    optimizer(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["max_grad_norm"]),
                    optimizer(config["lr"], eps=1e-5),
                )
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
            return state, train_state, network

        rng, _rng_setup = jax.random.split(rng)
        state, train_state, network = train_setup(_rng_setup)

        def _train_loop(runner_state, unused):
            initial_timesteps = runner_state.timesteps

            # collect transitions
            def _env_step(runner_state, unused):
                train_state = runner_state.train_state
                state = runner_state.state
                done = runner_state.done
                rng = runner_state.rng

                # reset env if needed
                def reset_if_done(state, done, rng):
                    return jax.lax.cond(done, lambda: env.reset(rng), lambda: state)

                rng, _rng_reset = jax.random.split(rng)
                rng_resets = jax.random.split(_rng_reset, config["num_envs"])
                state = jax.vmap(reset_if_done)(state, done, rng_resets)

                # sample actions
                rng, _rng_action = jax.random.split(rng)
                pi, value = network.apply(train_state.params, state.obs)
                action = pi.sample(seed=_rng_action)
                # Clip continuous actions to valid range
                action = jnp.clip(action, -1.0, 1.0)
                log_prob = pi.log_prob(action).sum(
                    axis=-1
                )  # Sum across action dimensions

                # step the environment
                rng, _rng = jax.random.split(rng)
                new_state = jax.vmap(env.step)(state, action)

                # Update timesteps and check horizon
                timesteps = runner_state.timesteps + 1
                horizon_done = timesteps >= config["horizon"]
                new_done = jnp.asarray(new_state.done, dtype=bool) | horizon_done

                # Reset timesteps for environments that are done
                timesteps = jnp.where(new_done, 0, timesteps)

                transition = Transition(  # transitions are batched (num_actors, ...)
                    obs=state.obs,
                    action=action.squeeze(),
                    log_prob=log_prob,
                    reward=new_state.reward,
                    done=state.done,
                    new_done=new_done,
                    value=value.squeeze(),
                    info=new_state.info,
                )

                runner_state = RunnerState(
                    train_state=train_state,
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
            train_state = runner_state.train_state
            last_state = runner_state.state
            rng = runner_state.rng

            _, last_value = network.apply(train_state.params, last_state.obs)
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
                train_state = update_state.train_state
                traj_batch = update_state.traj_batch
                advantages = update_state.advantages
                targets = update_state.targets
                rng = update_state.rng

                rng, _rng_permute = jax.random.split(rng)
                permutation = jax.random.permutation(_rng_permute, config["num_envs"])
                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                shuffled_batch = jax.tree.map(  # (time, envs, ...)
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                shuffled_batch_split = jax.tree.map(
                    lambda x: jnp.reshape(  # split into minibatches along actor dimension (dim 1)
                        x,
                        [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                    ),
                    shuffled_batch,
                )
                minibatches = jax.tree.map(  # swap minibatch and time axis,
                    lambda x: jnp.swapaxes(x, 0, 1),
                    shuffled_batch_split,
                )

                def _update_minibatch(train_state, minibatch):
                    traj_minibatch, advantages_minibatch, targets_minibatch = minibatch

                    def _loss(params, traj_minibatch, gae_minibatch, targets_minibatch):
                        # rerun network
                        pi, value = network.apply(params, traj_minibatch.obs)
                        log_prob = pi.log_prob(traj_minibatch.action).sum(
                            axis=-1
                        )  # Sum across action dimensions

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
                        approx_kl = ((ratio - 1) - logratio).mean()
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
                            "approx_kl": approx_kl,
                            "clip_frac": clip_frac,
                            "gae_mean": gae_minibatch.mean(),
                            "gae_std": gae_minibatch.std(),
                            "gae_max": gae_minibatch.max(),
                        }

                    grad_fn = jax.value_and_grad(_loss, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        traj_minibatch,
                        advantages_minibatch,
                        targets_minibatch,
                    )
                    train_state = train_state.apply_gradients(
                        grads=grads,
                    )

                    total_loss[1]["grad_norm"] = pytree_norm(grads)
                    return train_state, total_loss

                train_state, total_loss = jax.lax.scan(
                    _update_minibatch,
                    train_state,
                    minibatches,
                )

                update_state = Updatestate(
                    train_state=train_state,
                    traj_batch=traj_batch,
                    advantages=advantages,
                    targets=targets,
                    rng=rng,
                )

                return update_state, total_loss

            update_state = Updatestate(
                train_state=train_state,
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
            only_returns = jnp.where(done, returns, 0)  # only returns at done steps
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

            # Calculate average episode length from completed episodes
            only_episode_ends = jnp.where(
                done, episode_lengths, 0
            )  # only lengths at done steps
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
                train_state=update_state.train_state,
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
            train_state=train_state,
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


@hydra.main(version_base=None, config_path="./", config_name="config_ppo")
def main(config):
    try:
        # wandb
        config = OmegaConf.to_container(config)
        job_type = f"ppo_{config['env_name']}"
        group = f"ppo_{config['env_name']}"
        if config["use_timestamp"]:
            group += datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
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
        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Starting compile...")
        train_vmap = jax.vmap(make_train(config))
        train_vjit = jax.jit(train_vmap)

        # profile
        with jax.profiler.trace("train_vjit", create_perfetto_link=True):
            out = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
    finally:
        LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
