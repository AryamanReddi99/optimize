import jax
import jax.numpy as jnp
import hydra
from flax.training.train_state import TrainState
from typing import NamedTuple, Dict
from jax._src.typing import Array
from omegaconf import OmegaConf
import datetime
from optimize.utils.wandb_multilogger import WandbMultiLogger
from optimize.networks.mlp import ActorCriticContinuous
import numpy as np
import optax
from brax import envs
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
    env = envs.get_environment("ant")
    env = envs.wrappers.EpisodeWrapper(
        env, episode_length=config["horizon"], action_repeat=1
    )
    env = envs.wrappers.AutoResetWrapper(env)

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

            network = ActorCriticContinuous(
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

                # AutoResetWrapper handles resets automatically

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

                return_t = advantages + traj_batch.value
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                return advantages, return_t

            advantages, targets = _calculate_gae(traj_batch, last_value)

            # update networks
            def _update_epoch(update_state, unused):
                train_state = update_state.train_state
                traj_batch = update_state.traj_batch
                advantages = update_state.advantages
                targets = update_state.targets
                rng = update_state.rng

                rng, _rng = jax.random.split(rng)
                batch_size = config["minibatch_size"]
                assert (
                    batch_size % config["num_minibatches"] == 0
                ), f"batch_size {batch_size} must be divisible by num_minibatches {config['num_minibatches']}"
                minibatch_size = batch_size // config["num_minibatches"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[1:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x,
                        [config["num_minibatches"], minibatch_size] + list(x.shape[1:]),
                    ),
                    shuffled_batch,
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
                        clip_frac = (
                            (jnp.abs((ratio - 1.0)) > config["clip_eps"])
                            .astype(jnp.float32)
                            .mean()
                        )

                        # total loss
                        loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss
                            - config["ent_coef"] * entropy
                        )

                        return loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            approx_kl,
                            clip_frac,
                        )

                    grad_fn = jax.value_and_grad(_loss, has_aux=True)
                    loss_info, grads = grad_fn(
                        train_state.params,
                        traj_minibatch,
                        advantages_minibatch,
                        targets_minibatch,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_info

                train_state, loss_info = jax.lax.scan(
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
                return update_state, loss_info

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

            train_state = update_state.train_state
            metric = traj_batch.info
            rng = update_state.rng

            runner_state = RunnerState(
                train_state=train_state,
                state=runner_state.state,
                done=runner_state.done,
                cumulative_return=runner_state.cumulative_return,
                timesteps=runner_state.timesteps,
                update_step=runner_state.update_step + 1,
                rng=rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
            train_state=train_state,
            state=state,
            done=jnp.zeros((config["num_envs"],), dtype=bool),
            cumulative_return=jnp.zeros((config["num_envs"],)),
            timesteps=jnp.zeros((config["num_envs"],), dtype=jnp.int32),
            update_step=0,
            rng=_rng,
        )

        runner_state, metric = jax.lax.scan(
            _train_loop,
            runner_state,
            None,
            config["num_updates"],
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


@hydra.main(version_base=None, config_path=".", config_name="config_ppo")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    config = dict(config)

    # wandb logging
    if config["use_timestamp"]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config["exp_name"] = f"{config['exp_name']}_{timestamp}"

    logger = WandbMultiLogger(
        project=config["project"],
        name=config["exp_name"],
        mode=config["wandb_mode"],
    )

    # train
    train_jit = jax.jit(make_train(config))
    rng = jax.random.PRNGKey(config["seed"])

    for seed in range(config["num_seeds"]):
        rng, _rng = jax.random.split(rng)
        out = train_jit(_rng, seed)
        logger.log_config(config)
        logger.log_metrics(out["metric"], step=out["metric"]["update_step"])


if __name__ == "__main__":
    main()
