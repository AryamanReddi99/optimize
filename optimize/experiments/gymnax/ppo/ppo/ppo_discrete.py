import os

# # disable randomness
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import jax.numpy as jnp
import hydra
from flax.training.train_state import TrainState
from typing import Any, NamedTuple
from omegaconf import OmegaConf
import datetime
from optimize.optimizers.optimizers import myano
from optimize.utils.wandb_multilogger import WandbMultiLogger
from optimize.networks.mlp import ActorDiscrete, CriticDiscrete
from optimize.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
import numpy as np
import optax
import gymnax
from optimize.utils.jax_utils import pytree_norm, cosine_similarity
import pickle


class Transition(NamedTuple):
    obs: FloatArray
    action: IntArray
    log_prob: FloatArray
    reward: FloatArray
    done: BoolArray
    new_done: BoolArray
    value: FloatArray
    info: dict[str, Any]


class RunnerState(NamedTuple):
    actor_train_state: TrainState
    critic_train_state: TrainState
    state: Any
    obs: FloatArray
    done: BoolArray
    cumulative_return: FloatArray
    timesteps: IntArray
    update_step: int
    running_grad_actor: Any  # Running gradient for cosine similarity (actor params)
    running_grad_critic: Any
    rng: PRNGKeyArray


class Updatestate(NamedTuple):
    actor_train_state: TrainState
    critic_train_state: TrainState
    traj_batch: Transition
    advantages: FloatArray
    targets: FloatArray
    running_grad_actor: Any
    running_grad_critic: Any
    rng: PRNGKeyArray


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
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )

    def train(rng, exp_id):
        def train_setup(rng):
            # env reset
            rng, _rng_reset = jax.random.split(rng)
            _rng_resets = jax.random.split(_rng_reset, config["num_envs"])
            obs, state = jax.vmap(env.reset, in_axes=(0, None))(_rng_resets, env_params)

            # network and optimizers (separate actor / critic)
            def make_lr_schedule(base_lr):
                if config["anneal_lr"]:

                    def linear_schedule(count):
                        frac = 1.0 - (count // config["num_gradient_steps"])
                        return base_lr * frac

                    return linear_schedule
                return base_lr

            lr_schedule_actor = make_lr_schedule(config["lr_actor"])
            lr_schedule_critic = make_lr_schedule(config["lr_critic"])

            def make_tx(lr_schedule):
                if config["optimizer"] == "adam":
                    return optax.chain(
                        optax.clip_by_global_norm(config["max_grad_norm"]),
                        optax.adam(
                            learning_rate=lr_schedule,
                            eps=1e-5,
                            b1=config["beta_1"],
                            b2=config["beta_2"],
                        ),
                    )
                elif config["optimizer"] == "myano":
                    return optax.chain(
                        optax.clip_by_global_norm(config["max_grad_norm"]),
                        myano(
                            learning_rate=lr_schedule,
                            eps=1e-5,
                            b1=config["beta_1"],
                            b2=config["beta_2"],
                            gamma=config["myano_gamma"],
                        ),
                    )
                if config["optimizer"] == "sgd":
                    return optax.chain(
                        optax.clip_by_global_norm(config["max_grad_norm"]),
                        optax.sgd(learning_rate=lr_schedule),
                    )
                raise ValueError(f"Unknown optimizer: {config['optimizer']}")

            tx_actor = make_tx(lr_schedule_actor)
            tx_critic = make_tx(lr_schedule_critic)

            actor = ActorDiscrete(
                action_dim=env.num_actions,
                activation=config["activation"],
                hidden_dim=config["fc_dim_size"],
            )
            critic = CriticDiscrete(
                activation=config["activation"],
                hidden_dim=config["fc_dim_size"],
            )
            rng, _rng_a, _rng_c = jax.random.split(rng, 3)
            init_x = jnp.zeros(obs.shape)
            actor_params = actor.init(_rng_a, init_x)
            critic_params = critic.init(_rng_c, init_x)

            actor_train_state = TrainState.create(
                apply_fn=actor.apply,
                params=actor_params,
                tx=tx_actor,
            )
            critic_train_state = TrainState.create(
                apply_fn=critic.apply,
                params=critic_params,
                tx=tx_critic,
            )

            running_grad_actor = jax.tree.map(jnp.zeros_like, actor_params)
            running_grad_critic = jax.tree.map(jnp.zeros_like, critic_params)

            return (
                obs,
                state,
                actor_train_state,
                critic_train_state,
                running_grad_actor,
                running_grad_critic,
                actor,
                critic,
            )

        rng, _rng_setup = jax.random.split(rng)
        (
            obs,
            state,
            actor_train_state,
            critic_train_state,
            running_grad_actor,
            running_grad_critic,
            actor,
            critic,
        ) = train_setup(_rng_setup)

        def _train_loop(runner_state, unused):
            initial_timesteps = runner_state.timesteps

            # collect transitions
            def _env_step(runner_state, unused):
                actor_train_state = runner_state.actor_train_state
                critic_train_state = runner_state.critic_train_state
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
                pi = actor.apply(actor_train_state.params, obs)
                value = critic.apply(critic_train_state.params, obs)
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
                    actor_train_state=actor_train_state,
                    critic_train_state=critic_train_state,
                    running_grad_actor=runner_state.running_grad_actor,
                    running_grad_critic=runner_state.running_grad_critic,
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
                config["num_steps_per_env_per_update"],
            )

            # advantages
            critic_train_state = runner_state.critic_train_state
            last_obs = runner_state.obs
            rng = runner_state.rng

            last_value = critic.apply(critic_train_state.params, last_obs)
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

            actor_train_state = runner_state.actor_train_state
            critic_train_state = runner_state.critic_train_state

            # update networks (separate actor / critic optimizers)
            def _update_epoch(update_state, unused):
                actor_train_state = update_state.actor_train_state
                critic_train_state = update_state.critic_train_state
                running_grad_actor = update_state.running_grad_actor
                running_grad_critic = update_state.running_grad_critic
                traj_batch = update_state.traj_batch
                advantages = update_state.advantages
                targets = update_state.targets
                rng = update_state.rng

                rng, _rng_permute = jax.random.split(rng)
                batch = (traj_batch, advantages.squeeze(), targets.squeeze())

                def _reshape_batch(x):
                    if x.ndim == 2:
                        return x.reshape(-1)
                    elif x.ndim == 3:
                        return x.reshape(-1, *x.shape[2:])
                    else:
                        return x.reshape(-1, *x.shape[3:])

                batch_reshaped = jax.tree_util.tree_map(_reshape_batch, batch)
                permutation = jax.random.permutation(
                    _rng_permute, config["batch_shuffle_dim"]
                )
                batch_shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch_reshaped
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape(config["num_minibatches"], -1, *x.shape[1:]),
                    batch_shuffled,
                )

                def _update_minibatch(carry, minibatch):
                    (
                        actor_train_state,
                        critic_train_state,
                        running_grad_actor,
                        running_grad_critic,
                    ) = carry
                    traj_minibatch, advantages_minibatch, targets_minibatch = minibatch

                    def _actor_loss(actor_params, traj_minibatch, gae_minibatch):
                        pi = actor.apply(actor_params, traj_minibatch.obs)
                        log_prob = pi.log_prob(traj_minibatch.action)
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
                        approx_kl_backward = ((ratio - 1) - logratio).mean()
                        approx_kl_forward = (ratio * logratio - (ratio - 1)).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["clip_eps"])
                        loss = loss_actor - config["ent_coef"] * entropy
                        aux = {
                            "actor_loss_entreg": loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio": ratio,
                            "approx_kl_backward": approx_kl_backward,
                            "approx_kl_forward": approx_kl_forward,
                            "clip_frac": clip_frac,
                        }
                        return loss, aux

                    def _critic_loss(critic_params, traj_minibatch, targets_minibatch):
                        value = critic.apply(critic_params, traj_minibatch.obs)
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
                        return value_loss, {"value_loss": value_loss}

                    actor_grad_fn = jax.value_and_grad(_actor_loss, has_aux=True)
                    critic_grad_fn = jax.value_and_grad(_critic_loss, has_aux=True)

                    (actor_loss_val, actor_aux), actor_grads = actor_grad_fn(
                        actor_train_state.params,
                        traj_minibatch,
                        advantages_minibatch,
                    )
                    (critic_loss_val, critic_aux), critic_grads = critic_grad_fn(
                        critic_train_state.params,
                        traj_minibatch,
                        targets_minibatch,
                    )

                    total_loss_scalar = actor_loss_val + critic_loss_val
                    updated_actor_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    updated_critic_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    cos_sim_a = cosine_similarity(actor_grads, running_grad_actor)
                    cos_sim_c = cosine_similarity(critic_grads, running_grad_critic)
                    cos_sim_mu_a = cosine_similarity(
                        actor_grads, updated_actor_state.opt_state[1][0].mu
                    )
                    cos_sim_mu_c = cosine_similarity(
                        critic_grads, updated_critic_state.opt_state[1][0].mu
                    )

                    aux = {
                        **actor_aux,
                        **critic_aux,
                        "grad_norm_actor": pytree_norm(actor_grads),
                        "grad_norm_critic": pytree_norm(critic_grads),
                        "mu_norm_actor": pytree_norm(
                            updated_actor_state.opt_state[1][0].mu
                        ),
                        "mu_norm_critic": pytree_norm(
                            updated_critic_state.opt_state[1][0].mu
                        ),
                        "nu_norm_actor": pytree_norm(
                            updated_actor_state.opt_state[1][0].nu
                        ),
                        "nu_norm_critic": pytree_norm(
                            updated_critic_state.opt_state[1][0].nu
                        ),
                        "cosine_similarity_actor": cos_sim_a,
                        "cosine_similarity_critic": cos_sim_c,
                        "cosine_similarity_mu_actor": cos_sim_mu_a,
                        "cosine_similarity_mu_critic": cos_sim_mu_c,
                    }

                    total_loss = (total_loss_scalar, aux)

                    return (
                        (
                            updated_actor_state,
                            updated_critic_state,
                            actor_grads,
                            critic_grads,
                        ),
                        total_loss,
                    )

                (
                    final_actor_state,
                    final_critic_state,
                    final_running_grad_actor,
                    final_running_grad_critic,
                ), total_loss = jax.lax.scan(
                    _update_minibatch,
                    (
                        actor_train_state,
                        critic_train_state,
                        running_grad_actor,
                        running_grad_critic,
                    ),
                    minibatches,
                )

                update_state = Updatestate(
                    actor_train_state=final_actor_state,
                    critic_train_state=final_critic_state,
                    running_grad_actor=final_running_grad_actor,
                    running_grad_critic=final_running_grad_critic,
                    traj_batch=traj_batch,
                    advantages=advantages,
                    targets=targets,
                    rng=rng,
                )

                return update_state, total_loss

            update_state = Updatestate(
                actor_train_state=actor_train_state,
                critic_train_state=critic_train_state,
                running_grad_actor=runner_state.running_grad_actor,
                running_grad_critic=runner_state.running_grad_critic,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=targets,
                rng=rng,
            )

            update_state, loss_info = jax.lax.scan(
                _update_epoch,
                update_state,
                None,
                config["num_epochs"],
            )

            # Episode metrics in one pass over the rollout. Carries
            # (partial_return, steps_into_episode) match RunnerState at the *start* of this
            # rollout, so episodes that began in a prior update (or mid-rollout) still get
            # full return and length at terminal steps. Env steps do not mutate
            # cumulative_return; we recompute from traj_batch rewards here.
            reward = traj_batch.reward
            done = traj_batch.new_done

            def _rollout_episode_metrics(carry, inputs):
                partial_return, timestep_carry = carry
                reward, done = inputs
                new_return = partial_return + reward
                new_len = timestep_carry + 1
                return_at_done = jnp.where(done, new_return, 0.0)
                len_at_done = jnp.where(done, new_len.astype(jnp.float32), 0.0)
                next_partial = jnp.where(
                    done, jnp.zeros_like(partial_return), new_return
                )
                next_timestep = jnp.where(done, jnp.zeros_like(timestep_carry), new_len)
                return (next_partial, next_timestep), (return_at_done, len_at_done)

            (new_cumulative_return, _), (ret_at_done, len_at_done) = jax.lax.scan(
                _rollout_episode_metrics,
                (runner_state.cumulative_return, initial_timesteps),
                (reward, done),
            )
            num_episodes_completed = done.sum()
            returns_avg = jnp.where(
                num_episodes_completed > 0,
                ret_at_done.sum() / num_episodes_completed,
                0.0,
            )
            episode_length_avg = jnp.where(
                num_episodes_completed > 0,
                len_at_done.sum() / num_episodes_completed,
                0.0,
            )

            # log network stats (actor and critic separately)
            def _param_stats(params, prefix: str):
                network_leaves = jax.tree.leaves(params)
                flat_network = jnp.concatenate([jnp.ravel(x) for x in network_leaves])
                return {
                    f"{prefix}_network_l1": jnp.sum(jnp.abs(flat_network)),
                    f"{prefix}_network_l2": jnp.linalg.norm(flat_network),
                    f"{prefix}_network_linfty": jnp.max(jnp.abs(flat_network)),
                    f"{prefix}_network_mu": jnp.mean(flat_network),
                    f"{prefix}_network_std": jnp.std(flat_network),
                    f"{prefix}_network_max": jnp.max(flat_network),
                    f"{prefix}_network_min": jnp.min(flat_network),
                }

            # metric_actor = _param_stats(update_state.actor_train_state.params, "actor")
            # metric_critic = _param_stats(update_state.critic_train_state.params, "critic")

            # log info
            total_loss, loss_info = loss_info
            loss_info["total_loss"] = total_loss
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            # wandb
            metric = {}
            metric["update_step"] = runner_state.update_step
            metric["env_step"] = (
                runner_state.update_step
                * config["num_envs"]
                * config["num_steps_per_env_per_update"]
            )
            metric["return"] = returns_avg
            metric["episode_length"] = episode_length_avg
            metric["num_episodes_completed"] = num_episodes_completed
            # metric.update(metric_actor)
            # metric.update(metric_critic)
            metric.update(loss_info)

            def callback(exp_id, metric):
                np_log_dict = {k: np.array(v) for k, v in metric.items()}
                LOGGER.log(int(exp_id), np_log_dict)

            jax.experimental.io_callback(callback, None, exp_id, metric)

            runner_state = RunnerState(
                actor_train_state=update_state.actor_train_state,
                critic_train_state=update_state.critic_train_state,
                running_grad_actor=update_state.running_grad_actor,
                running_grad_critic=update_state.running_grad_critic,
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
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            running_grad_actor=running_grad_actor,
            running_grad_critic=running_grad_critic,
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
            length=config["num_update_steps"],
        )
        return final_runner_state, metrics_batch

    return train


@hydra.main(version_base=None, config_path="./", config_name="config_ppo_discrete")
def main(config):
    try:
        # vmap and compile
        config = OmegaConf.to_container(config)
        config["num_update_steps"] = (
            config["total_timesteps"]
            // config["num_envs"]
            // config["num_steps_per_env_per_update"]
        )
        config["num_gradient_steps"] = (
            config["num_update_steps"]
            * config["num_epochs"]
            * config["num_minibatches"]
        )

        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Starting compile...")
        train_vmap = jax.vmap(make_train(config))
        train_vjit = jax.block_until_ready(jax.jit(train_vmap))
        print("Compile finished...")

        # wandb
        job_type = f"{config['job_type']}_{config['env_name']}_{config['optimizer']}"
        if config["optimizer"] == "myano":
            job_type += f"_gamma_{config['myano_gamma']}"
        group = config["job_type"] + datetime.datetime.now().strftime(
            "_%Y-%m-%d_%H-%M-%S"
        )
        global LOGGER
        LOGGER = WandbMultiLogger(
            project=config["project"],
            group=group,
            job_type=job_type,
            config=config,
            mode=(lambda: "online" if config["wandb"] else "disabled")(),
            seed=config["seed"],
            num_seeds=config["num_seeds"],
        )

        # run
        print("Running...")
        out = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
    finally:
        LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
