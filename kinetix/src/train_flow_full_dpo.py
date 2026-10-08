"""Full Counterfactual DPO training for Kinetix.

Two-phase training in one script:
  Phase 1: Load pre-trained VLASH checkpoint as reference policy
  Phase 2: Fine-tune with DPO loss using model-generated preference pairs

At each training step:
  - obs_oracle = full future obs (delay=0, privileged teacher)
  - obs_naive  = full stale obs (delay=k, no compensation)
  - obs_vlash  = mixed obs (env@stale + robot@future, VLASH style)
  - action_preferred = ref_policy(obs_oracle)  [model-generated, stop_gradient]
  - action_rejected  = ref_policy(obs_naive)   [model-generated, stop_gradient]
  - Loss = SFT(expert_action) + λ * DPO(action_preferred, action_rejected)

Usage:
    .venv/bin/python src/train_flow_full_dpo.py \
        --config.run-path official_assets/expert \
        --config.ref-checkpoint-dir checkpoints/vlash-kinetix-policy-async5 \
        --config.async-interval 5 \
        --config.dpo-beta 1.0 --config.dpo-lambda 0.1
"""

import concurrent.futures
import dataclasses
import functools
import pathlib
import pickle
from typing import Sequence

import einops
from flax import struct
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import eval_flow as _eval
import generate_data
import model as _model
import train_expert
import compute_robot_indices

WANDB_PROJECT = "rtc-kinetix-dpo"


@dataclasses.dataclass(frozen=True)
class Config:
    run_path: str
    ref_checkpoint_dir: str  # Path to pre-trained VLASH checkpoint
    ref_step: int = -1       # Which epoch to load (-1 = last)
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
        "worlds/l/catapult.json",
        "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        "worlds/l/mjc_half_cheetah.json",
        "worlds/l/mjc_swimmer.json",
        "worlds/l/mjc_walker.json",
        "worlds/l/h17_unicycle.json",
        "worlds/l/chain_lander.json",
        "worlds/l/catcher_v3.json",
        "worlds/l/trampoline.json",
        "worlds/l/car_launch.json",
    )
    batch_size: int = 512
    num_epochs: int = 32
    seed: int = 0
    output_dir: str = "logs-dpo"

    eval: _eval.EvalConfig = _eval.EvalConfig()

    learning_rate: float = 3e-4
    grad_norm_clip: float = 10.0
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 1000
    lr_schedule: str = "constant"  # "constant", "cosine", or "trapezoid"
    lr_decay_fraction: float = 0.2  # fraction of training for final decay (trapezoid only)
    async_interval: int = 5

    # DPO parameters
    dpo_beta: float = 1.0
    dpo_lambda: float = 0.1
    dpo_failure_threshold: float = 0.0  # Only apply DPO where action_dist > threshold
    dpo_position_weighted: bool = False  # Use position-weighted DPO for horizon robustness
    dpo_weight_mode: str = "truncated"  # "linear" (1→0 across chunk) or "truncated" (first 2 only)
    dpo_full_margin: bool = False  # Include reference model terms in DPO margin (mathematically correct)
    dpo_reject_vlash: bool = False  # Use VLASH obs (stale env + future robot) as rejected instead of naive (all stale)
    dpo_include_d0: bool = False  # Include d=0 samples in DPO mean (original DPO behavior)
    sft_lambda: float = 1.0  # Weight on SFT anchor loss; set to 0 to disable (ablation)
    num_flow_steps_ref: int = 5  # Denoising steps for reference policy inference
    # Cap the DPO training-delay pool. -1 = no cap (all sampled delays 1..async_interval-1
    # are used for DPO). Set to e.g. 2 to train preference pairs only on d∈{1,2}.
    # SFT delay range is always [0, async_interval) regardless of this cap.
    dpo_max_delay: int = -1
    # If True, score preferred action under obs_oracle (fresh) and rejected
    # under obs_rejected (stale) — "matched-condition scoring" ablation.
    # Default (False) scores both under obs_vlash (mixed stale deployment ctx).
    dpo_matched_context: bool = False


@struct.dataclass
class EpochCarry:
    rng: jax.Array
    train_state: nnx.State
    graphdef: nnx.GraphDef[tuple[_model.FlowPolicy, nnx.Optimizer]]


def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    mesh = jax.make_mesh((jax.local_device_count(),), ("level",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("level"))

    action_chunk_size = config.eval.model.action_chunk_size

    # Load data
    def load_data(level_path: str):
        level_name = level_path.replace("/", "_").replace(".json", "")
        return dict(np.load(pathlib.Path(config.run_path) / "data" / f"{level_name}.npz"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(load_data, config.level_paths))
    with jax.default_device(jax.devices("cpu")[0]):
        data = jax.tree.map(lambda *x: einops.rearrange(jnp.stack(x), "l s e ... -> l (e s) ..."), *data)
        max_async = max(0, config.async_interval - 1)
        valid_steps = data["obs"].shape[1] - action_chunk_size - max_async + 1
        data = jax.tree.map(
            lambda x: x[:, : (valid_steps // config.batch_size) * config.batch_size + action_chunk_size + max_async - 1], data
        )
        data = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                x.shape, sharding,
                [jax.device_put(y, d) for y, d in zip(jnp.split(x, jax.local_device_count()), jax.local_devices(), strict=True)],
            ), data,
        )

    data: generate_data.Data = generate_data.Data(**data)
    steps_per_epoch = valid_steps // config.batch_size
    total_train_steps = steps_per_epoch * config.num_epochs
    print(f"Truncated data to {data.obs.shape[1]:_} steps ({steps_per_epoch:_} batches/epoch, {total_train_steps:_} total)")

    obs_dim = data.obs.shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    # Compute robot masks for async training
    robot_masks = jnp.stack([compute_robot_indices.compute_robot_mask(p, obs_dim) for p in config.level_paths])
    robot_masks = jax.device_put(robot_masks, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("level")))
    print(f"Full counterfactual DPO: beta={config.dpo_beta}, lambda={config.dpo_lambda}")

    # Load reference policy checkpoint
    print(f"Loading reference policy from {config.ref_checkpoint_dir}")
    ref_state_dicts = []
    for level_path in config.level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(config.ref_checkpoint_dir).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        with (log_dirs[config.ref_step] / "policies" / f"{level_name}.pkl").open("rb") as f:
            ref_state_dicts.append(pickle.load(f))
    ref_state_dicts = jax.tree.map(lambda *x: jnp.stack(x), *ref_state_dicts)
    ref_state_dicts = jax.device_put(ref_state_dicts, sharding)
    print(f"Loaded reference policy from epoch {config.ref_step}")

    # Build LR schedule
    if config.lr_schedule == "cosine":
        lr_sched = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=config.learning_rate,
            warmup_steps=config.lr_warmup_steps,
            decay_steps=total_train_steps,
            end_value=config.learning_rate * 0.01,
        )
    elif config.lr_schedule == "trapezoid":
        # Warmup → constant → linear decay to 10%
        decay_steps = int(total_train_steps * config.lr_decay_fraction)
        constant_steps = total_train_steps - config.lr_warmup_steps - decay_steps
        lr_sched = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0, config.learning_rate, config.lr_warmup_steps),
                optax.constant_schedule(config.learning_rate),
                optax.linear_schedule(config.learning_rate, config.learning_rate * 0.1, decay_steps),
            ],
            boundaries=[config.lr_warmup_steps, config.lr_warmup_steps + constant_steps],
        )
    else:
        lr_sched = optax.warmup_constant_schedule(0, config.learning_rate, config.lr_warmup_steps)
    print(f"LR schedule: {config.lr_schedule}, total_steps={total_train_steps}")

    @functools.partial(jax.jit, in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def init(rng: jax.Array) -> EpochCarry:
        rng, key = jax.random.split(rng)
        policy = _model.FlowPolicy(
            obs_dim=obs_dim, action_dim=action_dim, config=config.eval.model, rngs=nnx.Rngs(key),
        )
        total_params = sum(x.size for x in jax.tree.leaves(nnx.state(policy, nnx.Param)))
        print(f"Total params: {total_params:,}")
        optimizer = nnx.Optimizer(
            policy,
            optax.chain(
                optax.clip_by_global_norm(config.grad_norm_clip),
                optax.adamw(lr_sched, weight_decay=config.weight_decay),
            ),
        )
        graphdef, train_state = nnx.split((policy, optimizer))
        return EpochCarry(rng, train_state, graphdef)

    @functools.partial(jax.jit, donate_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def train_epoch(epoch_carry: EpochCarry, level: kenv_state.EnvState,
                    data: generate_data.Data, robot_mask, ref_state_dict):
        def train_minibatch(carry: tuple[jax.Array, nnx.State], batch_idxs: jax.Array):
            rng, train_state = carry
            policy, optimizer = nnx.merge(epoch_carry.graphdef, train_state)

            rng, key = jax.random.split(rng)

            def loss_fn(policy: _model.FlowPolicy):
                obs_current = data.obs[batch_idxs]

                # PURE SFT FAST PATH: when dpo_lambda=0, match VLASH's train_flow.py exactly
                # (RNG split of 2 keys, no ref policy creation, no auxiliary computation)
                if config.dpo_lambda == 0.0:
                    rng_local, key_delay = jax.random.split(key)
                    delays = jax.random.randint(key_delay, (batch_idxs.shape[0],), 0, config.async_interval)
                    obs_future = data.obs[batch_idxs + delays]
                    obs_vlash = jnp.where(robot_mask[None, :], obs_future, obs_current)
                    action_indices = batch_idxs[:, None] + delays[:, None] + jnp.arange(action_chunk_size)[None, :]
                    action_chunks = data.action[action_indices]
                    done_chunks = data.done[action_indices]
                    done_idxs = jnp.where(jnp.any(done_chunks, axis=-1), jnp.argmax(done_chunks, axis=-1), action_chunk_size)
                    action_chunks = jnp.where(jnp.arange(action_chunk_size)[None, :, None] >= done_idxs[:, None, None], 0.0, action_chunks)
                    return policy.loss(rng_local, obs_vlash, action_chunks)

                # Sample random delay (SFT uses full range, DPO uses curriculum-controlled range)
                rng_local, key_delay, key_ref_oracle, key_ref_naive = jax.random.split(key, 4)
                delays = jax.random.randint(key_delay, (batch_idxs.shape[0],), 0, config.async_interval)

                # VLASH mixed obs: env@current + robot@future
                obs_future = data.obs[batch_idxs + delays]
                obs_vlash = jnp.where(robot_mask[None, :], obs_future, obs_current)

                # Shifted action targets (preferred expert action)
                action_indices = batch_idxs[:, None] + delays[:, None] + jnp.arange(action_chunk_size)[None, :]
                action_chunks = data.action[action_indices]
                done_chunks = data.done[action_indices]
                done_idxs = jnp.where(jnp.any(done_chunks, axis=-1), jnp.argmax(done_chunks, axis=-1), action_chunk_size)
                action_chunks = jnp.where(jnp.arange(action_chunk_size)[None, :, None] >= done_idxs[:, None, None], 0.0, action_chunks)

                # --- Full counterfactual DPO (single forward pass) ---
                # Create reference policy (frozen)
                ref_policy = _model.FlowPolicy(
                    obs_dim=obs_dim, action_dim=action_dim,
                    config=config.eval.model, rngs=nnx.Rngs(0),
                )
                ref_graphdef, ref_state = nnx.split(ref_policy)
                ref_state.replace_by_pure_dict(ref_state_dict)
                ref_policy = nnx.merge(ref_graphdef, ref_state)

                # Oracle obs (full future = no delay)
                obs_oracle = obs_future
                # Rejected obs: VLASH (stale env + future robot) or naive (all stale)
                if config.dpo_reject_vlash:
                    obs_rejected = obs_vlash  # VLASH-style: stale env + future robot
                else:
                    obs_rejected = obs_current  # Naive: all stale

                # Model-generated actions from reference policy (stop gradient)
                # CRITICAL: use same RNG for both queries so difference comes from obs only
                action_preferred = jax.lax.stop_gradient(
                    ref_policy.action(key_ref_oracle, obs_oracle, config.num_flow_steps_ref)
                )
                action_rejected = jax.lax.stop_gradient(
                    ref_policy.action(key_ref_oracle, obs_rejected, config.num_flow_steps_ref)
                )

                # Single forward pass: SFT on expert + DPO on model-generated
                # Share noise and timestep between SFT and DPO
                noise_rng, time_rng = jax.random.split(rng_local, 2)
                time = jax.random.uniform(time_rng, (batch_idxs.shape[0],))
                noise = jax.random.normal(noise_rng, shape=action_chunks.shape)

                # Interpolate using expert action (SFT target)
                x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * action_chunks
                u_expert = action_chunks - noise  # SFT target velocity

                # One forward pass
                v_t = policy(obs_vlash, x_t, time)

                # SFT loss
                sft_loss = jnp.mean(jnp.square(v_t - u_expert))

                # DPO loss using same v_t prediction
                u_preferred = action_preferred - noise
                u_rejected = action_rejected - noise

                if config.dpo_matched_context:
                    # Matched-condition scoring: each action scored under its
                    # own generating observation. Requires 2 extra forwards per
                    # batch (v_t_pref under obs_oracle, v_t_rej under obs_rejected).
                    x_t_pref = (1 - time[:, None, None]) * noise + time[:, None, None] * action_preferred
                    x_t_rej  = (1 - time[:, None, None]) * noise + time[:, None, None] * action_rejected
                    v_t_pref = policy(obs_oracle,  x_t_pref, time)
                    v_t_rej  = policy(obs_rejected, x_t_rej,  time)
                    mse_pref = jnp.sum(jnp.square(v_t_pref - u_preferred), axis=(-2, -1))
                    mse_rej  = jnp.sum(jnp.square(v_t_rej  - u_rejected), axis=(-2, -1))
                elif config.dpo_position_weighted:
                    if config.dpo_weight_mode == "linear":
                        pos_w = jnp.linspace(1.0, 0.0, action_chunk_size)[None, :, None]
                    else:
                        pos_w = jnp.zeros(action_chunk_size)
                        pos_w = pos_w.at[0].set(1.0).at[1].set(0.5)
                        pos_w = pos_w[None, :, None]
                    mse_pref = jnp.sum(pos_w * jnp.square(v_t - u_preferred), axis=(-2, -1))
                    mse_rej = jnp.sum(pos_w * jnp.square(v_t - u_rejected), axis=(-2, -1))
                else:
                    mse_pref = jnp.sum(jnp.square(v_t - u_preferred), axis=(-2, -1))
                    mse_rej = jnp.sum(jnp.square(v_t - u_rejected), axis=(-2, -1))

                # Full margin: include reference model terms (mathematically correct DPO)
                if config.dpo_full_margin:
                    if config.dpo_matched_context:
                        v_ref_pref = jax.lax.stop_gradient(ref_policy(obs_oracle,  x_t_pref, time))
                        v_ref_rej  = jax.lax.stop_gradient(ref_policy(obs_rejected, x_t_rej,  time))
                        mse_ref_pref = jnp.sum(jnp.square(v_ref_pref - u_preferred), axis=(-2, -1))
                        mse_ref_rej  = jnp.sum(jnp.square(v_ref_rej  - u_rejected), axis=(-2, -1))
                    else:
                        v_ref = jax.lax.stop_gradient(ref_policy(obs_vlash, x_t, time))
                        if config.dpo_position_weighted:
                            mse_ref_pref = jnp.sum(pos_w * jnp.square(v_ref - u_preferred), axis=(-2, -1))
                            mse_ref_rej = jnp.sum(pos_w * jnp.square(v_ref - u_rejected), axis=(-2, -1))
                        else:
                            mse_ref_pref = jnp.sum(jnp.square(v_ref - u_preferred), axis=(-2, -1))
                            mse_ref_rej = jnp.sum(jnp.square(v_ref - u_rejected), axis=(-2, -1))
                    margin = (mse_rej - mse_pref) + jax.lax.stop_gradient(mse_ref_pref - mse_ref_rej)
                else:
                    margin = mse_rej - mse_pref

                action_dist = jnp.sqrt(jnp.sum(jnp.square(action_preferred - action_rejected), axis=(-2, -1)))
                # Skip d=0 (preferred == rejected, no signal)
                is_confused = (action_dist > config.dpo_failure_threshold) & (delays >= 1)
                # Optional cap on DPO training-delay range (e.g. dpo_max_delay=2 → only {1,2})
                if config.dpo_max_delay >= 0:
                    is_confused = is_confused & (delays <= config.dpo_max_delay)

                dpo_per_sample = -jax.nn.log_sigmoid(config.dpo_beta * margin)
                if config.dpo_include_d0 and config.dpo_max_delay < 0:
                    # Original DPO behavior: simple mean over all samples (d=0 included)
                    # d=0 contributes log(2) to loss but 0 gradient, acts as implicit regularizer
                    dpo_loss = dpo_per_sample.mean()
                elif config.dpo_include_d0:
                    # include d=0 but respect delay cap
                    keep = (delays <= config.dpo_max_delay)
                    dpo_loss = jnp.where(keep, dpo_per_sample, 0.0).sum() / jnp.maximum(keep.sum(), 1)
                else:
                    n_confused = jnp.maximum(is_confused.sum(), 1)
                    dpo_loss = jnp.where(is_confused, dpo_per_sample, 0.0).sum() / n_confused

                total_loss = config.sft_lambda * sft_loss + config.dpo_lambda * dpo_loss
                return total_loss

            loss, grads = nnx.value_and_grad(loss_fn)(policy)
            info = {"loss": loss, "grad_norm": optax.global_norm(grads)}
            optimizer.update(grads)
            _, train_state = nnx.split((policy, optimizer))
            return (rng, train_state), info

        # Shuffle
        rng, key = jax.random.split(epoch_carry.rng)
        max_async = max(0, config.async_interval - 1)
        permutation = jax.random.permutation(key, data.obs.shape[0] - action_chunk_size - max_async + 1)
        permutation = permutation.reshape(-1, config.batch_size)
        # Train
        (rng, train_state), train_info = jax.lax.scan(
            train_minibatch, (epoch_carry.rng, epoch_carry.train_state), permutation
        )
        train_info = jax.tree.map(lambda x: x.mean(), train_info)
        # Eval
        rng, key = jax.random.split(rng)
        eval_policy, _ = nnx.merge(epoch_carry.graphdef, train_state)
        eval_info = {}
        for horizon in range(1, config.eval.model.action_chunk_size + 1):
            eval_config = dataclasses.replace(config.eval, execute_horizon=horizon)
            info, _ = _eval.eval(eval_config, env, key, level, eval_policy, env_params, static_env_params)
            eval_info.update({f"{k}_{horizon}": v for k, v in info.items()})
        video = None

        return (
            dataclasses.replace(epoch_carry, rng=rng, train_state=train_state),
            {**train_info, **eval_info},
            video,
        )

    # Initialize
    rng = jax.random.PRNGKey(config.seed)
    rngs = jax.random.split(rng, len(config.level_paths))
    rngs = jax.device_put(rngs, sharding)
    epoch_carry = init(rngs)

    # Initialize from reference checkpoint (start from VLASH, not random)
    print("Initializing policy weights from reference checkpoint")
    epoch_carry = dataclasses.replace(
        epoch_carry,
        train_state=jax.tree.map(
            lambda s, r: jnp.where(jnp.ones_like(s, dtype=bool), r, s) if s.shape == r.shape else s,
            epoch_carry.train_state,
            # Only copy policy params, not optimizer state
            epoch_carry.train_state,  # placeholder - actual init handled below
        )
    )
    # Re-init with ref weights
    @functools.partial(jax.jit, in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def init_from_ref(rng, ref_state_dict):
        rng, key = jax.random.split(rng)
        policy = _model.FlowPolicy(
            obs_dim=obs_dim, action_dim=action_dim, config=config.eval.model, rngs=nnx.Rngs(key),
        )
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(ref_state_dict)
        policy = nnx.merge(graphdef, state)
        optimizer = nnx.Optimizer(
            policy,
            optax.chain(
                optax.clip_by_global_norm(config.grad_norm_clip),
                optax.adamw(lr_sched, weight_decay=config.weight_decay),
            ),
        )
        graphdef, train_state = nnx.split((policy, optimizer))
        return EpochCarry(rng, train_state, graphdef)

    epoch_carry = init_from_ref(rngs, ref_state_dicts)
    print("Policy initialized from reference checkpoint")

    # W&B setup
    wandb_run = wandb.init(project=WANDB_PROJECT, config=dataclasses.asdict(config), mode="offline")

    # Training loop
    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dummy mask if async_interval=0
    dummy_masks = jnp.zeros((len(config.level_paths), obs_dim), dtype=bool)
    masks = robot_masks if robot_masks is not None else dummy_masks

    for epoch in (pbar := tqdm.tqdm(range(config.num_epochs))):
        epoch_carry, info, video = train_epoch(epoch_carry, levels, data, masks, ref_state_dicts)
        info = jax.tree.map(lambda x: x.tolist(), info)

        # Log
        log_dict = {}
        for level_idx, level_path in enumerate(config.level_paths):
            level_name = pathlib.Path(level_path).stem
            for k, v in info.items():
                log_dict[f"{level_name}/{k}"] = v[level_idx]
        for k, v in info.items():
            log_dict[f"mean/{k}"] = np.mean(v)
        wandb_run.log(log_dict, step=epoch)
        pbar.set_postfix(loss=f"{log_dict['mean/loss']:.4f}")

        # Save checkpoints
        epoch_dir = output_dir / str(epoch) / "policies"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        policy_merged, _ = nnx.merge(epoch_carry.graphdef, epoch_carry.train_state)
        _, all_states = nnx.split(policy_merged)
        for level_idx, level_path in enumerate(config.level_paths):
            level_name = level_path.replace("/", "_").replace(".json", "")
            state_dict = jax.tree.map(lambda x: x[level_idx], all_states.to_pure_dict())
            with (epoch_dir / f"{level_name}.pkl").open("wb") as f:
                pickle.dump(state_dict, f)

    wandb_run.finish()
    print(f"Training complete! Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
