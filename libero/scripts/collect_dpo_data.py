#!/usr/bin/env python
"""Counterfactual DPO data collection for delay robustness.

For each (task, init_state), runs the model with delay=k in the simulator.
At each step t:
  - Model sees (image[t-k], state[t]) → action_delayed (rejected)
  - Model sees (image[t], state[t])   → action_clean   (preferred / privileged teacher)
  - Stores (stale_obs, action_clean, action_delayed) as a preference pair

Uses batched parallel envs for speed (~30min for 10 tasks × 10 init_states).

Usage:
    python -m vlash.collect_dpo_data \
        --policy_path=outputs/train/pi05_mask_state_libero/checkpoints/030000/pretrained_model \
        --task=libero_spatial --num_tasks=10 --inits_per_task=10 \
        --delay=3 --output_path=outputs/dpo_data/spatial_d3.pt
"""

import argparse
import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

import vlash.configs  # noqa: F401

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from lerobot.envs.utils import preprocess_observation
from vlash.libero_gym import LiberoEnvWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_path", type=str, required=True)
    p.add_argument("--task", type=str, default="libero_spatial")
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--inits_per_task", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8, help="Parallel envs per batch")
    p.add_argument("--delay", type=int, default=3)
    p.add_argument("--max_episode_steps", type=int, default=230)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--naive_reject", action="store_true",
                   help="Use naive async (stale image + stale state) for rejected instead of VLASH (stale image + current state)")
    return p.parse_args()


def collect_task_data(policy, suite_name, task_name, task_language,
                      init_state_ids, delay, max_steps, device, naive_reject=False):
    """Collect preference pairs for one task with batched envs.

    For each step, queries model twice:
      1. stale obs (image[t-k] + state[t]) → action_delayed (rejected)
      2. clean obs (image[t] + state[t])   → action_clean (preferred)

    Returns list of dicts with step-level preference data.
    """
    N = len(init_state_ids)

    # Create and reset all envs
    envs = []
    observations = []
    for sid in init_state_ids:
        env = LiberoEnvWrapper(suite_name=suite_name, task_name=task_name, init_state_id=sid)
        obs, _ = env.reset()
        obs = preprocess_observation(obs)
        obs = {k: v.to(device) for k, v in obs.items()}
        observations.append(obs)
        envs.append(env)

    obs_keys = [k for k in observations[0].keys() if k != "task"]
    img_keys = [k for k in obs_keys if k != "observation.state"]

    obs_buffers = [[] for _ in range(N)]
    dones = [False] * N
    successes = [False] * N
    preference_pairs = []  # all step-level pairs across envs

    for step in range(max_steps):
        active = [i for i in range(N) if not dones[i]]
        if not active:
            break
        N_active = len(active)

        # Store current obs in delay buffers
        for i in active:
            obs_buffers[i].append(deepcopy(observations[i]))

        # --- Query 1: Stale obs → action_delayed (rejected) ---
        batch_states = []
        batch_stale_images = {k: [] for k in img_keys}
        batch_clean_images = {k: [] for k in img_keys}

        batch_stale_states = []
        for i in active:
            delay_idx = max(0, len(obs_buffers[i]) - 1 - delay)
            delayed = obs_buffers[i][delay_idx]
            current = observations[i]

            batch_states.append(current["observation.state"])
            batch_stale_states.append(delayed["observation.state"])
            for k in img_keys:
                batch_stale_images[k].append(delayed[k])
                batch_clean_images[k].append(current[k])

        state_tensor = torch.cat(batch_states, dim=0)
        stale_state_tensor = torch.cat(batch_stale_states, dim=0)

        # Build stale input
        # naive_reject: use stale state (naive async), otherwise use current state (VLASH)
        stale_input = {
            "observation.state": stale_state_tensor if naive_reject else state_tensor,
            "task": [task_language] * N_active,
        }
        for k in img_keys:
            stale_input[k] = torch.cat(batch_stale_images[k], dim=0)

        # Build clean input (same state, current images)
        clean_input = {
            "observation.state": state_tensor,
            "task": [task_language] * N_active,
        }
        for k in img_keys:
            clean_input[k] = torch.cat(batch_clean_images[k], dim=0)

        with torch.no_grad():
            # Shared noise for both queries — ensures difference comes from obs only
            shared_noise = torch.randn(
                N_active, policy.config.chunk_size, policy.config.max_action_dim,
                device=device,
            )

            # Stale obs → delayed action
            norm_stale = policy.normalize_inputs(deepcopy(stale_input))
            imgs_s, masks_s = policy.prepare_images(norm_stale)
            state_s = policy.prepare_state(norm_stale)
            tok_s, tmask_s = policy.prepare_language(norm_stale)
            actions_delayed = policy.model.sample_actions(
                imgs_s, masks_s, tok_s, tmask_s, state_s, noise=shared_noise.clone()
            )

            # Clean obs → privileged action (same noise!)
            norm_clean = policy.normalize_inputs(deepcopy(clean_input))
            imgs_c, masks_c = policy.prepare_images(norm_clean)
            state_c = policy.prepare_state(norm_clean)
            tok_c, tmask_c = policy.prepare_language(norm_clean)
            actions_clean = policy.model.sample_actions(
                imgs_c, masks_c, tok_c, tmask_c, state_c, noise=shared_noise.clone()
            )

        # Unnormalize actions
        action_dim = 7
        delayed_trimmed = actions_delayed[:, :, :action_dim]
        clean_trimmed = actions_clean[:, :, :action_dim]
        delayed_unnorm = policy.unnormalize_outputs({"action": delayed_trimmed})["action"]
        clean_unnorm = policy.unnormalize_outputs({"action": clean_trimmed})["action"]

        # Store preference pairs (only when delay is actually active: step >= delay)
        if step >= delay:
            for idx, i in enumerate(active):
                preference_pairs.append({
                    # Stale observation (input for DPO training)
                    "stale_images": {k: batch_stale_images[k][idx].cpu()
                                     for k in img_keys},
                    "state": batch_states[idx].cpu(),
                    "task": task_language,
                    # Preferred action (from clean obs - privileged teacher)
                    "action_preferred": actions_clean[idx].cpu(),  # normalized
                    # Rejected action (from stale obs - delayed model)
                    "action_rejected": actions_delayed[idx].cpu(),  # normalized
                })

        # Step envs with delayed action (follow the delayed policy)
        for idx, i in enumerate(active):
            action_np = delayed_unnorm[idx, 0, :action_dim].cpu().numpy()
            obs_raw, reward, terminated, truncated, info = envs[i].step(action_np)

            if info.get("is_success", False):
                successes[i] = True
                dones[i] = True
            elif terminated or truncated:
                dones[i] = True
            else:
                obs = preprocess_observation(obs_raw)
                obs = {k: v.to(device) for k, v in obs.items()}
                observations[i] = obs

    for env in envs:
        env.close()

    n_success = sum(successes)
    return preference_pairs, n_success, N


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Config: delay={args.delay}, tasks={args.num_tasks}, "
                 f"inits_per_task={args.inits_per_task}, batch={args.batch_size}")

    # Load policy
    logging.info(f"Loading policy from {args.policy_path}")
    from vlash.policies.factory import make_policy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    import vlash.configs  # noqa: register configs

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    env_cfg = LiberoEnv(task=args.task)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.to(device)
    policy.eval()

    total_params = sum(p.numel() for p in policy.parameters())
    logging.info(f"Model: {total_params/1e6:.0f}M params")

    # Get tasks
    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()[args.task]()
    n_tasks = min(args.num_tasks, bench.n_tasks)
    task_names = [bench.get_task(i).name for i in range(n_tasks)]
    task_languages = [bench.get_task(i).language for i in range(n_tasks)]
    logging.info(f"Collecting data for {n_tasks} tasks")

    all_pairs = []
    total_success = 0
    total_episodes = 0

    for task_id in range(n_tasks):
        task_start = time.time()
        all_init_ids = list(range(args.inits_per_task))

        task_pairs = []
        task_success = 0
        task_total = 0

        # Process in batches
        for batch_start in range(0, len(all_init_ids), args.batch_size):
            batch_ids = all_init_ids[batch_start:batch_start + args.batch_size]

            pairs, n_succ, n_total = collect_task_data(
                policy, args.task, task_names[task_id], task_languages[task_id],
                batch_ids, delay=args.delay,
                max_steps=args.max_episode_steps, device=device,
                naive_reject=args.naive_reject,
            )
            task_pairs.extend(pairs)
            task_success += n_succ
            task_total += n_total

        task_time = time.time() - task_start
        logging.info(f"  task {task_id}/{n_tasks}: {len(task_pairs)} pairs, "
                     f"{task_success}/{task_total} success ({task_time:.0f}s)")

        all_pairs.extend(task_pairs)
        total_success += task_success
        total_episodes += task_total

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "pairs": all_pairs,
        "metadata": {
            "delay": args.delay,
            "task": args.task,
            "num_tasks": n_tasks,
            "inits_per_task": args.inits_per_task,
            "total_pairs": len(all_pairs),
            "total_success": total_success,
            "total_episodes": total_episodes,
        }
    }, output_path)

    logging.info(f"Saved {len(all_pairs)} preference pairs to {output_path}")
    logging.info(f"Overall: {total_success}/{total_episodes} success "
                 f"({total_success/total_episodes:.1%})")


if __name__ == "__main__":
    main()
