#!/usr/bin/env python
"""Delay sweep for GR00T N1.7 on LIBERO (naive asynchronous execution).

Protocol matches the rest of this project: at delay d the policy replans every
K = max(1, d) steps from an observation that is d steps stale, and the chunk it
returns is executed from index 0. d=0 therefore reduces to replanning every step
on a fresh observation, which is what the released eval does -- so the d=0 column
doubles as a reproduction check against NVIDIA's published per-suite numbers.
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import multiprocessing as mp

import numpy as np

STATE_KEYS = ["state.x", "state.y", "state.z", "state.roll", "state.pitch",
              "state.yaw", "state.gripper"]
VIDEO_KEYS = ["video.image", "video.wrist_image"]
LANG_KEY = "annotation.human.action.task_description"
ACTION_KEYS = ["action.x", "action.y", "action.z", "action.roll", "action.pitch",
               "action.yaw", "action.gripper"]


def batchify(obs):
    """Flat single-frame env obs -> the (B=1, T=1, ...) layout the wrapper wants."""
    out = {}
    for k in VIDEO_KEYS:
        out[k] = np.asarray(obs[k], dtype=np.uint8)[None, None]      # (1,1,H,W,C)
    for k in STATE_KEYS:
        out[k] = np.asarray(obs[k], dtype=np.float32)[None, None]    # (1,1,D)
    out[LANG_KEY] = [obs[LANG_KEY]]
    return out


def chunk_from_action(action, horizon):
    """Wrapper output -> (T, 7) array in the env's action-key order.

    get_action returns (actions, extra) -- the released rollout unpacks it as
    `actions, _ = policy.get_action(obs)` -- so take the first element when a
    tuple comes back.
    """
    if isinstance(action, tuple):
        action = action[0]
    cols = []
    for k in ACTION_KEYS:
        v = np.asarray(action[k])
        v = v[0] if v.ndim == 3 else v          # drop batch dim if present
        cols.append(v.reshape(v.shape[0], -1) if v.ndim > 1 else v.reshape(-1, 1))
    arr = np.concatenate(cols, axis=-1)
    return arr[:horizon]


def batchify_many(obs_list):
    """Stack several single-frame observations into one (B, T=1, ...) batch."""
    out = {}
    for k in VIDEO_KEYS:
        out[k] = np.stack([np.asarray(o[k], dtype=np.uint8) for o in obs_list])[:, None]
    for k in STATE_KEYS:
        out[k] = np.stack([np.asarray(o[k], dtype=np.float32) for o in obs_list])[:, None]
    out[LANG_KEY] = [o[LANG_KEY] for o in obs_list]
    return out


def chunks_from_batched(action, horizon, n):
    """Split a batched policy output into n per-environment (T, 7) chunks."""
    if isinstance(action, tuple):
        action = action[0]
    cols = []
    for k in ACTION_KEYS:
        v = np.asarray(action[k])
        if v.ndim == 2:                      # (B, T) -- one scalar dim per step
            v = v[..., None]
        elif v.ndim == 1:                    # degenerate: single env, single dim
            v = v[None, :, None]
        cols.append(v)
    arr = np.concatenate(cols, axis=-1)      # (B, T, 7)
    return [arr[i][:horizon] for i in range(n)]


def run_task_batched(policy, task_bddl, task_desc, init_states, n_eps, delay,
                     max_steps, horizon, ep_offset=0, batch=1, action_steps=0,
                     trace=None, tid=None):
    """Same rollout as run_task, but stepping `batch` episodes together.

    One process holding M environments beats M processes holding one each:
    the policy is loaded once instead of M times (VRAM stops scaling with
    parallelism -- 20 single-env workers already wedge a 180 GB B200), and the
    policy sees one batch=M forward per step instead of M separate batch=1
    calls queueing on the same GPU. With batch=1 this is the original loop.

    Episodes finish at different times, so only the still-running ones are fed
    to the policy and stepped; a finished environment is dropped from the batch
    rather than being stepped with dummy actions, which would change its result.
    """
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv

    K = action_steps if action_steps > 0 else max(1, delay)
    n_success = 0
    eps = list(range(ep_offset, ep_offset + n_eps))
    for start in range(0, len(eps), batch):
        group = eps[start:start + batch]
        envs = [LiberoEnv(task_bddl_file=task_bddl, task_description=task_desc)
                for _ in group]
        bufs, chunks, idxs, alive = [], [], [], []
        for env, ep in zip(envs, group):
            obs, _ = env.reset()
            if init_states is not None:
                env._env.set_init_state(init_states[ep % len(init_states)])
                obs, _, _, _, _ = env.step({k: np.zeros(1) for k in ACTION_KEYS})
            bufs.append(deque([obs] * (delay + 1), maxlen=delay + 1))
            chunks.append(None); idxs.append(0); alive.append(True)
        rec = [{"tid": tid, "ep": ep, "success": 0, "state": [], "replanned": []}
               for ep in group] if trace is not None else None

        for _ in range(max_steps):
            need = [i for i in range(len(group))
                    if alive[i] and (chunks[i] is None or idxs[i] >= K)]
            if need:
                batched = batchify_many([bufs[i][0] for i in need])
                new = chunks_from_batched(policy.get_action(batched), horizon, len(need))
                for slot, i in enumerate(need):
                    chunks[i] = new[slot]; idxs[i] = 0
            any_alive = False
            for i in range(len(group)):
                if not alive[i]:
                    continue
                a = chunks[i][min(idxs[i], len(chunks[i]) - 1)]
                obs, _, done, _, info = envs[i].step(
                    {k: np.atleast_1d(a[j]) for j, k in enumerate(ACTION_KEYS)})
                bufs[i].append(obs); idxs[i] += 1
                if trace is not None:
                    # the 7-d state is what the phase segmentation needs: xyz
                    # gives the end-effector path, gripper gives open/close
                    # transitions, and replanned marks where a fresh chunk began
                    rec[i]["state"].append(
                        np.concatenate([np.atleast_1d(obs[k]) for k in STATE_KEYS]).astype(np.float32))
                    rec[i]["replanned"].append(1 if idxs[i] == 1 else 0)
                if info.get("success"):
                    n_success += 1; alive[i] = False
                    if trace is not None: rec[i]["success"] = 1
                elif done:
                    alive[i] = False
                else:
                    any_alive = True
            if not any_alive:
                break
        for env in envs:
            env.close()
        if trace is not None:
            for r in rec:
                trace.append({"tid": r["tid"], "ep": r["ep"], "success": r["success"],
                              "state": np.asarray(r["state"], dtype=np.float32),
                              "replanned": np.asarray(r["replanned"], dtype=np.int8)})
    return n_success


def run_task(policy, task_bddl, task_desc, init_states, n_eps, delay, max_steps,
             horizon, ep_offset=0, action_steps=0):
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv

    env = LiberoEnv(task_bddl_file=task_bddl, task_description=task_desc)
    K = action_steps if action_steps > 0 else max(1, delay)
    n_success = 0
    for ep in range(ep_offset, ep_offset + n_eps):
        obs, _ = env.reset()
        if init_states is not None:
            env._env.set_init_state(init_states[ep % len(init_states)])
            obs, _, _, _, _ = env.step({k: np.zeros(1) for k in ACTION_KEYS})
        # A d-deep buffer: index 0 is what the policy is allowed to see now.
        buf = deque([obs] * (delay + 1), maxlen=delay + 1)
        chunk, idx, success = None, 0, False
        for t in range(max_steps):
            if chunk is None or idx >= K:
                chunk = chunk_from_action(policy.get_action(batchify(buf[0])), horizon)
                idx = 0
            a = chunk[min(idx, len(chunk) - 1)]
            obs, _, done, _, info = env.step({k: np.atleast_1d(a[i]) for i, k in enumerate(ACTION_KEYS)})
            buf.append(obs)
            idx += 1
            if info.get("success"):
                success = True
                break
            if done:
                break
        n_success += int(success)
    env.close()
    return n_success


def make_units(n_tasks, per_task, nw):
    """Split the sweep into (tid, ep_offset, n_eps) units covering each episode once.

    Splitting by task alone caps parallelism at the task count -- 10 for LIBERO
    -- which wastes cores once the CPU grant is raised. Slicing each task's
    episode range lets the worker count run up to the VRAM ceiling instead
    (~20 copies of a 3B policy on a 180 GB B200). ep_offset indexes the same
    init_states list the serial path uses, so the union of the units is exactly
    the episode set the serial run would have executed -- no episode is dropped
    or evaluated twice, which is the only way the success counts stay comparable.
    """
    chunks = max(1, min(per_task, -(-nw // n_tasks)))       # ceil division
    units = []
    for tid in range(n_tasks):
        base, extra = divmod(per_task, chunks)
        off = 0
        for c in range(chunks):
            n = base + (1 if c < extra else 0)
            if n:
                units.append((tid, off, n))
                off += n
        assert off == per_task, f"task {tid}: covered {off} of {per_task}"
    return units


def cpu_quota():
    """Cores this container may actually use.

    os.cpu_count() reports the HOST's cores -- 256 on these nodes -- while the
    cgroup grants 16 (jd_submit_lite passes --cpu-milli 16000). Everything that
    sizes a thread pool from cpu_count therefore oversubscribes by 16x, and
    numpy/torch/MuJoCo all do exactly that by default. That contention, not the
    GPU, is what made identical delay sweeps differ by 40% in wall time
    depending on how many jobs shared a node.
    """
    try:                                            # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:                               # noqa: BLE001
        pass
    try:                                            # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read())
        if quota > 0:
            return max(1, quota // period)
    except Exception:                               # noqa: BLE001
        pass
    return os.cpu_count() or 4


def _run_task_slice(payload):
    """One worker process: load the policy once, run its slice of the suite.

    Tasks are independent -- each has its own bddl and its own fixed
    init_states -- so splitting them across processes cannot change any
    success count, it only stops one MuJoCo simulation from being the whole
    job. The bottleneck here is CPU: physics plus two 256x256 EGL renders per
    step, which is why d=0 and d=4 take nearly the same wall time despite a 4x
    difference in policy calls, and why a low-success delay is SLOWER (failed
    episodes run the full step budget).

    Spawned rather than forked: by this point the parent may hold a CUDA
    context, and a forked child inheriting one aborts -- the same failure mode
    that forces SubprocVectorEnv+spawn elsewhere in this project. Every import
    that touches torch or EGL therefore happens inside this function.
    """
    (model_path, suite_name, units, delay, max_steps, horizon, threads,
     env_batch, action_steps, trace_dir, wid, seed) = payload
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)              # must precede numpy/torch
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    if seed >= 0:
        # distinct stream per (seed, delay, worker); delay enters so the four
        # delay sweeps of one job do not replay the same noise sequence
        import random as _random
        s_ = seed * 100_003 + delay * 1_009 + wid
        _random.seed(s_); np.random.seed(s_ % (2**32)); torch.manual_seed(s_)
        torch.cuda.manual_seed_all(s_)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from libero.libero import benchmark

    policy = Gr00tSimPolicyWrapper(
        Gr00tPolicy(embodiment_tag=EmbodimentTag.LIBERO_PANDA,
                    model_path=model_path, device=0)
    )
    suite = benchmark.get_benchmark_dict()[suite_name]()
    out = []
    trace = [] if trace_dir else None
    for tid, ep_off, n_eps in units:
        task = suite.get_task(tid)
        bddl = (suite.get_task_bddl_file_path(tid)
                if hasattr(suite, "get_task_bddl_file_path") else task.bddl_file)
        inits = suite.get_task_init_states(tid)
        n = run_task_batched(policy, bddl, task.language, inits, n_eps,
                             delay, max_steps, horizon, ep_offset=ep_off,
                             batch=env_batch, action_steps=action_steps,
                             trace=trace, tid=tid)
        out.append((tid, ep_off, n, n_eps))
        print(f"  [d={delay}] task {tid} eps {ep_off}-{ep_off+n_eps-1} "
              f"done ({n}/{n_eps})", flush=True)
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        # ragged episodes -> object array; each entry keeps its own length
        np.savez_compressed(
            os.path.join(trace_dir, f"d{delay}_w{wid}.npz"),
            tid=np.array([t["tid"] for t in trace], dtype=np.int32),
            ep=np.array([t["ep"] for t in trace], dtype=np.int32),
            success=np.array([t["success"] for t in trace], dtype=np.int8),
            states=np.array([t["state"] for t in trace], dtype=object),
            replanned=np.array([t["replanned"] for t in trace], dtype=object),
            allow_pickle=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--delays", default="0,1,2,3,4")
    p.add_argument("--n-episodes", type=int, default=200, help="total across the suite's tasks")
    p.add_argument("--max-steps", type=int, default=520)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--out", default="runs/groot_eval.json")
    p.add_argument("--num-workers", type=int, default=1,
                   help="processes to split the suite's tasks across; 0 = auto")
    p.add_argument("--action-steps", type=int, default=0,
                   help="fixed replan interval K; 0 keeps this project's protocol "
                        "K=max(1,delay). GR00T's own RolloutConfig uses 8, so pass "
                        "8 (with --max-steps 504) to reproduce their setup.")
    p.add_argument("--trace-dir", default="",
                   help="if set, write per-episode trajectories here (7-d state "
                        "and replan flags per step) for failure analysis")
    p.add_argument("--seed", type=int, default=-1,
                   help="seed torch/numpy/random for the policy's sampling noise "
                        "(LIBERO init states are fixed by the suite files, so this "
                        "is the only randomness); -1 = unseeded (legacy runs)")
    p.add_argument("--seeds", default="",
                   help="comma list; runs the whole delay sweep once per seed in one "
                        "job (results keyed seed<S>_d<D>). Overrides --seed.")
    p.add_argument("--env-batch", type=int, default=1,
                   help="environments stepped together inside each worker; "
                        "raises GPU batch size without another copy of the model")
    args = p.parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from libero.libero import benchmark

    policy = None
    if args.num_workers == 1:
        policy = Gr00tSimPolicyWrapper(
            Gr00tPolicy(embodiment_tag=EmbodimentTag.LIBERO_PANDA,
                        model_path=args.model_path, device=0)
        )
    suite = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = suite.n_tasks
    per_task = max(1, args.n_episodes // n_tasks)
    print(f"suite={args.suite} tasks={n_tasks} eps/task={per_task}", flush=True)

    nw = args.num_workers
    if nw == 0:
        # jd_submit_lite hands every job 16 cores and 128 GiB (--cpu-milli
        # 16000) while a serial sweep uses one: MuJoCo is single-threaded and
        # the two EGL renders per step are cheap. So the ceiling is really the
        # task count and VRAM -- each worker holds its own ~7 GB copy of the
        # 3B policy, and a B200 has 180 GB. Leave a few cores for rendering
        # and the parent.
        nw = max(1, min(n_tasks * per_task, max(1, cpu_quota() - 2), 20))
    if nw > 1:
        print(f"parallel eval: {nw} workers x {args.env_batch} envs "
              f"= {nw * args.env_batch} concurrent over {n_tasks} tasks "
              f"(cgroup grants {cpu_quota()} cores; host shows {os.cpu_count()})",
              flush=True)

    results = {}
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]
    for seed in seeds:
        args.seed = seed
        print(f"seed={args.seed}", flush=True)
        for delay in [int(x) for x in args.delays.split(",")]:
            t0, total, done_eps = time.time(), 0, 0
            if args.seed >= 0 and nw <= 1:
                import random as _random
                import numpy as _np
                import torch as _torch
                s_ = args.seed * 100_003 + delay * 1_009
                _random.seed(s_); _np.random.seed(s_ % (2**32)); _torch.manual_seed(s_)
                _torch.cuda.manual_seed_all(s_)
            if nw > 1:
                units = make_units(n_tasks, per_task, nw)
                slices = [units[i::nw] for i in range(nw)]
                slices = [sl for sl in slices if sl]
                per_worker = max(1, cpu_quota() // max(1, len(slices)))
                payloads = [(args.model_path, args.suite, sl, delay,
                             args.max_steps, args.horizon, per_worker,
                             args.env_batch, args.action_steps, args.trace_dir, wi,
                             args.seed)
                            for wi, sl in enumerate(slices)]
                ctx = mp.get_context("spawn")
                with ctx.Pool(len(payloads)) as pool:
                    for chunk in pool.imap_unordered(_run_task_slice, payloads):
                        for tid, ep_off, n, n_eps in chunk:
                            total += n
                            done_eps += n_eps
                print(f"  [d={delay}] all tasks done, SR={100*total/done_eps:.1f}%", flush=True)
            else:
                for tid in range(n_tasks):
                    task = suite.get_task(tid)
                    bddl = os.path.join(suite.get_task_bddl_file_path(tid)) if hasattr(
                        suite, "get_task_bddl_file_path") else task.bddl_file
                    inits = suite.get_task_init_states(tid)
                    total += run_task(policy, bddl, task.language, inits, per_task,
                                      delay, args.max_steps, args.horizon,
                                      action_steps=args.action_steps)
                    done_eps += per_task
                    print(f"  [d={delay}] task {tid} done, running SR={100*total/done_eps:.1f}%", flush=True)
            sr = 100.0 * total / done_eps
            results[f"seed{seed}_d{delay}" if args.seeds else delay] = sr
            print(f"OVERALL {args.suite} seed={seed} d={delay}: SR={sr:.2f}% ({done_eps} eps, {time.time()-t0:.1f}s)", flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
