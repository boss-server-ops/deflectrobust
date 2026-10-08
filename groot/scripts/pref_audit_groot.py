"""Preference-label audit on GR00T N1.7 / LIBERO (reviewer Q3).

DEFLECT's training signal rests on an assumption we have never measured: that
A+ = pi_ref(o_{t+d}) is a better action chunk than A- = pi_ref(o_t). This forks
the simulator at a real state and executes both, so the label is checked
against outcomes rather than assumed.

Why here and not only on Kinetix. The audit can only resolve the label where the
delay actually decides the outcome. On Kinetix a 1-4 step delay leaves 85% of
states with an identical result, so the headline is pinned near 0.5 whatever the
label does. On LIBERO the same delays move GR00T from 74.20 to 85.50 (d=4 alone
is +22.70), i.e. there is far more for the label to be right or wrong about.
The venue is chosen on tie rate, which is fixed before any precision is seen.

Pairing. Both branches start from the same flattened MuJoCo state, inherit the
same observation buffer, and draw the same flow-matching noise (torch seed reset
before every policy call), so the only difference between them is which
observation generated the first chunk. After the first K steps both continue
under the SAME delayed policy: the question is whether that one chunk was
better, not whether running delay-free is better.
"""
import argparse
import json
import os
from collections import deque
from pathlib import Path

import multiprocessing as mp
import numpy as np

# Import rather than restate: a hand-copied list drifted to "action.gripper_close"
# and LiberoEnv.step raised KeyError on the real key, "action.gripper".
from eval_groot_delay import ACTION_KEYS


def _sim_of(env):
    """Find the live MjSim under LiberoEnv's wrappers without hard-coding depth.

    Must be called at every use. LIBERO rebuilds the MuJoCo sim inside reset(),
    so a reference hoisted before the episode loop goes stale and raises
    "'MjSim' object has no attribute 'data'" on the next snapshot. The chain is
    LiberoEnv -> OffScreenRenderEnv (owns .sim, .get_sim_state, .set_init_state)
    -> Libero_Tabletop_Manipulation.
    """
    node = env
    for _ in range(6):
        if hasattr(node, "sim"):
            return node.sim
        for attr in ("_env", "env"):
            if hasattr(node, attr):
                node = getattr(node, attr)
                break
        else:
            break
    raise RuntimeError("no .sim found under LiberoEnv")


def _audit_slice(payload):
    (model_path, suite_name, units, delay, max_steps, horizon, threads,
     action_steps, n_seeds, seed0, null_control, sub_chunks, wid) = payload
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)              # must precede numpy/torch
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    # torch.manual_seed alone does not make the pair reproducible: the GPU
    # forward is non-deterministic (TF32 reductions, cuDNN algorithm choice) and
    # the flow integration amplifies the low bits. The null control measured the
    # cost -- 20% of IDENTICAL-chunk pairs still diverged, i.e. a fifth of the
    # "decided" pairs were decided by jitter rather than by the label. Pin what
    # can be pinned; warn_only so an unsupported op degrades instead of aborting.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:
        print(f"[determinism] not fully available: {exc}", flush=True)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from libero.libero import benchmark
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv

    import eval_groot_delay as EG

    policy = Gr00tSimPolicyWrapper(
        Gr00tPolicy(embodiment_tag=EmbodimentTag.LIBERO_PANDA,
                    model_path=model_path, device=0)
    )
    suite = benchmark.get_benchmark_dict()[suite_name]()
    K = action_steps if action_steps > 0 else max(1, delay)
    rows = []

    def act(obs_for_policy, s):
        torch.manual_seed(s)
        return EG.chunk_from_action(policy.get_action(EG.batchify(obs_for_policy)),
                                    horizon)

    for tid, ep_off, n_eps in units:
        task = suite.get_task(tid)
        bddl = (suite.get_task_bddl_file_path(tid)
                if hasattr(suite, "get_task_bddl_file_path") else task.bddl_file)
        inits = suite.get_task_init_states(tid)
        env = LiberoEnv(task_bddl_file=bddl, task_description=task.language)
        rng = np.random.RandomState(seed0 + 1009 * tid + 31 * delay)

        for ep in range(ep_off, ep_off + n_eps):
            fork_at = int(rng.randint(delay + 1, 120))
            obs, _ = env.reset()
            env._env.set_init_state(inits[ep % len(inits)])
            obs, _, _, _, _ = env.step({k: np.zeros(1) for k in ACTION_KEYS})
            buf = deque([obs] * (delay + 1), maxlen=delay + 1)

            # phase 1: reach a real deployment state under the delayed policy
            chunk, idx, dead = None, 0, False
            for t in range(fork_at):
                if chunk is None or idx >= K:
                    chunk = act(buf[0], seed0 + t)
                    idx = 0
                a = chunk[min(idx, len(chunk) - 1)]
                obs, _, done, _, info = env.step(
                    {k: np.atleast_1d(a[i]) for i, k in enumerate(ACTION_KEYS)})
                buf.append(obs); idx += 1
                if info.get("success") or done:
                    dead = True          # settled before the fork; no label to test
                    break
            if dead:
                continue

            snap = _sim_of(env).get_state().flatten().copy()
            buf_snap = list(buf)

            for m in range(n_seeds):
                s_pair = seed0 + 7919 * m + 13 * ep
                # one noise draw per branch, identical across branches
                a_plus = act(buf_snap[-1], s_pair)     # pi_ref(o_{t+d}) -- fresh
                a_minus = act(buf_snap[0], s_pair)     # pi_ref(o_t)     -- stale
                if null_control:
                    # A+ vs A+: the two branches are now identical, so any gap
                    # between sr_plus and sr_minus is asymmetry manufactured by
                    # the fork/restore path, not a property of the label. Run
                    # this before believing any signed result.
                    a_minus = a_plus
                res = {}
                for tag, chunk0 in (("plus", a_plus), ("minus", a_minus)):
                    # --sustained changes the QUESTION, not the pairing.
                    #
                    # The default asks "is this one A+ chunk better than this one
                    # A- chunk", substituting once and then letting both branches
                    # continue identically. Training does not do that: it pushes
                    # toward the fresh-conditioned action at EVERY replan. A
                    # single substitution can be a coin flip while the sustained
                    # difference is large -- the base model itself scores ~95% at
                    # d=0 and far less at d=4, which is exactly that gap.
                    # Under --sustained the plus branch keeps re-planning from
                    # the fresh observation and the minus branch from the stale
                    # one; state, seeds and everything else stay paired.
                    # How many chunks the plus branch gets to draw from the
                    # FRESH observation before being handed back to the deployed
                    # (stale) policy. 1 = only the initial A+ chunk, which is the
                    # single-substitution question; a large N = the sustained
                    # question. Sweeping N in between is the point: it shows the
                    # weak per-sample signal accumulating, and N=2/4 are not the
                    # trivial "delay-0 vs delay-d" comparison that a reviewer can
                    # dismiss the sustained arm as.
                    n_fresh = sub_chunks if tag == "plus" else 0
                    n_planned = 1          # chunk0 already counts as the first
                    # Restoring only the physics leaves the wrapper's own
                    # done flag set, and the next step raises "executing action
                    # in terminated episode" as soon as one branch finishes.
                    # Go through LIBERO's episode-start path instead: reset()
                    # clears the wrapper, set_init_state() accepts exactly this
                    # flattened sim state, and the zero-action step mirrors what
                    # the evaluator does after set_init_state.
                    env.reset()
                    env._env.set_init_state(snap)
                    env.step({k: np.zeros(1) for k in ACTION_KEYS})
                    b = deque(buf_snap, maxlen=delay + 1)
                    succ, steps = False, max_steps - fork_at
                    c, i2 = chunk0, 0
                    for t in range(max_steps - fork_at):
                        if i2 >= K:
                            # continuation is identical for both branches: the
                            # deployed (delayed) policy, same seed stream
                            c = act(b[-1] if n_planned < n_fresh else b[0],
                                    s_pair + 101 * (t + 1))
                            n_planned += 1
                            i2 = 0
                        a = c[min(i2, len(c) - 1)]
                        obs, _, done, _, info = env.step(
                            {k: np.atleast_1d(a[j]) for j, k in enumerate(ACTION_KEYS)})
                        b.append(obs); i2 += 1
                        if info.get("success"):
                            succ, steps = True, t + 1
                            break
                        if done:
                            break
                    res[tag] = (succ, steps)
                rows.append(dict(task=int(tid), ep=int(ep), delay=int(delay),
                                 seed=int(m), fork=fork_at,
                                 succ_plus=int(res["plus"][0]),
                                 succ_minus=int(res["minus"][0]),
                                 steps_plus=int(res["plus"][1]),
                                 steps_minus=int(res["minus"][1])))
            print(f"  [d={delay}] task {tid} ep {ep} fork={fork_at} "
                  f"+{rows[-1]['succ_plus']} -{rows[-1]['succ_minus']}", flush=True)
        env.close()
    return rows


def summarize(rows):
    """Primary: success difference. Secondary: speed among both-success pairs.

    A chunk substitution at one state flips the episode outcome only sometimes,
    so success alone leaves many ties. Among pairs where both branches succeed,
    the faster one is the better chunk -- that breaks ties without a threshold.
    """
    out = {}
    d = np.array([r["succ_plus"] - r["succ_minus"] for r in rows])
    if d.size == 0:
        return {"error": "no rows"}
    dec = d[d != 0]
    out["overall"] = dict(
        n=int(d.size), tie_rate=float(np.mean(d == 0)),
        precision=float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size),
        precision_decided=float(np.mean(dec > 0)) if dec.size else None,
        n_decided=int(dec.size),
        mean_delta=float(d.mean()),
        sr_plus=float(np.mean([r["succ_plus"] for r in rows])),
        sr_minus=float(np.mean([r["succ_minus"] for r in rows])))
    both = [r for r in rows if r["succ_plus"] and r["succ_minus"]]
    if both:
        sp = np.array([r["steps_plus"] - r["steps_minus"] for r in both])
        out["speed_among_both_success"] = dict(
            n=len(both), faster_plus=float(np.mean(sp < 0)),
            mean_step_diff=float(sp.mean()))
    by = {}
    for v in sorted({r["delay"] for r in rows}):
        m = np.array([r["delay"] == v for r in rows])
        dv = d[m]; dvd = dv[dv != 0]
        by[int(v)] = dict(n=int(m.sum()), tie_rate=float(np.mean(dv == 0)),
                          precision_decided=(float(np.mean(dvd > 0))
                                             if dvd.size else None),
                          n_decided=int(dvd.size), mean_delta=float(dv.mean()),
                          sr_plus=float(np.mean([r["succ_plus"] for r, k
                                                 in zip(rows, m) if k])),
                          sr_minus=float(np.mean([r["succ_minus"] for r, k
                                                  in zip(rows, m) if k])))
    out["by_delay"] = by
    bt = {}
    for v in sorted({r["task"] for r in rows}):
        m = np.array([r["task"] == v for r in rows])
        dv = d[m]; dvd = dv[dv != 0]
        bt[int(v)] = dict(n=int(m.sum()), tie_rate=float(np.mean(dv == 0)),
                          precision_decided=(float(np.mean(dvd > 0))
                                             if dvd.size else None),
                          n_decided=int(dvd.size), mean_delta=float(dv.mean()))
    out["by_task"] = bt
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="the REFERENCE policy that generated the pairs (the base)")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--delays", default="1,2,3,4")
    p.add_argument("--eps-per-task", type=int, default=20)
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=720)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--action-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=5)
    p.add_argument("--sub-chunks", type=int, default=1,
                   help="number of chunks the plus branch draws from the "
                        "fresh observation before reverting to the stale "
                        "policy; 1 = single substitution")
    p.add_argument("--sustained", action="store_true",
                   help="plus branch re-plans from fresh obs at every step, "
                        "minus from stale: the distributional question")
    p.add_argument("--null-control", action="store_true",
                   help="run A+ against A+; sr_plus must equal sr_minus")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/groot_pref.json")
    args = p.parse_args()

    import eval_groot_delay as EG
    quota = EG.cpu_quota()
    nw = max(1, args.num_workers)
    threads = max(1, quota // nw)
    print(f"[threads] cgroup grants {quota} cores; {threads} per worker", flush=True)

    from libero.libero import benchmark
    n_tasks = len(benchmark.get_benchmark_dict()[args.suite]().get_task_names())

    all_rows = []
    for delay in [int(x) for x in args.delays.split(",")]:
        # make_units returns the FLAT work list of (tid, ep_off, n) covering
        # every episode once -- it is not already partitioned per worker. Deal
        # it out round-robin the way eval_groot_delay does, or each worker
        # receives a bare tuple and unpacking an int raises.
        units = EG.make_units(n_tasks, args.eps_per_task, nw)
        slices = [sl for sl in (units[i::nw] for i in range(nw)) if sl]
        payloads = [(args.model_path, args.suite, sl, delay, args.max_steps,
                     args.horizon, threads, args.action_steps, args.n_seeds,
                     args.seed, args.null_control,
                     10**9 if args.sustained else args.sub_chunks, i)
                    for i, sl in enumerate(slices)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            for got in pool.map(_audit_slice, payloads):
                all_rows.extend(got)
        print(f"=== d={delay} done, {len(all_rows)} rows total ===", flush=True)

    summary = summarize(all_rows)
    Path(args.out).write_text(json.dumps(dict(summary=summary, rows=all_rows), indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)      # MuJoCo/EGL workers do not always die on close()
