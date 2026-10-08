#!/usr/bin/env python
"""Collect DEFLECT preference pairs for GR00T N1.7 on LIBERO.

At each replanning point of a delayed rollout we ask the frozen reference policy
twice: once on the observation it would actually see at deployment (stale by d
steps) and once on the execution-time observation it *would have* seen with no
delay. The first is the rejected branch, the second the preferred one.

The two queries must differ only in the observation, so the sampling noise is
pinned by reseeding torch before each call -- the initial noise is the single
torch.randn inside get_action_with_features, so an identical seed reproduces it.
A bitwise-identical A+/A- is the failure mode that silently wasted the first
SmolVLA run (its framework reused the noise buffer in place), so this script
measures |A+ - A-| and refuses to save a shard that looks degenerate.
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

STATE_KEYS = ["state.x", "state.y", "state.z", "state.roll", "state.pitch",
              "state.yaw", "state.gripper"]
VIDEO_KEYS = ["video.image", "video.wrist_image"]
LANG_KEY = "annotation.human.action.task_description"
ACTION_KEYS = ["action.x", "action.y", "action.z", "action.roll", "action.pitch",
               "action.yaw", "action.gripper"]


def batchify(obs):
    out = {}
    for k in VIDEO_KEYS:
        out[k] = np.asarray(obs[k], dtype=np.uint8)[None, None]
    for k in STATE_KEYS:
        out[k] = np.asarray(obs[k], dtype=np.float32)[None, None]
    out[LANG_KEY] = [obs[LANG_KEY]]
    return out


def to_chunk(action):
    if isinstance(action, tuple):
        action = action[0]
    cols = []
    for k in ACTION_KEYS:
        v = np.asarray(action[k])
        v = v[0] if v.ndim == 3 else v
        cols.append(v.reshape(v.shape[0], -1) if v.ndim > 1 else v.reshape(-1, 1))
    return np.concatenate(cols, axis=-1).astype(np.float32)


def _sim_of(env):
    """Locate the live MjSim under LiberoEnv's wrappers, at every use.

    LIBERO rebuilds the MuJoCo sim inside reset(), so a reference hoisted
    before the loop goes stale and raises "'MjSim' object has no attribute
    'data'" on the next snapshot.
    """
    e = env
    for _ in range(6):
        if hasattr(e, "sim") and hasattr(e.sim, "get_state"):
            return e.sim
        e = getattr(e, "_env", None) or getattr(e, "env", None)
        if e is None:
            break
    raise RuntimeError("could not locate MjSim under the env wrappers")


def flat_to_nested(policy, flat):
    """Flat sim-format observation -> the nested layout Gr00tPolicy expects.

    Mirrors Gr00tSimPolicyWrapper._get_action; we cannot call the wrapper itself
    because it decodes actions to physical units, and the preference pairs must
    live in the model's normalized space.
    """
    out = {}
    for modality in ["video", "state", "language"]:
        out[modality] = {}
        for key in policy.modality_configs[modality].modality_keys:
            parsed = key if modality == "language" else f"{modality}.{key}"
            arr = flat[parsed]
            out[modality][key] = ([[str(x)] for x in arr] if modality == "language" else arr)
    return out

def query(policy, obs, seed):
    """One reference query with the sampling noise pinned to `seed`.

    Returns (normalized_chunk, physical_chunk). The DEFLECT loss lives in the
    model's normalized action space -- that is where flow matching is defined --
    while the environment needs physical units, so we take both from the single
    forward rather than storing decoded actions and hunting for an encoder
    (the processor exposes decode_action but no inverse).
    """
    from gr00t.data.types import MessageType

    inner = policy.policy                      # Gr00tSimPolicyWrapper -> Gr00tPolicy
    unbatched = inner._unbatch_observation(flat_to_nested(inner, batchify(obs)))
    processed, states = [], []
    for o in unbatched:
        step = inner._to_vla_step_data(o)
        states.append(step.states)
        processed.append(inner.processor([{"type": MessageType.EPISODE_STEP.value,
                                           "content": step}]))
    collated = inner.collate_fn(processed)
    from gr00t.policy.gr00t_policy import _rec_to_dtype
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)

    torch.manual_seed(seed)
    with torch.inference_mode():
        pred = inner.model.get_action(**collated)
    normalized = pred["action_pred"].float()

    batched_states = {k: np.stack([s[k] for s in states], axis=0)
                      for k in inner.modality_configs["state"].modality_keys}
    physical = inner.processor.decode_action(
        normalized.cpu().numpy(), inner.embodiment_tag, batched_states)
    # decode_action returns bare modality keys; the sim format the env consumes
    # prefixes them with "action." (the wrapper does this on the way out).
    phys_chunk = to_chunk({f"action.{k}": v.astype(np.float32) for k, v in physical.items()})
    return normalized[0].cpu().numpy().astype(np.float32), phys_chunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--delay", type=int, required=True)
    p.add_argument("--max-pairs", type=int, default=3000)
    p.add_argument("--verify-steps", type=int, default=0,
                   help="if >0, fork the simulator at the pair's state and roll "
                       "BOTH chunks forward this many steps, recording which "
                       "actually earns more return. The audit measures the "
                       "as-constructed label at ~0.53 precision on a single "
                       "chunk -- i.e. half the training signal is noise -- and "
                       "the suites where preference loses to distillation "
                       "(goal 0.702, object 0.633) are exactly the ones with the "
                       "weakest labels. Verified pairs carry 'verdict': +1 if "
                       "A+ wins, -1 if A- wins, 0 if tied.")
    p.add_argument("--keep-top", type=int, default=0,
                   help="over-collect, then keep only the KEEP_TOP pairs with the "
                        "largest ||A+ - A-|| (evaluated per shard, i.e. within one "
                        "delay, so the delay distribution is untouched). Set "
                        "--max-pairs to a multiple of this. S5.61 filtered on the "
                        "same quantity but WITHOUT backfilling the count, so the "
                        "sharper contrast was cancelled by losing 70%% of the data; "
                        "over-collecting keeps the volume fixed and isolates the "
                        "margin effect.")
    p.add_argument("--average-k", type=int, default=1,
                   help="average k reference draws per branch (shared seed set) so "
                        "the stored A+/A- are lower-variance estimates of what the "
                        "policy does at the fresh / stale observation. Raises the "
                        "temporal-counterfactual SNR by ~sqrt(k) without discarding "
                        "any pair. The rollout still EXECUTES the single-sample "
                        "chunk, so the state distribution is unchanged.")
    p.add_argument("--live-dims", type=int, default=7,
                   help="number of leading action dims LIBERO actually writes "
                        "(GR00T pads 132; the training run logs 'live action dims: "
                        "7/132 indices=[0..6]'). Every metric over the action space "
                        "must be scored on these only.")
    p.add_argument("--confidence-k", type=int, default=0,
                   help="extra samples of A+ drawn from the SAME fresh observation "
                        "with different noise seeds, used to measure how determinate "
                        "the reference policy is at that state. Large ||A+ - A-|| says "
                        "the two branches differ; it does NOT say A+ is the better one. "
                        "If the policy is multi-modal at this state, A+ is just one mode "
                        "sampled at random and the preference label is arbitrary -- which "
                        "is what low audited precision on goal (0.702) / object (0.633) "
                        "looks like. Spread across seeds measures exactly that.")
    p.add_argument("--keep-mode", choices=["margin", "random", "consistency"], default="margin",
                   help="how --keep-top selects. 'margin' keeps the largest "
                        "||A+ - A-||; 'random' keeps a uniform sample. Over-collecting "
                        "widens the pool of EPISODES a task draws from (quota 300/task "
                        "reaches ~15 of 50 init states, quota 900/task reaches ~45), so "
                        "margin-selection and episode-breadth move together unless the "
                        "random arm is run as the control.")
    p.add_argument("--per-task", action="store_true",
                   help="spend the max-pairs budget EVENLY over the suite's tasks "
                        "instead of filling it from the first tasks. Without this, "
                        "3000 pairs are exhausted on task 0-1 of 10 (long: task 0 "
                        "alone), so the preference signal covers 10-20%% of the "
                        "tasks the policy is later evaluated on -- which is why "
                        "only spatial tolerated large lambda / long training, and "
                        "why long collapsed first: it had the narrowest coverage.")
    p.add_argument("--max-steps", type=int, default=520)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-path", required=True)
    p.add_argument("--action-steps", type=int, default=0,
                   help="replan interval K used while rolling out. 0 keeps the old "
                        "K=max(1,delay). It must match the K the policy is evaluated "
                        "under: K sets which states the rollout visits and how long "
                        "one decision has to stay good, so pairs collected at K=1 "
                        "describe a different problem than deployment at K=8.")
    args = p.parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from libero.libero import benchmark

    policy = Gr00tSimPolicyWrapper(
        Gr00tPolicy(embodiment_tag=EmbodimentTag.LIBERO_PANDA,
                    model_path=args.model_path, device=0)
    )
    suite = benchmark.get_benchmark_dict()[args.suite]()
    d = args.delay
    K = args.action_steps if args.action_steps > 0 else max(1, d)
    print(f"collecting at delay={d}, replan interval K={K}", flush=True)
    pairs, diffs = [], []
    live_dims = None          # inferred from the first pair; see --confidence-k
    rng = np.random.RandomState(args.seed)
    t0 = time.time()

    quota = (args.max_pairs // max(1, suite.n_tasks)) if args.per_task else args.max_pairs
    for tid in range(suite.n_tasks):
        if len(pairs) >= args.max_pairs:
            break
        cap = min(args.max_pairs, len(pairs) + quota) if args.per_task else args.max_pairs
        task = suite.get_task(tid)
        bddl = suite.get_task_bddl_file_path(tid) if hasattr(
            suite, "get_task_bddl_file_path") else task.bddl_file
        inits = suite.get_task_init_states(tid)
        env = LiberoEnv(task_bddl_file=bddl, task_description=task.language)

        ep = 0
        while len(pairs) < cap and ep < len(inits):
            obs, _ = env.reset()
            env._env.set_init_state(inits[ep])
            obs, _, _, _, _ = env.step({k: np.zeros(1) for k in ACTION_KEYS})
            ep += 1
            buf = deque([obs] * (d + 1), maxlen=d + 1)
            chunk, idx = None, 0
            for t in range(args.max_steps):
                if chunk is None or idx >= K:
                    seed = int(rng.randint(0, 2**31 - 1))
                    stale = buf[0]                 # what deployment actually sees
                    fresh = buf[-1]                # execution-time observation
                    rej_norm, rej_phys = query(policy, stale, seed)
                    if d > 0:
                        pref_norm, pref_phys = query(policy, fresh, seed)
                    else:
                        pref_norm, pref_phys = rej_norm, rej_phys
                    if args.average_k > 1 and d > 0:
                        # Lower the DENOMINATOR of the temporal-counterfactual SNR
                        # instead of filtering on its numerator. A+ is one sample
                        # from pi_ref(o_{t+d}); where that policy is multi-modal the
                        # sample -- and therefore the label "A+ beats A-" -- is
                        # arbitrary, which is what goal's 0.702 audited precision
                        # looks like. Averaging k draws leaves the observation
                        # effect untouched while shrinking the sampling noise by
                        # sqrt(k), so SNR rises by sqrt(k) and NO pair is discarded
                        # (unlike --keep-top, which lost 2/3 of the data and made
                        # things worse). Both branches share the same seed set, so
                        # what common-mode noise survives cancels in the difference.
                        # The mean is also the MSE-optimal target, which is exactly
                        # how A+ enters the flow-matching DPO term.
                        seeds_k = [int(rng.randint(0, 2**31 - 1))
                                   for _ in range(args.average_k - 1)]
                        pa = [pref_norm] + [query(policy, fresh, sk)[0] for sk in seeds_k]
                        ra = [rej_norm] + [query(policy, stale, sk)[0] for sk in seeds_k]
                        pref_norm = np.stack(pa).mean(0)
                        rej_norm = np.stack(ra).mean(0)
                    chunk, idx = rej_phys, 0       # deployment executes the stale plan
                    verdict = None
                    if d > 0 and args.verify_steps > 0:
                        # Fork at the CURRENT state and run each chunk from it.
                        # Both branches share the snapshot and the step budget, so
                        # the only difference is which chunk is executed.
                        snap = _sim_of(env).get_state().flatten().copy()
                        rets = []
                        # BUG FIX (2026-09-14): this used to do
                        #   cphys = policy.unnormalize(cand) if hasattr(policy, "unnormalize") else cand
                        # No object in gr00t or in this bundle defines `unnormalize`, so the
                        # fallback ALWAYS ran and the env was stepped with NORMALIZED actions
                        # (query() decodes to physical units via processor.decode_action).
                        # Both branches executed garbage, neither could succeed, and the
                        # audit came out 99.8% ties (report S5.67) -- an artefact, not a
                        # property of the labels. The physical chunks were already computed
                        # by the two query() calls above; use them.
                        for cphys in (pref_phys, rej_phys):
                            # reset() first: robosuite refuses to step a
                            # terminated episode, and a branch that reached
                            # success/done leaves the env in exactly that state.
                            # set_init_state alone does not clear the flag.
                            env.reset()
                            env._env.set_init_state(snap)
                            env.step({k: np.zeros(1) for k in ACTION_KEYS})
                            r = 0.0
                            for j in range(args.verify_steps):
                                aj = cphys[min(j, len(cphys) - 1)]
                                _, rr, dn, _, inf = env.step(
                                    {k: np.atleast_1d(aj[i]) for i, k in enumerate(ACTION_KEYS)})
                                r += float(rr) + (1.0 if inf.get("success") else 0.0)
                                if dn or inf.get("success"):
                                    break
                            rets.append(r)
                        verdict = 1 if rets[0] > rets[1] else (-1 if rets[0] < rets[1] else 0)
                        env.reset()                        # restore for the real rollout
                        env._env.set_init_state(snap)
                        env.step({k: np.zeros(1) for k in ACTION_KEYS})
                    conf = None
                    if d > 0 and args.confidence_k > 0:
                        # Live-dim mask. GR00T pads 125 of 132 action dims; the
                        # head's output there is essentially the sampling noise it
                        # was handed. A+ and A- share a seed, so the padding cancels
                        # in their difference -- but the confidence samples use
                        # DIFFERENT seeds, so unmasked spread is dominated by
                        # sqrt(125)~11.2 of pure padding noise and comes out the
                        # same (10.084) on every suite. Both terms must be scored
                        # on the dims LIBERO actually writes.
                        if live_dims is None:
                            # NOT inferred from where A+ and A- differ: the two use
                            # the same seed but DIFFERENT observations, so the head's
                            # padding outputs shift a little too and every one of the
                            # 132 dims tests non-zero. The training run resolves them
                            # from the processor's action_mask and logs
                            # "live action dims: 7/132 indices=[0..6]".
                            live_dims = np.zeros(pref_norm.shape[-1], dtype=bool)
                            live_dims[:args.live_dims] = True
                        # Spread of INDEPENDENT single draws only. With
                        # --average-k, pref_norm is already a mean of k draws and
                        # mixing it in would understate the policy's own spread.
                        alt = []
                        for _ in range(args.confidence_k):
                            alt.append(query(policy, fresh,
                                             int(rng.randint(0, 2**31 - 1)))[0])
                        arr = np.stack(alt)[..., live_dims]
                        conf = float(np.linalg.norm(
                            arr - arr.mean(0, keepdims=True), axis=-1).mean())
                    if d > 0:
                        diffs.append(float(np.abs(pref_norm - rej_norm).mean()))
                        pairs.append({
                            "obs_stale": {
                                "image": np.asarray(stale["video.image"], dtype=np.uint8),
                                "wrist_image": np.asarray(stale["video.wrist_image"], dtype=np.uint8),
                                "state": np.concatenate([np.atleast_1d(stale[k]) for k in STATE_KEYS]).astype(np.float32),
                            },
                            "task": stale[LANG_KEY],
                            "action_preferred": torch.from_numpy(pref_norm),
                            "action_rejected": torch.from_numpy(rej_norm),
                            **({"verdict": verdict} if verdict is not None else {}),
                            **({"conf_spread": conf} if conf is not None else {}),
                        })
                        if len(pairs) >= cap:
                            break
                a = chunk[min(idx, len(chunk) - 1)]
                obs, _, done, _, info = env.step(
                    {k: np.atleast_1d(a[i]) for i, k in enumerate(ACTION_KEYS)})
                buf.append(obs)
                idx += 1
                if info.get("success") or done:
                    break
        env.close()
        print(f"[{args.suite} d={d}] task {tid} done, pairs so far: {len(pairs)}", flush=True)

    if args.keep_top > 0 and len(pairs) > args.keep_top:
        mg = [float(torch.linalg.vector_norm(
            q["action_preferred"].float() - q["action_rejected"].float())) for q in pairs]
        # Select WITHIN each task, not globally. Tasks differ systematically in
        # ||A+ - A-||, so a global top-K collapses onto the few widest-contrast
        # tasks and re-creates the coverage defect that --per-task just fixed.
        if args.keep_mode == "margin":
            pick = lambda idx, k: sorted(idx, key=lambda i: -mg[i])[:k]
        elif args.keep_mode == "consistency":
            cf = [pairs[i].get("conf_spread", 0.0) for i in range(len(pairs))]
            pick = lambda idx, k: sorted(idx, key=lambda i: cf[i])[:k]   # LOW spread first
        else:
            pick = lambda idx, k: list(rng.permutation(idx)[:k])
        if args.per_task:
            groups = {}
            for i, q in enumerate(pairs):
                groups.setdefault(q["task"], []).append(i)
            per = max(1, args.keep_top // max(1, len(groups)))
            order = []
            for idx in groups.values():
                order += list(pick(idx, per))
        else:
            order = list(pick(list(range(len(pairs))), args.keep_top))
        kept_m = sum(mg[i] for i in order) / len(order)
        all_m = sum(mg) / len(mg)
        ntask = len({pairs[i]["task"] for i in order})
        if args.keep_mode == "consistency":
            cf = [pairs[i].get("conf_spread", 0.0) for i in range(len(pairs))]
            print(f"conf_spread: all {sum(cf)/len(cf):.4f} -> kept "
                  f"{sum(cf[i] for i in order)/len(order):.4f}", flush=True)
        pairs = [pairs[i] for i in sorted(order)]
        print(f"keep-top[{args.keep_mode}]: {len(pairs)}/{len(mg)} kept over {ntask} tasks; "
              f"mean ||A+-A-|| {all_m:.4f} -> {kept_m:.4f} "
              f"({kept_m/max(1e-9,all_m):.2f}x)", flush=True)
    if args.verify_steps > 0:
        vs = [q["verdict"] for q in pairs if "verdict" in q]
        if vs:
            w = sum(1 for v in vs if v == 1); l = sum(1 for v in vs if v == -1)
            t_ = len(vs) - w - l
            dec = w + l
            print(f"verdict: A+ wins {w}, A- wins {l}, ties {t_} "
                  f"(decided precision = {w/max(1,dec):.3f} over {dec})", flush=True)
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    print(f"mean |A+ - A-| = {mean_diff:.6f} over {len(diffs)} pairs", flush=True)
    # Temporal-counterfactual SNR. The numerator is how much the fresh observation
    # CHANGES the reference policy's action; the denominator is how much that action
    # moves on its own when only the sampling noise changes. A label "A+ beats A-"
    # carries information only when the first exceeds the second -- otherwise A+ is
    # just one arbitrary mode and the preference is a coin flip. This is the quantity
    # that should separate the suites where preference beats distillation
    # (spatial 0.796 / long 0.813 audited precision) from those where it does not
    # (goal 0.702 / object 0.633), and it is measurable BEFORE any training.
    cf_all = [q["conf_spread"] for q in pairs if "conf_spread" in q]
    if cf_all:
        num = float(np.mean([
            float(np.linalg.norm(
                (q["action_preferred"].float()
                 - q["action_rejected"].float()).numpy()[..., live_dims],
                axis=-1).mean()) for q in pairs]))
        print(f"live action dims: {int(live_dims.sum())}/{live_dims.size}", flush=True)
        den = float(np.mean(cf_all))
        print(f"SNR: signal {num:.6f} / noise {den:.6f} = {num/max(den,1e-9):.4f} "
              f"over {len(cf_all)} pairs", flush=True)
    if mean_diff < 1e-6:
        raise SystemExit(
            "A+ and A- are effectively identical -- the two queries are not seeing "
            "different observations, or the sampling noise is being reused in place. "
            "Refusing to save a degenerate shard.")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pairs": pairs,
                "meta": {"delay": d, "suite": args.suite, "mean_abs_diff": mean_diff,
                         "model_path": args.model_path}},
               args.output_path)
    print(f"saved {len(pairs)} pairs -> {args.output_path} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
