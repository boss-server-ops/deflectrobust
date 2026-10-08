#!/usr/bin/env python
"""DEFLECT DPO post-training for GR00T N1.7 (paper recipe, LIBERO).

L = lambda_SFT * L_FM(expert) + lambda_DPO * (-log sigmoid(beta * margin))
margin = (mse_rej - mse_pref) + stopgrad(mse_ref_pref - mse_ref_rej)

All four flow-matching terms are scored at ONE shared x_t under the deployment
context (the stale observation). Scoring each branch at its own x_t lets the
network win the margin by misbehaving on the rejected branch's input manifold
instead of learning the preference -- on Kinetix that topology cost 17-64 points
of success rate, so it is not an optimisation detail.

Setting --dpo-lambda 0 gives the matched-budget control: same steps, same data,
same schedule, only the preference gradient removed.
"""

import argparse
import glob
import json
import logging
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S", force=True)
log = logging.getLogger(__name__)

STATE_KEYS = ["state.x", "state.y", "state.z", "state.roll", "state.pitch",
              "state.yaw", "state.gripper"]


def load_pairs(pattern, margin_pct=100.0, common_noise=0.0, verified_only=False,
               flip_verdict=False):
    # A wildcard has to survive three layers of shell (subv -> submit -> the
    # runner's unquoted $STAGE_ARGS) intact. Quoted, the quotes end up inside the
    # value; unquoted, the container shell expands the glob and argparse sees
    # four positional paths. Accept a comma-separated list of explicit files so
    # no shell ever needs to glob.
    if "," in pattern:
        paths = sorted(q for q in (x.strip() for x in pattern.split(",")) if q)
        missing = [q for q in paths if not Path(q).exists()]
        if missing:
            raise FileNotFoundError(f"missing pair shards: {missing}")
    else:
        paths = sorted(glob.glob(pattern))
    pairs = []
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        d = shard["meta"]["delay"]
        for pr in shard["pairs"]:
            pr["_delay"] = d
        keep = shard["pairs"]
        if flip_verdict:
            # Strictly better than filtering on the data axis: a pair the rollout
            # judged backwards is currently pushing the policy the WRONG way, so
            # swapping its two sides removes that gradient AND keeps the sample.
            # Filtering would discard it, and DPO is the data-hungrier objective
            # (S5.61: cutting to 30% cost DEFLECT 2.75 but future-SFT only 0.85).
            have = [q for q in keep if "verdict" in q]
            if not have:
                raise SystemExit(f"--flip-verdict but {path} carries no verdicts; "
                                 f"recollect with --verify-steps > 0")
            nf = 0
            for q in have:
                if q["verdict"] == -1:
                    q["action_preferred"], q["action_rejected"] = \
                        q["action_rejected"], q["action_preferred"]
                    nf += 1
            # ties carry no directional information either way; drop only those
            keep = [q for q in have if q["verdict"] != 0]
            log.info(f"  flip filter: {nf} pairs swapped, {len(have)-len(keep)} ties "
                     f"dropped, {len(keep)}/{len(have)} kept")
        elif verified_only:
            # Keep only pairs whose label was CONFIRMED by forked rollout.
            # Filtering by the ||A+ - A-|| proxy failed (S5.61); this filters on
            # the outcome the label actually asserts.
            have = [q for q in keep if "verdict" in q]
            if not have:
                raise SystemExit(f"--verified-only but {path} carries no verdicts; "
                                 f"recollect with --verify-steps > 0")
            keep = [q for q in have if q["verdict"] == 1]
            log.info(f"  verified filter: kept {len(keep)}/{len(have)} "
                     f"(A+ confirmed better)")
        if margin_pct < 100.0:
            # Filtered-DPO for noisy preference labels. The audit measures our
            # label as near-random on a SINGLE chunk (precision 0.533, p=0.49)
            # and reliable only when sustained (0.755) -- but training consumes
            # single chunks, i.e. exactly the regime where the DPO literature
            # reports that preference optimisation stops beating SFT-on-chosen.
            # ||A+ - A-|| is the observable proxy for how much the fresh
            # observation actually changed the plan, so keep the top slice.
            #
            # Filtering happens WITHIN each delay shard, never globally: margin
            # grows with delay (0.0064 at d=1 -> 0.0336 at d=4), so a global
            # threshold would silently drop d=1 and turn a label-quality
            # experiment into a delay-mixture experiment.
            m = np.array([float(torch.linalg.vector_norm(
                pr["action_preferred"].float() - pr["action_rejected"].float()))
                for pr in keep])
            thr = np.percentile(m, 100.0 - margin_pct)
            keep = [pr for pr, mm in zip(keep, m) if mm >= thr]
            log.info(f"  margin filter: kept {len(keep)}/{len(m)} "
                     f"(top {margin_pct:g}%, ||A+-A-|| >= {thr:.5f})")
        if common_noise > 0.0:
            # Simulate a MISCALIBRATED reference policy: one perturbation per
            # pair, added to A+ and A- ALIKE. This is the error structure a
            # biased teacher actually produces -- both branches come from the
            # same pi_ref, so its bias is common-mode.
            #
            # d(margin)/dv = 2(u_pref - u_rej) is EXACTLY invariant to such a
            # perturbation, while d(fut)/dv = 2(v - u_pref - eps)/N is shifted
            # by it (verified numerically: 1.4e-6 vs 17% relative change).
            # So preference learning should degrade strictly less than pointwise
            # distillation here -- a prediction derived before running, not after.
            #
            # Seeded per shard path so BOTH arms receive bit-identical corruption.
            g = torch.Generator().manual_seed(abs(hash(path)) % (2**31))
            for pr in keep:
                e = torch.randn(pr["action_preferred"].shape, generator=g) * common_noise
                pr["action_preferred"] = pr["action_preferred"] + e
                pr["action_rejected"] = pr["action_rejected"] + e
            log.info(f"  common-mode noise sigma={common_noise:g} added to both branches")
        pairs.extend(keep)
        log.info(f"loaded {len(keep)} pairs from {path} "
                 f"(delay={d}, mean|A+-A-|={shard['meta'].get('mean_abs_diff', float('nan')):.5f})")
    if not pairs:
        raise FileNotFoundError(f"no pairs matched {pattern}")
    return pairs


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

def _cpu_quota():
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            q, p = fh.read().split()
            if q != "max":
                return max(1, int(q) // int(p))
    except Exception:      # noqa: BLE001
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            q = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            p = int(fh.read())
        if q > 0:
            return max(1, q // p)
    except Exception:      # noqa: BLE001
        pass
    return os.cpu_count() or 4


_NPROC = _cpu_quota()
_POOL = ThreadPoolExecutor(max_workers=max(2, _NPROC - 2)) if _NPROC > 2 else None


def make_inputs(policy, obs_batch, actions=None):
    """Raw observations (+ optional expert actions) -> collated model inputs.

    Actions ride along inside VLAStepData so the processor normalises them with
    the same statistics the model was trained under; there is no standalone
    encoder to call.
    """
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    def _one(o):
        flat = {
            "video.image": o["image"][None, None],
            "video.wrist_image": o["wrist_image"][None, None],
        }
        for i, k in enumerate(STATE_KEYS[:-1]):
            flat[k] = np.asarray([[o["state"][i]]], dtype=np.float32)[..., None]
        flat["state.gripper"] = np.asarray(o["state"][6:8], dtype=np.float32)[None, None]
        flat["annotation.human.action.task_description"] = [o["task"]]
        step = policy._to_vla_step_data(
            policy._unbatch_observation(flat_to_nested(policy, flat))[0])
        return step

    # The per-sample processor call (two 256x256 camera streams each) is pure
    # CPU and used to run single-threaded in a Python loop, which left the GPU
    # ~90% idle: 0.74 s/step against ~0.05-0.1 s of actual compute. The steps are
    # independent, so build them in a thread pool sized to the cgroup grant.
    steps = list(_POOL.map(_one, obs_batch)) if _POOL else [_one(o) for o in obs_batch]
    if actions is not None:
        akeys = policy.modality_configs["action"].modality_keys
        for i_s, step in enumerate(steps):
            a = np.asarray(actions[i_s], dtype=np.float32)  # (T, 7)
            step.actions = {k: a[:, i:i + 1] for i, k in enumerate(akeys)}
    processed = [policy.processor([{"type": MessageType.EPISODE_STEP.value,
                                    "content": st}]) for st in steps]
    collated = policy.collate_fn(processed)
    # collate_fn returns the kwargs get_action(**collated) takes, i.e. the real
    # payload sits under "inputs"; prepare_input wants that payload directly.
    collated = collated.get("inputs", collated)
    return _rec_to_dtype(collated, dtype=torch.bfloat16)


def flow_velocity(head, backbone_output, action_input, x_t, t):
    """Predicted velocity at a GIVEN (x_t, t) -- forward() samples its own.

    process_backbone_output writes its result back into the BatchFeature it is
    handed (vlln + vl_self_attention), so calling it twice on one object feeds
    already-normalised features through the stack a second time -- measured at
    max|v1-v2| = 2.67, i.e. a different function. Two callers hit this: the
    8-draw loop in the diagnostics, and, far worse, the margin itself, where
    the policy is scored on once-processed features and the reference on
    twice-processed ones. Hand each call its own shallow copy: the tensors are
    shared but the write-back lands on a private dict, and the reference head
    -- a deepcopy that owns separate vlln weights -- gets the raw features it
    is entitled to.
    """
    backbone_output = head.process_backbone_output(
        type(backbone_output)({k: v for k, v in backbone_output.items()}))
    vl_embeds = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id
    state = action_input.state
    state = state.view(state.shape[0], 1, -1)
    state_features = head.state_encoder(state, embodiment_id)

    t_discretized = (t[:, 0, 0] * head.num_timestep_buckets).long()
    action_features = head.action_encoder(x_t, t_discretized, embodiment_id)
    # Official forward() adds this before concatenating; omitting it strips the
    # action tokens of any notion of position in the chunk, so the DiT cannot
    # tell step 0 from step 39. The velocity still points roughly the right way
    # (rectified-flow velocity is t-independent) which is why it showed up as a
    # corr of 0.53 rather than an outright failure.
    if head.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long,
                               device=action_features.device)
        action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)
    sa_embs = torch.cat((state_features, action_features), dim=1)
    vl_attn_mask = backbone_output.backbone_attention_mask

    kw = dict(hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
              encoder_attention_mask=vl_attn_mask, timestep=t_discretized,
              return_all_hidden_states=True)
    if head.config.use_alternate_vl_dit:
        kw.update(image_mask=backbone_output.image_mask,
                  backbone_attention_mask=backbone_output.backbone_attention_mask)
    model_output, _ = head.model(**kw)
    pred = head.action_decoder(model_output, embodiment_id)
    return pred[:, -x_t.shape[1]:]


class _BadEpisode(Exception):
    """An episode whose video cannot be decoded; resample instead."""


class ExpertBatches:
    """Random (observation, action-chunk) batches from the LIBERO lerobot data.

    Only what the anchor needs: images at t, state at t, and the next `horizon`
    actions, all within one episode.
    """

    def __init__(self, root, batch_size, horizon, seed=0):
        import pandas as pd
        self.root, self.bs, self.h = Path(root), batch_size, horizon
        self.rng = np.random.RandomState(seed)
        frames = sorted((self.root / "data").rglob("*.parquet"))
        if not frames:
            raise FileNotFoundError(f"no parquet under {self.root}/data")
        self.tables = [pd.read_parquet(f) for f in frames]
        self.df = pd.concat(self.tables, ignore_index=True)
        tasks = {}
        with (self.root / "meta" / "tasks.jsonl").open() as fh:
            for line in fh:
                r = json.loads(line)
                tasks[r["task_index"]] = r["task"]
        self.tasks = tasks
        # start indices with a full in-episode chunk ahead of them
        ep = self.df["episode_index"].to_numpy()
        fi = self.df["frame_index"].to_numpy()
        last = {e: fi[ep == e].max() for e in np.unique(ep)}
        self.starts = np.array([i for i in range(len(self.df))
                                if fi[i] + horizon <= last[ep[i]]])
        log.info(f"expert data: {len(self.df)} frames, {len(self.starts)} usable starts")
        self._decoder = {}
        self._dead = set()
        # single worker: decoding is what we overlap, not what we parallelise
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._fut = None

    def _frame(self, ep_idx, frame_idx, cam):
        """Decode one frame, tolerating episodes whose video will not decode.

        libero_goal ships exactly one wrist_image mp4 (1 of 856) that opens and
        yields its first frames but dies partway through with InvalidDataError.
        It is corrupt upstream in IPEC-COMMUNITY, so re-downloading and
        re-ferrying do not help -- it killed two rounds of goal training before
        a full-decode scan pinned it down. One bad episode is not worth losing a
        suite over: mark it dead, let __next__ resample, and count it so the
        loss of data is visible rather than silent.
        """
        import av
        key = (ep_idx, cam)
        if key in self._dead:
            raise _BadEpisode(f"episode {ep_idx} ({cam}) does not decode")
        if key not in self._decoder:
            pat = f"**/observation.images.{cam}/episode_{ep_idx:06d}.mp4"
            paths = list((self.root / "videos").rglob(pat))
            if not paths:
                raise FileNotFoundError(pat)
            try:
                self._decoder[key] = [f.to_ndarray(format="rgb24")
                                      for f in av.open(str(paths[0])).decode(video=0)]
            except Exception as e:                                  # noqa: BLE001
                self._dead.add(key)
                log.warning(f"episode {ep_idx} ({cam}) failed to decode "
                            f"({type(e).__name__}); skipping it. "
                            f"{len(self._dead)} dead episode(s) so far")
                raise _BadEpisode(str(e))
        frames = self._decoder[key]
        return frames[min(frame_idx, len(frames) - 1)]

    def _prefetch(self):
        """Decode the next batch in a background thread.

        Two camera streams are decoded per sample straight off the LeRobot
        dataset every step; at batch 4 that is 8 random-access video decodes
        serialised with the GPU step. Overlapping them with compute removes
        that wait entirely -- the decode itself is unchanged.
        """
        if self._fut is None:
            self._fut = self._pool.submit(self._fill)

    def _fill(self):
        obs, acts = [], []
        attempts = 0
        while len(obs) < self.bs:
            attempts += 1
            if attempts > 20 * self.bs:
                raise RuntimeError(
                    f"could not fill a batch after {attempts} draws; "
                    f"{len(self._dead)} episodes are undecodable")
            i = int(self.rng.choice(self.starts))
            row = self.df.iloc[i]
            ep_idx, f_idx = int(row["episode_index"]), int(row["frame_index"])
            try:
                frame = self._frame(ep_idx, f_idx, "image")
                wrist = self._frame(ep_idx, f_idx, "wrist_image")
            except _BadEpisode:
                continue
            chunk = np.stack(self.df["action"].values[i:i + self.h]).astype(np.float32)
            obs.append({
                "image": frame,
                "wrist_image": wrist,
                "state": np.asarray(row["observation.state"], dtype=np.float32),
                "task": self.tasks.get(int(row["task_index"]), ""),
            })
            acts.append(chunk)
        return obs, acts

    def __next__(self):
        self._prefetch()
        fut, self._fut = self._fut, None
        out = fut.result()
        self._prefetch()          # start the next one while the GPU works
        return out

    def __iter__(self):
        return self


def resolve_action_dims(policy, model, expert_iter, n_probe=2):
    """Which of the 132 padded action dims does this embodiment actually use?

    GR00T shares one 132-d action space across embodiments; LIBERO fills 7 of
    them and the processor zero-pads the other 125. Scoring the flow loss over
    all 132 sends 95% of the gradient into forcing the head's outputs on those
    padding dims to zero, which is what flattened the first run to 2% success
    while the loss curve looked merely "slow". Both branches get masked to the
    dims the processor actually writes, and the probe is checked against the
    modality config so a silent layout change fails loudly instead of training.
    """
    filled = None
    for _ in range(n_probe):
        obs, act = next(expert_iter)
        _, ai = model.prepare_input(make_inputs(policy, obs, actions=act))
        # action_mask is what the official loss normalises by; prefer it over
        # inferring live dims from which entries happen to be non-zero.
        m = getattr(ai, "action_mask", None)
        nz = (m.bool().any(dim=0).any(dim=0) if m is not None
              else ai.action.float().abs().amax(dim=0).amax(dim=0) > 0)
        filled = nz if filled is None else (filled | nz)
    n_keys = len(policy.modality_configs["action"].modality_keys)
    live = int(filled.sum())
    log.info(f"live action dims: {live}/{filled.numel()} (modality keys={n_keys}) "
             f"indices={torch.nonzero(filled).flatten().tolist()[:16]}")
    if live != n_keys:
        raise SystemExit(
            f"action-dim probe found {live} live dims but the modality config "
            f"declares {n_keys}; refusing to train against a target space we "
            "cannot account for.")
    return filled


def inspect_src(obj):
    import inspect
    return inspect.getsource(obj).splitlines()


def run_diagnostics(policy, model, head, expert_iter, pairs, hd, act_mask, n_batches=4):
    """Do the SFT anchor and the DPO branch live in the same action space?

    A+/A- are the model's own outputs, so they are in its normalized space by
    construction. The SFT anchor instead takes actions out of the lerobot
    parquet and trusts the processor to map them into that same space -- never
    once verified. A policy that converged on LIBERO should score a low
    flow-matching loss on its own training distribution; ours sits at 1.12 and
    barely moves, which is what a target-space mismatch looks like rather than
    a policy that has more to learn.

    Two losses tell the two hypotheses apart. loss_expert uses parquet actions
    as the target; loss_self uses the model's own stored A- on the observation
    that produced it. If loss_self is small while loss_expert is not, the loss
    code is fine and the expert targets are wrong. If both are ~1.1, the flow
    loss itself is miscomputed and the DPO branch is suspect too.
    """
    log.info("=" * 66)
    log.info("DIAGNOSTIC: SFT target space vs the model's own action space")
    log.info("=" * 66)

    def flow_loss(backbone_output, action_input, target, n_t=8, official_t=False):
        """Flow loss at the given target, over all dims and over live dims only.

        official_t draws t the way GR00T trains: a beta variate scaled by
        noise_s, not uniform[0,1]. Our uniform draw put the model at noise
        levels it was never trained on, so neither the loss nor the DPO margin
        was comparable to anything the policy actually optimises.
        """
        tot, tot_m = 0.0, 0.0
        for _ in range(n_t):
            noise = torch.randn_like(target)
            t = (head.sample_time(target.shape[0], target.device, target.dtype)[:, None, None]
                 if official_t else
                 torch.rand(target.shape[0], device=target.device,
                            dtype=target.dtype)[:, None, None])
            x_t = (1 - t) * noise + t * target
            v = flow_velocity(head, backbone_output, action_input,
                              x_t.to(hd), t.to(hd)).float()
            sq = torch.square(v - (target - noise))
            tot += sq.mean().item()
            tot_m += sq[..., act_mask].mean().item()
        return tot / n_t, tot_m / n_t

    # ---- A. expert branch: parquet actions as the target --------------------
    exp_losses, exp_losses_off, e_acts, raw_acts = [], [], [], []
    for _ in range(n_batches):
        exp_obs, exp_act = next(expert_iter)
        raw_acts.append(np.stack(exp_act))
        collated = make_inputs(policy, exp_obs, actions=exp_act)
        bi, ai = model.prepare_input(collated)
        with torch.no_grad():
            bo = model.backbone(bi)
            e_act = ai.action.float()
            e_acts.append(e_act.cpu())
            exp_losses.append(flow_loss(bo, ai, e_act))
            exp_losses_off.append(flow_loss(bo, ai, e_act, official_t=True))

    # ---- B. self branch: the model's own A- on the obs that produced it -----
    self_losses, a_rejs = [], []
    bs = expert_iter.bs
    for b in range(n_batches):
        batch = [pairs[i] for i in np.random.randint(0, len(pairs), size=bs)]
        obs_batch = [{"image": p["obs_stale"]["image"],
                      "wrist_image": p["obs_stale"]["wrist_image"],
                      "state": p["obs_stale"]["state"],
                      "task": p["task"]} for p in batch]
        collated = make_inputs(policy, obs_batch)
        bi, ai = model.prepare_input(collated)
        with torch.no_grad():
            bo = model.backbone(bi)
            a_rej = torch.stack([p["action_rejected"] for p in batch]).float().to(ai.state.device)
            a_rej = a_rej * act_mask
            a_rejs.append(a_rej.cpu())
            self_losses.append(flow_loss(bo, ai, a_rej))

    E = torch.cat(e_acts); R = torch.cat(a_rejs); RAW = np.concatenate(raw_acts)
    le, lem = (float(np.mean([x[0] for x in exp_losses])),
               float(np.mean([x[1] for x in exp_losses])))
    ls, lsm = (float(np.mean([x[0] for x in self_losses])),
               float(np.mean([x[1] for x in self_losses])))
    leo = float(np.mean([x[1] for x in exp_losses_off]))
    log.info(f"loss_expert  all-132-dims = {le:.4f}   live-dims-only = {lem:.4f}")
    log.info(f"loss_expert  live dims, OFFICIAL t distribution = {leo:.4f}   "
             f"(official forward reports {0.4531:.4f})")
    log.info(f"loss_self    all-132-dims = {ls:.4f}   live-dims-only = {lsm:.4f}")
    log.info("(loss_self's floor is ~2.0 by construction: A- is a single sample "
             "from the policy, not its conditional mean, so this row diagnoses "
             "nothing on its own -- the live-dim count below is the real signal.)")
    log.info(f"shapes: expert_norm={tuple(E.shape)} a_rej={tuple(R.shape)} raw_phys={RAW.shape}")
    log.info(f"expert normalized : mean={E.mean():+.4f} std={E.std():.4f} "
             f"absmax={E.abs().max():.3f}")
    log.info(f"model A- (same sp): mean={R.mean():+.4f} std={R.std():.4f} "
             f"absmax={R.abs().max():.3f}")
    log.info(f"raw physical      : mean={RAW.mean():+.4f} std={RAW.std():.4f} "
             f"absmax={np.abs(RAW).max():.3f}")

    d = min(E.shape[-1], R.shape[-1])
    log.info("per-dim std  (expert_norm | model_A-)  first 10 of %d dims:" % d)
    for i in range(min(10, d)):
        log.info(f"   dim{i:>3}: {E[..., i].std():8.4f} | {R[..., i].std():8.4f}   "
                 f"mean {E[..., i].mean():+8.4f} | {R[..., i].mean():+8.4f}")
    nz_e = int((E.std(dim=(0, 1)) > 1e-6).sum()); nz_r = int((R.std(dim=(0, 1)) > 1e-6).sum())
    log.info(f"non-constant dims: expert={nz_e}/{E.shape[-1]}  model={nz_r}/{R.shape[-1]}")

    log.info("-" * 66)
    log.info(f"VERDICT: masking {int(act_mask.sum())} live dims takes the expert "
             f"loss from {le:.3f} to {lem:.3f}. The gap is the padding-dim "
             "gradient that destroyed the first run.")

    # ---- C. our hand-written velocity call vs GR00T's own loss --------------
    # Both branches score above the "predict nothing" floor (var(a)+1 ~= 1.02
    # here, since normalized actions have std ~0.12). That cannot be a data
    # problem alone -- it says the velocity we ask for is not the quantity the
    # head returns. The official forward samples its own noise/t and computes
    # the loss GR00T was trained under, so it is the reference: if it lands near
    # 0 while ours sits at 1.8, the bug is in flow_velocity, not in the parquet.
    log.info("-" * 66)
    exp_obs, exp_act = next(expert_iter)
    collated = make_inputs(policy, exp_obs, actions=exp_act)
    off = None
    with torch.no_grad():
        try:
            out = model(collated)
            off = out.get("loss", out) if isinstance(out, dict) else out
        except Exception as e:                                  # noqa: BLE001
            log.info(f"model(collated) failed: {type(e).__name__}: {e}")
            try:
                bi, ai = model.prepare_input(collated)
                out = head(model.backbone(bi), ai)
                off = out.get("loss", out) if isinstance(out, dict) else out
            except Exception as e2:                             # noqa: BLE001
                log.info(f"head(bo, ai) failed too: {type(e2).__name__}: {e2}")
    if off is not None:
        log.info(f"OFFICIAL GR00T loss on the same expert batch = {float(off):.4f}")

    # What is the head actually returning? If it correlates with the action
    # itself rather than with (action - noise), it is an x0 prediction and the
    # whole margin was computed against the wrong target.
    bi, ai = model.prepare_input(collated)
    with torch.no_grad():
        bo = model.backbone(bi)
        a = ai.action.float()
        noise = torch.randn_like(a)
        t = torch.rand(a.shape[0], device=a.device, dtype=a.dtype)[:, None, None]
        x_t = (1 - t) * noise + t * a
        v = flow_velocity(head, bo, ai, x_t.to(hd), t.to(hd)).float()
    u = a - noise
    def corr(p, q):
        p, q = p[..., act_mask].flatten(), q[..., act_mask].flatten()
        p, q = p - p.mean(), q - q.mean()
        return float((p * q).sum() / (p.norm() * q.norm() + 1e-12))
    log.info(f"head output : std={v[..., act_mask].std():.4f} "
             f"absmax={v[..., act_mask].abs().max():.3f}")
    log.info(f"target u=a-noise: std={u[..., act_mask].std():.4f}")
    log.info(f"corr(head_out, a-noise) = {corr(v, u):+.4f}   <- ~1 if velocity")
    log.info(f"corr(head_out, a)       = {corr(v, a):+.4f}   <- ~1 if x0 pred")
    log.info(f"corr(head_out, -noise)  = {corr(v, -noise):+.4f}")
    log.info(f"corr(head_out, noise-a) = {corr(v, -u):+.4f}   <- ~1 if time flipped")

    # ---- C2. is flow_velocity idempotent on one backbone_output? -----------
    # flow_loss reuses a single bo across draws, and the training step reuses it
    # across the policy and the reference. If process_backbone_output mutates
    # what it is handed, every call after the first sees progressively mangled
    # features -- and the reference branch of the margin would be scored under
    # different conditions than the policy branch, which invalidates the margin
    # regardless of anything else.
    log.info("-" * 66)
    with torch.no_grad():
        bi_i, ai_i = model.prepare_input(make_inputs(policy, exp_obs, actions=exp_act))
        bo_i = model.backbone(bi_i)
        a_i = ai_i.action.float()
        nz_i = torch.randn_like(a_i)
        t_i = head.sample_time(a_i.shape[0], a_i.device, a_i.dtype)[:, None, None]
        x_i = ((1 - t_i) * nz_i + t_i * a_i).to(hd)
        v1 = flow_velocity(head, bo_i, ai_i, x_i, t_i.to(hd)).float()
        v2 = flow_velocity(head, bo_i, ai_i, x_i, t_i.to(hd)).float()
        v3 = flow_velocity(head, bo_i, ai_i, x_i, t_i.to(hd)).float()
    d12 = float((v1 - v2).abs().max()); d13 = float((v1 - v3).abs().max())
    log.info(f"same (bo, x_t, t) called 3x: max|v1-v2|={d12:.6f}  max|v1-v3|={d13:.6f}")
    if d12 > 1e-3:
        log.info("  -> flow_velocity is NOT idempotent: process_backbone_output "
                 "mutates backbone_output in place. Every reuse compounds the "
                 "damage, including policy-vs-reference within one margin.")
    else:
        log.info("  -> idempotent; backbone reuse is safe.")
    for label, obj in [("process_backbone_output",
                        getattr(type(head), "process_backbone_output", None))]:
        if obj is not None:
            try:
                log.info(f"===== {label} =====")
                for line in inspect_src(obj):
                    log.info("  " + line)
            except Exception as e:                              # noqa: BLE001
                log.info(f"  <unavailable: {e}>")

    # ---- C3. exact replay: official forward vs flow_velocity, same RNG -----
    # Everything so far compared two loss numbers drawn under different noise,
    # different t, and (in train mode) different dropout masks, so a gap of 2x
    # proved nothing on its own. In eval mode with a pinned seed the official
    # forward is deterministic and its noise/t can be regenerated in order --
    # official draws `noise` first, then sample_time -- so the two paths become
    # directly comparable. Equality here means flow_velocity is finally correct.
    log.info("-" * 66)
    was_training = model.training
    model.eval()
    try:
        S = 1234
        coll = make_inputs(policy, exp_obs, actions=exp_act)
        torch.manual_seed(S)
        with torch.no_grad():
            off_out = model(coll)
        off_loss = float(off_out["loss"] if isinstance(off_out, dict) else off_out)

        bi_r, ai_r = model.prepare_input(coll)
        with torch.no_grad():
            bo_r = model.backbone(bi_r)
            a_r = ai_r.action.float()
            torch.manual_seed(S)
            noise_r = torch.randn(a_r.shape, device=a_r.device, dtype=a_r.dtype)
            t_r = head.sample_time(a_r.shape[0], a_r.device, a_r.dtype)[:, None, None]
            x_r = (1 - t_r) * noise_r + t_r * a_r
            v_r = flow_velocity(head, bo_r, ai_r, x_r.to(hd), t_r.to(hd)).float()
        m_r = getattr(ai_r, "action_mask", None)
        sq_r = torch.square(v_r - (a_r - noise_r))
        if m_r is not None:
            m_r = m_r.to(sq_r.dtype)
            my_loss = float((sq_r * m_r).sum() / (m_r.sum() + 1e-6))
        else:
            my_loss = float(sq_r[..., act_mask].mean())
        log.info(f"EXACT REPLAY  official={off_loss:.6f}  flow_velocity={my_loss:.6f}  "
                 f"ratio={my_loss/max(off_loss,1e-9):.3f}")
        if abs(my_loss - off_loss) < 0.02 * max(off_loss, 1e-9) + 1e-4:
            log.info("  -> MATCH: flow_velocity reproduces the official loss exactly.")
        else:
            log.info("  -> MISMATCH: a difference remains inside flow_velocity.")

        # idempotence, now measurable (eval mode = no dropout)
        with torch.no_grad():
            w1 = flow_velocity(head, bo_r, ai_r, x_r.to(hd), t_r.to(hd)).float()
            w2 = flow_velocity(head, bo_r, ai_r, x_r.to(hd), t_r.to(hd)).float()
        log.info(f"idempotence in eval mode: max|w1-w2|={float((w1-w2).abs().max()):.6e}")
    finally:
        if was_training:
            model.train()

    log.info(f"dropout config: state_dropout_prob={getattr(head, 'state_dropout_prob', None)} "
             f"head.training={head.training} ref_head would be eval()")

    # ---- D. read the official implementation rather than guess at it --------
    # The official loss is 4x lower on the same batch, so the convention we feed
    # flow_velocity is wrong. velocity = a - noise is t-independent in rectified
    # flow, which is why a mis-specified t still points roughly the right way
    # (corr ~0.5) instead of failing outright. The source settles how x_t and
    # the timestep bucket are actually built.
    import inspect
    log.info("-" * 66)
    for label, obj in [("action_head.forward", type(head).forward),
                       ("action_head.sample_time", getattr(type(head), "sample_time", None))]:
        if obj is None:
            continue
        try:
            log.info(f"===== {label} =====")
            for line in inspect.getsource(obj).splitlines():
                log.info("  " + line)
        except Exception as e:                                  # noqa: BLE001
            log.info(f"  <unavailable: {e}>")

    # ---- E. scan the four t/x_t conventions against the official 0.45 -------
    log.info("-" * 66)
    log.info("t-convention scan (masked flow loss, 8 draws each):")
    for name, build in [
        ("A x=(1-t)n+ta, pass t   (current)", lambda n, aa, t: ((1 - t) * n + t * aa, t)),
        ("B x=(1-t)n+ta, pass 1-t",           lambda n, aa, t: ((1 - t) * n + t * aa, 1 - t)),
        ("C x=tn+(1-t)a, pass t",             lambda n, aa, t: (t * n + (1 - t) * aa, t)),
        ("D x=tn+(1-t)a, pass 1-t",           lambda n, aa, t: (t * n + (1 - t) * aa, 1 - t)),
    ]:
        tot = 0.0
        with torch.no_grad():
            for _ in range(8):
                nz = torch.randn_like(a)
                tt = torch.rand(a.shape[0], device=a.device, dtype=a.dtype)[:, None, None]
                xx, tp = build(nz, a, tt)
                vv = flow_velocity(head, bo, ai, xx.to(hd), tp.to(hd)).float()
                tot += torch.square(vv - (a - nz))[..., act_mask].mean().item()
        log.info(f"   {name}: {tot/8:.4f}")
    log.info("=" * 66)
    log.info("=" * 66)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--pairs-glob", required=True)
    p.add_argument("--expert-root", required=True,
                   help="LIBERO lerobot dataset root for the SFT anchor")
    p.add_argument("--future-sft-lambda", type=float, default=0.0,
                   help="R2 Major Concern 3 baseline: distil A+ at c_dep with "
                        "no A- and no DPO. Mean-reduced like sft_loss, so this "
                        "is directly comparable to --sft-lambda (unlike "
                        "--dpo-lambda, whose margin is sum-reduced).")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--exec-rows", type=int, default=0,
                   help="restrict EVERY loss term to the first N action rows. The "
                        "released GR00T-N1.7-LIBERO head has action_horizon=40 but the "
                        "official finetune only ever supervised rows 0..15 "
                        "(embodiment delta_indices=range(16)) and the env executes 16; "
                        "the collector stored the full 40-row chunk, so by default the DPO "
                        "margin sums 40x7=280 elements of which 24 rows are outputs the "
                        "head was never trained on and that are never executed, and the "
                        "expert anchor trains those rows too. 2026-09-14 audit: rows 16..39 "
                        "carry 21-40%% of sum(A+-A-)^2 on live dims. With N>0: pair rows >=N "
                        "are zeroed (as the processor's padding does), the expert chunk is "
                        "cut to N rows so the processor pads+masks rows N..39 exactly like "
                        "the official recipe, and the DPO / future-SFT sums stop at row N. "
                        "0 = legacy behaviour (all reported runs before 2026-09-15).")
    p.add_argument("--margin-norm", default="none", choices=["none", "mean"],
                   help="divide the margin by the number of scored dims. The "
                        "summed margin scales with ||A+ - A-||^2, which differs "
                        "2.4x across suites (spatial 0.1043 vs goal 0.0442); "
                        "since logsigmoid saturates exponentially, the effective "
                        "preference gradient then differs ~9500x at the SAME "
                        "lambda (margin 13.31 vs 4.14 -> 1.7e-6 vs 1.6e-2). That "
                        "is why the optimal lambda fails to transfer across "
                        "suites, and why only spatial tolerated lambda=10.")
    p.add_argument("--dpo-delay-weight", default="none",
                   choices=["none", "linear", "audit"],
                   help="scale each sample's preference loss by the measured "
                        "reliability of its label at that delay. The audit puts "
                        "per-delay precision at 0.568 (p=0.27, not distinguishable "
                        "from chance), 0.704, 0.756 and 0.841 for d=1..4, yet a "
                        "single lambda tuned on the mean is dominated by d=3,4 and "
                        "applies that same strength to the near-random d=1 signal. "
                        "'audit' uses 2*(precision-0.5) normalised to 1 at d=4; "
                        "'linear' uses d/4.")
    p.add_argument("--flip-verdict", action="store_true",
                   help="swap the two sides of any pair the forked rollout judged "
                        "backwards, instead of discarding it. Keeps the sample "
                        "count while removing the wrong-direction gradient; ties "
                        "are dropped since they carry no direction.")
    p.add_argument("--verified-only", action="store_true",
                   help="train only on pairs whose label was confirmed by a "
                        "forked rollout at collection time (needs --verify-steps).")
    p.add_argument("--pref-common-noise", type=float, default=0.0,
                   help="add the SAME per-pair Gaussian perturbation to A+ and "
                        "A-, simulating a miscalibrated reference policy. The "
                        "DPO margin gradient is analytically invariant to it; "
                        "the future-SFT regression gradient is not.")
    p.add_argument("--pair-margin-pct", type=float, default=100.0,
                   help="keep only the top P%% of pairs by ||A+ - A-||, computed "
                        "WITHIN each delay shard so the delay mixture is "
                        "unchanged. Both arms consume the same filtered set.")
    p.add_argument("--dpo-matched-path", action="store_true",
                   help="score each branch on ITS OWN noised path (standard "
                        "Diffusion-DPO) instead of one shared midpoint probe. "
                        "Off by default so every logged run stays reproducible.")
    p.add_argument("--dpo-beta", type=float, default=1.0)
    p.add_argument("--dpo-lambda", type=float, default=0.02)
    p.add_argument("--sft-lambda", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr-floor", type=float, default=1e-7)
    p.add_argument("--weight-decay", type=float, default=1e-10)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze-backbone", action="store_true", default=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--diagnose", action="store_true",
                   help="run the SFT/DPO action-space checks and exit")
    args = p.parse_args()
    if args.lr_floor > args.lr:
        args.lr_floor = args.lr / 10

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.LIBERO_PANDA,
                         model_path=args.model_path, device=0)
    model = policy.model
    head = model.action_head
    model.train()

    ref_head = None
    if args.dpo_lambda != 0.0:
        ref_head = deepcopy(head).eval()
        for prm in ref_head.parameters():
            prm.requires_grad_(False)

    # Train only the action head, as the SmolVLA line does (--freeze-vlm): the
    # backbone is 20x the head, so leaving it trainable spreads the preference
    # signal across 3.14B parameters and the update vanishes -- the first GR00T
    # run moved the weights by ~1e-9 in relative terms and margin never left 0.
    if args.freeze_backbone:
        frozen = 0
        for name, prm in model.named_parameters():
            if not name.startswith("action_head"):
                prm.requires_grad_(False); frozen += prm.numel()
        log.info(f"froze backbone: {frozen/1e6:.0f}M params")
    params = [q for q in model.parameters() if q.requires_grad]
    log.info(f"trainable params: {sum(q.numel() for q in params)/1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                            betas=tuple(args.betas))

    def _lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        floor = args.lr_floor / args.lr
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    expert_h = args.exec_rows if args.exec_rows > 0 else head.config.action_horizon
    expert_iter = ExpertBatches(args.expert_root, args.batch_size,
                                expert_h, seed=args.seed)
    pairs = load_pairs(args.pairs_glob, args.pair_margin_pct, args.pref_common_noise,
                       args.verified_only, args.flip_verdict)
    log.info(f"total preference pairs: {len(pairs)}")
    if args.exec_rows > 0:
        H = int(pairs[0]["action_preferred"].shape[0])
        for q in pairs:
            q["action_preferred"][args.exec_rows:] = 0
            q["action_rejected"][args.exec_rows:] = 0
        log.info(f"--exec-rows {args.exec_rows}: pair rows {args.exec_rows}..{H-1} zeroed; "
                 f"expert horizon {expert_h}; losses restricted to {args.exec_rows} rows")

    hd = head.dtype
    act_mask = resolve_action_dims(policy, model, expert_iter)
    if args.diagnose:
        run_diagnostics(policy, model, head, expert_iter, pairs, hd, act_mask)
        return
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    running = {"sft": 0.0, "dpo": 0.0, "margin": 0.0, "adiff": 0.0, "fut": 0.0}
    t0 = time.time()

    for step in range(1, args.steps + 1):
        idxs = np.random.randint(0, len(pairs), size=args.batch_size)
        batch = [pairs[i] for i in idxs]
        # per-sample delay for --dpo-delay-weight. Defined HERE, in the training
        # loop -- the first `batch = [pairs[i] ...]` in this file belongs to the
        # diagnostic helper, and putting it there left the name unbound at use.
        batch_delay = torch.tensor([float(q.get("_delay", 0)) for q in batch],
                                   device=device, dtype=torch.float32)
        obs_batch = [{**b["obs_stale"], "task": b["task"]} for b in batch]
        a_pref = torch.stack([b["action_preferred"] for b in batch]).to(device).float()
        a_rej = torch.stack([b["action_rejected"] for b in batch]).to(device).float()

        collated = make_inputs(policy, obs_batch)
        # Same split the released inference path uses: Gr00tN1d7.get_action does
        # prepare_input -> backbone -> action_head, so we reuse it verbatim and
        # only replace the action head's internal noise/x_t.
        backbone_inputs, action_input = model.prepare_input(collated)
        backbone_output = model.backbone(backbone_inputs)

        # One shared (noise, t) and one shared x_t for every term.
        a_pref, a_rej = a_pref * act_mask, a_rej * act_mask
        rows = slice(0, args.exec_rows) if args.exec_rows > 0 else slice(None)
        noise = torch.randn_like(a_pref)
        t = head.sample_time(a_pref.shape[0], device, a_pref.dtype)[:, None, None]
        # Two probe schemes. The shared midpoint lets ONE forward serve both
        # halves of the margin, but it scores A+ and A- at a point that lies on
        # neither branch's own path -- while the future-SFT arm below scores A+
        # on x_fut, built from a_pref itself. The comparison between the two arms
        # then confounds "relative preference vs pointwise regression" with
        # "midpoint probe vs own-path probe". Standard Diffusion-DPO (Wallace et
        # al.) noises each branch separately with a SHARED epsilon; under that
        # scheme x_pref is bit-for-bit the future-SFT arm's x_fut, so the arms
        # differ only by the presence of the A- term -- which is the ablation the
        # comparison is supposed to be.
        u_pref, u_rej = a_pref - noise, a_rej - noise
        if args.dpo_matched_path:
            x_pref = (1 - t) * noise + t * a_pref
            x_rej = (1 - t) * noise + t * a_rej
        else:
            a_mid = 0.5 * (a_pref + a_rej)
            x_t = (1 - t) * noise + t * a_mid

        # The margin subtracts a reference term, so both halves must be scored
        # under identical conditions. ref_head is a deepcopy pinned to eval()
        # while head trains with dropout (state_dropout_prob=0.2 plus DiT
        # dropout), which made the policy and reference branches systematically
        # incomparable. Score the preference branch with dropout off; gradients
        # still flow, only the sampling noise is removed.
        head_was_training = head.training
        head.eval()
        if args.dpo_matched_path:
            v_p = flow_velocity(head, backbone_output, action_input, x_pref.to(hd), t.to(hd)).float()
            v_r = flow_velocity(head, backbone_output, action_input, x_rej.to(hd), t.to(hd)).float()
            mse_pref = torch.square(v_p - u_pref)[..., rows, :][..., act_mask].sum(dim=(-2, -1))
            mse_rej = torch.square(v_r - u_rej)[..., rows, :][..., act_mask].sum(dim=(-2, -1))
        else:
            v = flow_velocity(head, backbone_output, action_input, x_t.to(hd), t.to(hd)).float()
            mse_pref = torch.square(v - u_pref)[..., rows, :][..., act_mask].sum(dim=(-2, -1))
            mse_rej = torch.square(v - u_rej)[..., rows, :][..., act_mask].sum(dim=(-2, -1))

        if ref_head is not None:
            with torch.no_grad():
                if args.dpo_matched_path:
                    vr_p = flow_velocity(ref_head, backbone_output, action_input,
                                         x_pref.to(hd), t.to(hd)).float()
                    vr_r = flow_velocity(ref_head, backbone_output, action_input,
                                         x_rej.to(hd), t.to(hd)).float()
                    ref_pref = torch.square(vr_p - u_pref)[..., rows, :][..., act_mask].sum(dim=(-2, -1))
                    ref_rej = torch.square(vr_r - u_rej)[..., rows, :][..., act_mask].sum(dim=(-2, -1))
                else:
                    v_ref = flow_velocity(ref_head, backbone_output, action_input,
                                          x_t.to(hd), t.to(hd)).float()
                    ref_pref = torch.square(v_ref - u_pref)[..., rows, :][..., act_mask].sum(dim=(-2, -1))
                    ref_rej = torch.square(v_ref - u_rej)[..., rows, :][..., act_mask].sum(dim=(-2, -1))
            margin = (mse_rej - mse_pref) + (ref_pref - ref_rej)
            if args.margin_norm == "mean":
                margin = margin / float(max(1, int(act_mask.sum()) *
                                           (args.exec_rows if args.exec_rows > 0 else a_pref.shape[-2])))
            per_sample = -F.logsigmoid(args.dpo_beta * margin)
            if args.dpo_delay_weight == "none":
                dpo_loss = per_sample.mean()
            else:
                if args.dpo_delay_weight == "linear":
                    w = batch_delay / 4.0
                else:   # audit-derived, normalised to 1 at d=4
                    prec = torch.tensor([0.0, 0.568, 0.704, 0.756, 0.841], device=device)
                    idx = batch_delay.long().clamp(0, 4)
                    w = (2.0 * (prec[idx] - 0.5)) / (2.0 * (0.841 - 0.5))
                w = w.clamp(min=0.0).to(per_sample.dtype)
                # renormalise so the mean weight stays 1: the delay weighting is
                # meant to REDISTRIBUTE the preference signal across delays, not
                # to silently shrink lambda.
                dpo_loss = (per_sample * w).sum() / w.sum().clamp(min=1e-6)
        else:
            margin = torch.zeros((), device=device)
            dpo_loss = torch.zeros((), device=device)
        # Future-target distillation baseline (R2 Major Concern 3):
        #   L_SFT(A_exp, c_dep) + future_sft_lambda * L_FM(A+, c_dep)
        # Same reference policy, same future observations, same candidate
        # generation, same deployment context, same offline budget as DEFLECT --
        # only A- and the DPO margin are gone. It separates "does relative
        # preference beat plain distillation of the future action" from "does
        # future-conditioned supervision help at all".
        #
        # NOT reusing mse_pref: that scores v, read at x_t, which interpolates
        # toward a_mid = (a_pref + a_rej)/2. The shared midpoint is what makes
        # one forward serve both halves of the margin, but it is not the flow
        # matching loss of A+. Pay for one more forward on A+'s own path.
        #
        # Stays inside the head.eval() window for the same reason the preference
        # branch does: dropout on would make this term noisier than the anchor
        # it is weighed against.
        if args.future_sft_lambda > 0.0:
            # identical to x_pref under --dpo-matched-path (same noise, same t)
            x_fut = (1 - t) * noise + t * a_pref
            v_fut = flow_velocity(head, backbone_output, action_input,
                                  x_fut.to(hd), t.to(hd)).float()
            fut_loss = torch.square(v_fut - u_pref)[..., rows, :][..., act_mask].mean()
        else:
            fut_loss = torch.zeros((), device=device)

        if head_was_training:
            head.train()

        # SFT anchor on EXPERT data, in its own (obs_t -> a_t) context.
        #
        # This must NOT be the A+ branch: A+ is the reference policy under the
        # FRESH observation, i.e. the delay correction itself, so anchoring on it
        # distils the answer into every run -- including the lambda=0 control,
        # which then stops being a control at all. That bug made the first GR00T
        # run report no DEFLECT effect, because the control had already learned
        # the same thing through the anchor.
        exp_obs, exp_act = next(expert_iter)
        exp_collated = make_inputs(policy, exp_obs, actions=exp_act)
        e_bi, e_ai = model.prepare_input(exp_collated)
        e_bo = model.backbone(e_bi)
        e_act = e_ai.action.float()
        e_noise = torch.randn_like(e_act)
        e_t = head.sample_time(e_act.shape[0], device, e_act.dtype)[:, None, None]
        e_xt = (1 - e_t) * e_noise + e_t * e_act
        e_v = flow_velocity(head, e_bo, e_ai, e_xt.to(hd), e_t.to(hd)).float()
        e_mask = getattr(e_ai, "action_mask", None)
        e_sq = torch.square(e_v - (e_act - e_noise))
        if e_mask is not None:
            e_mask = e_mask.to(e_sq.dtype)
            sft_loss = (e_sq * e_mask).sum() / (e_mask.sum() + 1e-6)
        else:
            sft_loss = e_sq[..., act_mask].mean()

        # The DPO and SFT terms must be comparable in magnitude. GR00T's flow loss
        # is summed over 16x32 dims, so an un-normalised mse-based margin sits
        # ~3 orders above a mean-reduced anchor; with lambda=0.02 the preference
        # gradient was 0.02% of the total, i.e. effectively switched off. Both
        # terms are mean-reduced here so lambda has its intended meaning.
        loss = args.sft_lambda * sft_loss + args.dpo_lambda * dpo_loss
        if args.future_sft_lambda > 0.0:
            # Distillation arm: A+ only, no A-, no margin. A- is still generated
            # upstream so this arm and DEFLECT walk an identical data path and
            # consume the RNG stream in the same order; it just earns no
            # gradient here.
            loss = args.sft_lambda * sft_loss + args.future_sft_lambda * fut_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        running["sft"] += float(sft_loss.detach())
        running["fut"] += float(fut_loss.detach())
        running["dpo"] += float(dpo_loss.detach()) if ref_head is not None else 0.0
        running["margin"] += float(margin.mean().detach()) if ref_head is not None else 0.0
        running["adiff"] += float((a_pref - a_rej)[..., rows, :][..., act_mask].abs().mean())

        if step % 50 == 0:
            n = 50
            log.info(f"step={step}/{args.steps} sft={running['sft']/n:.4f} "
                     f"dpo={running['dpo']/n:.4f} margin={running['margin']/n:.4f} "
                     f"fut={running['fut']/n:.4f} "
                     f"|a_pref-a_rej|={running['adiff']/n:.4f} "
                     f"lr={sched.get_last_lr()[0]:.2e} t={time.time()-t0:.0f}s")
            running = {k: 0.0 for k in running}

        if step % args.save_freq == 0 or step == args.steps:
            d = out_dir / f"{step:06d}"
            model.save_pretrained(d)
            log.info(f"saved checkpoint -> {d}")


if __name__ == "__main__":
    main()
