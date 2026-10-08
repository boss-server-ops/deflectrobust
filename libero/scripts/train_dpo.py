#!/usr/bin/env python
"""Counterfactual DPO training for VLA delay robustness.

Two data sources in each training step (matching Kinetix's successful approach):
  1. Expert SFT: standard flow matching on VLASH expert data (anchor)
  2. DPO: counterfactual contrastive loss on model-generated pairs (refinement)

Usage:
    python -m vlash.train_dpo \
        --policy_path=outputs/train/pi05_vlash_fair_30k/checkpoints/030000/pretrained_model \
        --data_path=outputs/dpo_data/vlash_spatial_d4.pt \
        --output_dir=outputs/train/pi05_dpo_vlash
"""

import argparse
import logging
import os
import time
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True, help="Path to DPO data (.pt) or glob pattern")
    p.add_argument("--output_dir", type=str, default="outputs/train/pi05_dpo_full")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dpo_beta", type=float, default=1.0, help="DPO temperature")
    p.add_argument("--dpo_lambda", type=float, default=0.1, help="DPO loss weight")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_freq", type=int, default=2000)
    p.add_argument("--log_freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    # Expert data config (same as VLASH training)
    p.add_argument("--dataset_repo", type=str, default="lerobot/libero")
    p.add_argument("--max_delay_steps", type=int, default=4)
    # LAM: Latency-Adaptive Margin
    p.add_argument("--lam_mode", type=str, default="none", choices=["none", "margin", "weight", "margin_pdn"],
                   help="LAM mode: none=standard DPO, margin=subtractive margin, weight=per-sample weighting")
    p.add_argument("--lam_gamma", type=float, default=1.0,
                   help="LAM strength. margin: sigmoid shift per delay step; weight: linear scale per delay step")
    return p.parse_args()


class DPODataset(torch.utils.data.Dataset):
    """Dataset wrapping collected preference pairs."""
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        item = {}
        for k, v in pair["stale_images"].items():
            if v.dim() == 4 and v.shape[0] == 1:
                v = v.squeeze(0)
            item[k] = v
        state = pair["state"]
        if state.dim() == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        item["observation.state"] = state
        item["task"] = pair["task"]
        item["action_preferred"] = pair["action_preferred"]
        item["action_rejected"] = pair["action_rejected"]
        item["delay"] = torch.tensor(pair.get("delay", 1), dtype=torch.float32)
        return item


def collate_dpo(batch):
    result = {}
    for k in batch[0].keys():
        if k == "task":
            result[k] = [b[k] for b in batch]
        else:
            result[k] = torch.stack([b[k] for b in batch])
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Load preference data (supports glob pattern for multiple files)
    import glob as glob_module
    data_files = sorted(glob_module.glob(args.data_path))
    if not data_files:
        data_files = [args.data_path]  # single file

    # Subsample to avoid OOM: each file has ~15-22K pairs with full image tensors
    # Keep memory under 200G by limiting total pairs
    max_pairs_per_file = 3000  # 3K × 16 files = 48K total, fits in ~60G

    pairs = []
    for f in data_files:
        logging.info(f"Loading {f}...")
        d = torch.load(f, map_location="cpu")
        file_pairs = d["pairs"]
        if len(file_pairs) > max_pairs_per_file:
            indices = torch.randperm(len(file_pairs))[:max_pairs_per_file].tolist()
            file_pairs = [file_pairs[i] for i in indices]

        # Extract delay from metadata or filename (e.g. "naive_libero_spatial_d3.pt" → 3)
        file_delay = d.get("metadata", {}).get("delay", None)
        if file_delay is None:
            import re
            m = re.search(r'_d(\d+)\.pt$', f)
            file_delay = int(m.group(1)) if m else 1
        for p in file_pairs:
            p["delay"] = file_delay

        pairs.extend(file_pairs)
        logging.info(f"  {len(file_pairs)} pairs, delay={file_delay} (total: {len(pairs)})")
        del d
        import gc; gc.collect()
    logging.info(f"Loaded {len(pairs)} total preference pairs from {len(data_files)} files")
    if args.lam_mode != "none":
        delay_counts = {}
        for p in pairs:
            d = p.get("delay", 1)
            delay_counts[d] = delay_counts.get(d, 0) + 1
        logging.info(f"LAM mode={args.lam_mode}, gamma={args.lam_gamma}, delay distribution: {delay_counts}")

    # Load expert dataset for SFT anchor
    logging.info(f"Loading expert dataset: {args.dataset_repo}")
    from vlash.datasets.vlash_dataset import VLASHDataset
    from lerobot.configs.default import DatasetConfig
    import vlash.configs  # noqa: register configs

    # Load policy using the same pipeline as eval (make_policy approach)
    # This correctly handles config parsing + weight loading
    from vlash.policies.factory import make_policy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    import vlash.configs  # noqa: register configs

    # Parse config the same way eval does (handles extra fields gracefully)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path

    # Create env config for feature inference
    env_cfg = LiberoEnv(task="libero_spatial")

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    logging.info(f"Model loaded via make_policy: {sum(p.numel() for p in policy.parameters())/1e6:.0f}M params")

    fps = 10  # LIBERO default
    chunk_size = policy.config.chunk_size
    n_obs_steps = policy.config.n_obs_steps

    # Map model feature names to dataset column names
    # Official model uses "wrist_image" but lerobot/libero dataset uses "image2"
    model_to_dataset = {"observation.images.wrist_image": "observation.images.image2"}
    dataset_to_model = {v: k for k, v in model_to_dataset.items()}

    delta_timestamps = {
        "action": [i / fps for i in range(chunk_size)],
        "observation.state": [i / fps for i in range(1 - n_obs_steps, 1)],
    }
    for key in policy.config.image_features:
        ds_key = model_to_dataset.get(key, key)  # use dataset column name
        delta_timestamps[ds_key] = [i / fps for i in range(1 - n_obs_steps, 1)]

    expert_dataset = VLASHDataset(
        args.dataset_repo,
        delta_timestamps=delta_timestamps,
        max_delay_steps=args.max_delay_steps,
        use_state_ground_truth=True,  # LIBERO needs ground truth state
    )
    def collate_expert(batch):
        """Collate that squeezes extra dims from VLASHDataset."""
        from torch.utils.data._utils.collate import default_collate
        # Squeeze any [1, D] tensors to [D] before stacking
        fixed = []
        for item in batch:
            for k, v in item.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[0] == 1:
                    item[k] = v.squeeze(0)
            fixed.append(item)
        return default_collate(fixed)

    expert_loader = torch.utils.data.DataLoader(
        expert_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, drop_last=True, collate_fn=collate_expert,
    )
    logging.info(f"Expert dataset: {len(expert_dataset)} samples")

    # Move policy to device
    policy.to(device)

    # Create frozen reference policy (for correct DPO margin)
    import copy
    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False
    logging.info("Created frozen reference policy for DPO margin")

    total_params = sum(p.numel() for p in policy.parameters())
    logging.info(f"Model: {total_params/1e6:.0f}M params")

    # Freeze VLM backbone, only train action expert + suffix
    for name, param in policy.named_parameters():
        if "action_expert" in name or "suffix_embedder" in name or "action_out_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"Trainable: {trainable/1e6:.1f}M / {total_params/1e6:.0f}M params (VLM frozen)")

    # DPO dataloader
    dpo_dataset = DPODataset(pairs)
    dpo_loader = torch.utils.data.DataLoader(
        dpo_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_dpo, drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    os.makedirs(args.output_dir, exist_ok=True)

    policy.train()
    step = 0
    expert_iter = iter(expert_loader)
    dpo_iter = iter(dpo_loader)
    running_sft = 0.0
    running_dpo = 0.0
    running_total = 0.0
    start_time = time.time()

    while step < args.steps:
        # --- Expert SFT batch ---
        try:
            expert_batch = next(expert_iter)
        except StopIteration:
            expert_iter = iter(expert_loader)
            expert_batch = next(expert_iter)

        # Standard policy forward on expert data (same as VLASH training)
        # Rename dataset column names to model feature names (e.g. image2 → wrist_image)
        expert_batch = {dataset_to_model.get(k, k): v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in expert_batch.items()}
        sft_loss, _ = policy.forward(expert_batch)

        # --- DPO batch ---
        try:
            dpo_batch = next(dpo_iter)
        except StopIteration:
            dpo_iter = iter(dpo_loader)
            dpo_batch = next(dpo_iter)

        for k in dpo_batch:
            if isinstance(dpo_batch[k], torch.Tensor):
                dpo_batch[k] = dpo_batch[k].to(device)

        # Prepare DPO inputs
        norm_batch = policy.normalize_inputs(dpo_batch)
        images, img_masks = policy.prepare_images(norm_batch)
        state = policy.prepare_state(norm_batch)
        lang_tokens, lang_masks = policy.prepare_language(norm_batch)

        action_preferred = dpo_batch["action_preferred"].to(device)
        action_rejected = dpo_batch["action_rejected"].to(device)

        max_adim = policy.config.max_action_dim
        if action_preferred.shape[-1] < max_adim:
            action_preferred = F.pad(action_preferred, (0, max_adim - action_preferred.shape[-1]))
            action_rejected = F.pad(action_rejected, (0, max_adim - action_rejected.shape[-1]))

        # Forward pass: policy (trainable) + reference (frozen, same noise/time)
        result = policy.model.forward(
            images, img_masks, lang_tokens, lang_masks, state,
            action_preferred, return_velocity=True,
        )
        dpo_sft_losses, _, v_t, noise, time_val = result

        # Reference model velocity (frozen, no grad) — needed for correct DPO margin
        with torch.no_grad():
            ref_result = ref_policy.model.forward(
                images, img_masks, lang_tokens, lang_masks, state,
                action_preferred, noise=noise, time=time_val, return_velocity=True,
            )
            _, _, v_ref, _, _ = ref_result

        u_preferred = noise - action_preferred
        u_rejected = noise - action_rejected

        # Full DPO margin: f(θ) + C
        # f(θ) = ||v_θ - u_rej||² - ||v_θ - u_pref||²  (trainable part)
        # C     = ||v_ref - u_pref||² - ||v_ref - u_rej||²  (reference constant)
        # Dropping C is WRONG because log-sigmoid is nonlinear:
        # gradient = -(1 - σ(f+C)) * ∇f  ≠  -(1 - σ(f)) * ∇f
        sq_pref = F.mse_loss(v_t, u_preferred, reduction="none")[:, :, :max_adim]
        sq_rej = F.mse_loss(v_t, u_rejected, reduction="none")[:, :, :max_adim]
        sq_ref_pref = F.mse_loss(v_ref, u_preferred, reduction="none")[:, :, :max_adim]
        sq_ref_rej = F.mse_loss(v_ref, u_rejected, reduction="none")[:, :, :max_adim]

        mse_pref = sq_pref.sum(dim=(-1, -2))
        mse_rej = sq_rej.sum(dim=(-1, -2))
        mse_ref_pref = sq_ref_pref.sum(dim=(-1, -2))
        mse_ref_rej = sq_ref_rej.sum(dim=(-1, -2))

        # Full margin: (log π_θ(w) - log π_ref(w)) - (log π_θ(l) - log π_ref(l))
        # = (-mse_pref + mse_ref_pref) - (-mse_rej + mse_ref_rej)
        # = (mse_rej - mse_pref) + (mse_ref_pref - mse_ref_rej)
        margin = (mse_rej - mse_pref) + (mse_ref_pref - mse_ref_rej).detach()

        # LAM: Latency-Adaptive Margin
        delay_vals = dpo_batch["delay"].to(device)  # shape: (B,)
        if args.lam_mode == "margin_pdn":
            # Kinetix-proven config: margin shift + per-delay normalization
            # γ=0.1 margin shift prevents sigmoid saturation at high delay
            # Per-delay norm uses groups including d=0 (which contributes 0),
            # acting as implicit regularizer that dampens DPO by factor ~N-1/N
            lam_shift = args.lam_gamma * (delay_vals - 1).float()
            per_sample_loss = -F.logsigmoid(args.dpo_beta * margin - lam_shift)
            # Per-delay normalization: each delay group gets equal weight
            delay_min = int(delay_vals.min().item())
            delay_max = int(delay_vals.max().item())
            group_losses = []
            for d_val in range(delay_min, delay_max + 1):
                mask_d = (delay_vals == d_val)
                count_d = mask_d.sum().clamp(min=1)
                loss_d = (per_sample_loss * mask_d.float()).sum() / count_d
                group_losses.append(loss_d)
            dpo_loss = torch.stack(group_losses).mean()
        elif args.lam_mode == "margin":
            # Subtractive margin only (no per-delay norm)
            lam_shift = args.lam_gamma * (delay_vals - 1)
            per_sample_loss = -F.logsigmoid(args.dpo_beta * margin - lam_shift)
            dpo_loss = per_sample_loss.mean()
        elif args.lam_mode == "weight":
            # Per-sample weighting
            lam_weight = 1.0 + args.lam_gamma * (delay_vals - 1)
            per_sample_loss = -F.logsigmoid(args.dpo_beta * margin)
            dpo_loss = (lam_weight * per_sample_loss).mean()
        else:
            dpo_loss = -F.logsigmoid(args.dpo_beta * margin).mean()

        # Total: expert SFT (anchor) + DPO (refinement)
        total_loss = sft_loss + args.dpo_lambda * dpo_loss

        optimizer.zero_grad()
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1
        running_sft += sft_loss.item()
        running_dpo += dpo_loss.item()
        running_total += total_loss.item()

        if step % args.log_freq == 0:
            avg_sft = running_sft / args.log_freq
            avg_dpo = running_dpo / args.log_freq
            avg_total = running_total / args.log_freq
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            logging.info(
                f"step={step}/{args.steps} sft={avg_sft:.4f} dpo={avg_dpo:.3f} "
                f"total={avg_total:.4f} lr={lr:.1e} time={elapsed:.0f}s"
            )
            running_sft = 0.0
            running_dpo = 0.0
            running_total = 0.0

        if step % args.save_freq == 0:
            ckpt_dir = Path(args.output_dir) / "checkpoints" / f"{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_dir = ckpt_dir / "pretrained_model"
            save_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(save_dir)
            logging.info(f"Saved checkpoint to {ckpt_dir}")

    final_dir = Path(args.output_dir) / "checkpoints" / f"{step:06d}"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_dir = final_dir / "pretrained_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    logging.info(f"Training complete! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
