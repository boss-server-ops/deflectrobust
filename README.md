# DEFLECT — Code Release

Code and evaluation data accompanying **"DEFLECT: Temporal Counterfactual Preference Learning for Delay-Robust Asynchronous VLAs"**.

The Hong Kong University of Science and Technology (Guangzhou).

Contains:
- **`kinetix/`** — training and evaluation code for the Kinetix benchmark (small symbolic flow-matching policy, JAX/Flax).
- **`libero/`** — DEFLECT post-training and evaluation glue for the LIBERO benchmark on top of a PI0.5-style VLA (PyTorch).
- **`groot/`** — the same three stages for the standard-SFT GR00T N1.7 backbone on LIBERO: counterfactual pair collection, DEFLECT post-training, delay-aware evaluation, plus the paired-rollout label audit. See [`groot/README.md`](groot/README.md).

### What is and is not included

Trained checkpoints and evaluation outputs are not included in this repository.
What is provided is the training and evaluation code for all three backbones.

---

## Directory layout

```
.
├── README.md
├── requirements.txt
├── kinetix/
│   ├── src/                       # training + eval JAX code
│   │   ├── train_flow_full_dpo.py # DEFLECT training entry point
│   │   ├── eval_flow.py           # evaluation (Naive/VLASH/RTC/BID/DEFLECT/Oracle/Sync)
│   │   ├── model.py               # FlowPolicy + DPO loss
│   │   ├── train_expert.py        # env/wrappers/data
│   │   ├── generate_data.py       # expert data loader
│   │   └── compute_robot_indices.py
│   └── scripts/
│       ├── build_paper_figures.py # reproduce all Kinetix figures
│       ├── merge_results.py       # merge parallel eval shards
│       └── run_paper_eval_parallel.sh
├── libero/
│   └── scripts/
│       ├── train_dpo.py           # DEFLECT post-training on LIBERO
│       ├── collect_dpo_data.py    # fresh/stale counterfactual collection
│       └── eval_libero_base.py    # delay-aware LIBERO eval
├── groot/
│   ├── README.md                  # pipeline, protocol, reproduction notes
│   └── scripts/
│       ├── collect_pairs_groot.py # fresh/stale pair collection (GR00T N1.7)
│       ├── train_dpo_groot.py     # DEFLECT / Continued SFT / Future SFT
│       ├── eval_groot_delay.py    # delay sweep, K=8, 720-step episodes
│       └── pref_audit_groot.py    # paired-rollout preference-label audit
```

---

## Retraining DEFLECT

### Kinetix

Requires JAX + Flax + Kinetix env (see `requirements.txt`). The full DEFLECT recipe is in `kinetix/src/train_flow_full_dpo.py`:

```bash
python kinetix/src/train_flow_full_dpo.py \
    --config.run-path <path-to-expert-data> \
    --config.ref-checkpoint-dir <path-to-VLASH-async5-ckpt> \
    --config.async-interval 5 \
    --config.dpo-beta 1.0 \
    --config.dpo-lambda 0.02 \
    --config.sft-lambda 1.0 \
    --config.dpo-full-margin \
    --config.num-epochs 24 \
    --config.lr-schedule cosine \
    --config.output-dir <output-dir>
```

Single GPU; ~24 hours on one H100 80GB across all 12 Kinetix levels (12 policies trained in parallel via `jax.vmap` over the level axis; each policy is ~3M parameters).

### LIBERO

Requires PyTorch + `lerobot` 0.5.1 with the PI0.5 policy plugin. Build counterfactual preference data once, then DPO-train.

```bash
# 1) Collect counterfactual fresh/stale preference pairs
python libero/scripts/collect_dpo_data.py \
    --policy_path <path-to-VLASH-LIBERO-ckpt> \
    --task libero_spatial --num_tasks 10 --inits_per_task 10 \
    --delay 3 --output_path data/dpo_pairs/spatial_d3.pt

# 2) DEFLECT post-training
python libero/scripts/train_dpo.py \
    --policy_path <VLASH-ckpt> \
    --data_path "data/dpo_pairs/naive_libero_*_d*.pt" \
    --dataset_repo lerobot/libero \
    --steps 1000 --batch_size 8 --lr 1e-6 \
    --dpo_beta 1.0 --dpo_lambda 0.1 \
    --lam_mode none \
    --save_freq 100 \
    --output_dir <output-dir>
```

Single GPU. ~10 minutes for 1000 gradient steps on H100; the 200-step checkpoint is the one used in the headline LIBERO numbers.

### Evaluation (Kinetix)

```bash
bash kinetix/scripts/run_paper_eval_parallel.sh \
    <policy_run_path> <eval_output_dir> <num_gpus> <num_evals>
```

This runs all five evaluation methods (Naive / VLASH / RTC / BID / Oracle) at delays 0–7 with K=max(1,d) and at horizons 1–8 with d=1, on 12 Kinetix levels × 1024 rollouts per cell. The DEFLECT row is the same script, just evaluated on the DEFLECT-trained checkpoint with `method=vlash` (DEFLECT inherits the VLASH inference scheme; only the policy weights differ).

For the `Sync‡` reference (delay=0, K=1..7), pass `--eval-mode sync` to `eval_flow.py`.

## Evaluation scale

- Kinetix: 12 levels per cell, 1024 rollouts per (level, cell).
- LIBERO: 4 suites, 10 tasks per suite, 500 episodes per (suite, delay), for every
  backbone.

---

## License

MIT. See [`LICENSE`](LICENSE).
