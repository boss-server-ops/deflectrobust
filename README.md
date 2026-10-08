# DEFLECT — Anonymous Code & Data Release

Open-sourced code and evaluation data accompanying **"DEFLECT: Delay-Robust Execution via Flow-matching Likelihood-Estimated Counterfactual Tuning for VLA Policies"** (anonymous submission).

Contains:
- **`kinetix/`** — training and evaluation code for the Kinetix benchmark (small symbolic flow-matching policy, JAX/Flax).
- **`libero/`** — DEFLECT post-training and evaluation glue for the LIBERO benchmark on top of a PI0.5-style VLA (PyTorch).
- **`groot/`** — the same three stages for the standard-SFT GR00T N1.7 backbone on LIBERO: counterfactual pair collection, DEFLECT post-training, delay-aware evaluation, plus the paired-rollout label audit. See [`groot/README.md`](groot/README.md).
- **`figures/`** — scripts that consume the released eval data and reproduce every paper figure.
- **`data/`** — released evaluation results (per-method per-delay per-task CSVs) from every numerical claim in the paper.
- **`RAW_NUMBERS.md`** — every number plotted in every figure, inlined as Markdown tables (figure indices match the latest manuscript: Fig 3 – Fig 14).

### Checkpoint release

The DEFLECT-trained checkpoints — both the Kinetix per-task policies (12 flow-matching policies, JAX/Flax, ~3M params each) and the LIBERO π₀.₅ refinement (200-step DPO post-training on top of the released VLASH async5 base) — will be released upon acceptance. During the review period, only the training/evaluation code and the numerical eval results in `data/` are provided in this repo; trained weights are withheld to preserve double-blind anonymity.

---

## Directory layout

```
.
├── README.md
├── RAW_NUMBERS.md
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
├── figures/
│   └── (Markdown table of every figure → script mapping in RAW_NUMBERS.md)
└── data/
    ├── kinetix/
    │   ├── baselines/   # Naive, VLASH, RTC, BID, PFM, Sync‡ raw CSVs
    │   ├── deflect/     # main DEFLECT run delay + horizon sweeps
    │   ├── ablations/   # lambda sweep + component ablations + zero-shot Models A/C + scoring-context
    │   ├── mechanism/   # 12-task aggregate npz (ODE deflection energy, distribution-per-task)
    │   └── catapult/    # single-state d=6 catapult action distribution
    └── libero/
        ├── vlash/       # released VLASH-async5 ckpt eval, per (suite, delay)
        └── deflect/     # DEFLECT 200-step fine-tune eval, per (suite, delay)
```

---

## Reproducing figures from released data (no training)

All numerical results in the paper can be re-derived from `data/` without any training. The `RAW_NUMBERS.md` file gives every plotted value inlined as Markdown tables; the script `kinetix/scripts/build_paper_figures.py` regenerates the PDF panels directly from the CSVs / npz under `data/kinetix/`.

```bash
pip install -r requirements.txt
python kinetix/scripts/build_paper_figures.py
```

(LIBERO panels are simple line plots over the per-suite JSONs in `data/libero/`; a minimal plotting snippet is documented at the end of `RAW_NUMBERS.md`.)

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

# 2) DEFLECT post-training (200 gradient steps is the headline run)
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

---

## Eval CSV schema

Every CSV under `data/kinetix/` has the same 7 columns:

```
returned_episode_lengths, returned_episode_returns, returned_episode_solved,
delay, method, level, execute_horizon
```

- Each row is **one Kinetix level** at one `(delay, execute_horizon)` cell.
- `returned_episode_solved ∈ [0, 1]` is averaged over **1024 parallel rollouts** for that cell.
- To get the per-cell success rate reported in the paper, take the mean of `returned_episode_solved` across the 12 levels (i.e., `df.groupby(["method","delay","execute_horizon"])["returned_episode_solved"].mean() * 100`).
- 95% binomial CIs come from `(n=12 × 1024, p=mean)`, but in practice the per-cell rollout count (1024) is enough that the cross-task variance dominates.

LIBERO JSON schema (per `data/libero/{method}/libero_{suite}/d{d}.json`):

```jsonc
{
  "overall": {
    "pc_successes": 96.8,           // success rate (%)
    "avg_episode_length": 102.92,   // mean steps to terminate
    "avg_sum_rewards": 1.768,
    "avg_max_rewards": 0.968
  },
  "<task_name_1>": { ... per-task ... },
  ...
}
```

500 episodes per (suite, delay) cell.

---

## Data summary

- 12 Kinetix levels evaluated per cell, 1024 rollouts per (level, cell) → **12,288 rollouts** per cell figure
- 4 LIBERO suites × 10 task families per suite × 500 episodes per (suite, delay) → **5,000 episodes** per (suite, delay)
- DEFLECT main Kinetix run: 24 epochs × ~1980 minibatches × batch 512 ≈ 24.4M sample-passes
- DEFLECT main LIBERO run: 200 gradient steps × batch 8 = 1600 sample-passes (intentionally minimal; see paper Appendix on cosine-restart decomposition)

---

## License

Code and data released under MIT for the duration of review. After acceptance, the de-anonymized version will be hosted at a permanent URL stated in the camera-ready.
