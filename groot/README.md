# GR00T N1.7 — DEFLECT pipeline on LIBERO

Standard-SFT backbone. Three stages: collect counterfactual preference pairs
with a frozen reference policy, post-train with the DEFLECT objective, evaluate
under simulated inference delay.

## Requirements

- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (`gr00t` importable)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- The released per-suite checkpoints `nvidia/GR00T-N1.7-LIBERO/<suite>`
- The Cosmos backbone the checkpoints expect (`nvidia/Cosmos-Reason2-2B`);
  the LIBERO checkpoints do not bundle the VLM
- Expert demonstrations in LeRobot format for the SFT anchor

`LIBERO_CONFIG_PATH` is read at `import libero` time and defaults to `~/.libero`.
Point it at a directory of your own so a shared install is not modified:

```bash
export LIBERO_CONFIG_PATH=~/.libero_deflect
```

## 1. Collect preference pairs

For each delay `d`, the frozen reference policy is rolled out with a `d`-step
stale observation. At every replanning boundary it is queried twice under the
same sampling noise: once on the fresh observation `o_{t+d}` (preferred) and
once on the stale observation `o_t` (rejected). The rollout executes the stale
chunk, so the visited states are the ones deployment actually sees.

```bash
for d in 1 2 3 4; do
  python collect_pairs_groot.py \
    --model-path  /path/to/GR00T-N1.7-LIBERO/libero_spatial \
    --suite       libero_spatial \
    --delay       $d \
    --action-steps 8 \
    --max-pairs   3000 \
    --per-task \
    --out pairs/shard_spatial_d${d}.pt
done
```

`--action-steps` must match the evaluation protocol. Pairs collected at a
different replanning interval describe a different problem than the one the
policy is deployed on. `--per-task` spends the budget evenly over the suite's
ten tasks; without it the budget is exhausted on the first one or two.

Optional:

| flag | effect |
|---|---|
| `--verify-steps N` | fork the simulator at each pair and roll both chunks `N` steps, recording which one earns more return |
| `--keep-top K --keep-mode {margin,random,consistency}` | over-collect, then keep `K` pairs per task by the chosen criterion |
| `--average-k K` | store the mean of `K` reference draws per branch instead of a single sample |
| `--confidence-k K` | record the spread of `K` draws of the preferred chunk |

## 2. Post-train

```bash
python train_dpo_groot.py \
  --model-path  /path/to/GR00T-N1.7-LIBERO/libero_spatial \
  --expert-root /path/to/libero_expert/libero_spatial \
  --pairs-glob  'pairs/shard_spatial_d*.pt' \
  --steps 2000 --batch-size 4 \
  --lr 1e-5 --lr-floor 1e-6 --warmup-steps 100 \
  --dpo-beta 1.0 --dpo-lambda 0.1 --sft-lambda 1.0 \
  --seed 0 --save-freq 2000 \
  --output-dir runs/spatial_deflect
```

Controls and variants, same recipe otherwise:

| setting | flags |
|---|---|
| Continued SFT (λ_DPO = 0) | `--dpo-lambda 0` |
| Future SFT (regress to `A+`) | `--dpo-lambda 0 --future-sft-lambda 30` |
| Per-candidate interpolation | `--dpo-matched-path` |
| Restrict every loss term to the executed rows | `--exec-rows 16` |

`--exec-rows` is worth knowing about. The released head emits a 40-row action
tensor while the LIBERO embodiment declares `delta_indices = range(16)`, so
rows 16..39 are outputs the official finetune never supervised and the
environment never executes. Without the flag they are inside every loss term.

## 3. Evaluate

```bash
python eval_groot_delay.py \
  --model-path runs/spatial_deflect/002000 \
  --suite libero_spatial \
  --delays 1,2,3,4 \
  --n-episodes 500 \
  --action-steps 8 --max-steps 720 \
  --num-workers 12 \
  --out runs/spatial_deflect/eval.json
```

`--action-steps 8 --max-steps 720` is the protocol in Isaac-GR00T's
`examples/LIBERO/README.md`. The `RolloutConfig` dataclass default of 504 steps
truncates the long-horizon suite.

## 4. Audit the preference labels (optional)

`pref_audit_groot.py` runs paired rollouts from a restored simulator state: one
branch executes the fresh-observation chunk, the other the stale one, and the
episode returns are compared.

```bash
python pref_audit_groot.py \
  --model-path /path/to/GR00T-N1.7-LIBERO/libero_spatial \
  --suite libero_spatial --delay 4 \
  --out runs/pref_audit_spatial_d4.json
```

## Notes for anyone reproducing this

- Use `SubprocVectorEnv` with the `spawn` start method. `DummyVectorEnv` aborts
  when the parent process has already initialized CUDA, and forked workers
  inherit the CUDA context and abort the same way.
- Inside a container, `os.cpu_count()` reports the host's core count, not the
  cgroup grant. Read `/sys/fs/cgroup/cpu.max` and set `OMP_NUM_THREADS` and
  friends *before* importing numpy or torch.
- LIBERO evaluation is CPU-bound: the wall time is set by MuJoCo stepping and
  two EGL renders per step, not by the GPU.
- Results from parallel workers are not bit-identical to serial ones; the
  policy samples its own noise and each worker has an independent RNG. Check
  statistical agreement, not equality.
