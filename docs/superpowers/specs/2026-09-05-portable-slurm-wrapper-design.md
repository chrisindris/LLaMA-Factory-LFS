# Portable, repo-relative SLURM wrapper for CoT SFT

Date: 2026-09-05
Branch: `training_improvement`
Status: approved (design)

## Goal

Add a new SLURM wrapper for the `qwen2_5vl_lora_sft_CoT_traineval` experiment that
uses the same two-file strategy as
`models/qwen2_5vl_lora_sft_CoT/trillium_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh`
(thin `#SBATCH` header delegating to a shared body), but resolves **every** path
relative to the repository root so the checkout can be moved, renamed, or copied
to another cluster and still run.

Non-goal: changing the existing `trillium_*`, `killarney_*`, `nibi_*`, or
`rorqual_*` scripts. They keep working unchanged.

## Problem: hardcoded paths span five layers

Making the wrapper portable requires isolating all five layers behind one
resolver. Evidence from the current tree:

| Layer | File | Non-portable content |
|---|---|---|
| 1 | `models/qwen2_5vl_lora_sft_CoT/trillium_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh` | repo root inferred by matching the literal string `LLaMA-Factory-LFS` in `$PWD` (lines 15-22); `#SBATCH --mail-user=christopher.indris@torontomu.ca` (line 8) |
| 2 | `models/qwen2_5vl_lora_sft_CoT/slurm_qwen2_5vl_lora_sft_CoT_traineval.sh` | 18 occurrences of `/scratch/indrisch`, `/project/...`, `/home/indrisch`: H5 roots (59, 89-91), overlay (170), home bind (175, 317, 425), ScanNet parent binds (134-137), venv copies (257, 350), `pushd` into another tree (280, 474), Killarney venv (464) |
| 3 | `scripts/sysconfig.json` | 11 of 13 `TRILLIUM` values live under `/scratch/indrisch` or `/project/def-wangcs` |
| 4 | `examples/train_lora/trillium_qwen2_5vl_lora_sft_CoT_traineval.yaml` | `cache_dir` and `media_dir` absolute (`output_dir` and `deepspeed` are already relative) |
| 5 | `data/dataset_info.json` | `Scene30k` parquet and `SpatialSSRL_coldstart` JSON are absolute hub-snapshot paths; `3DThinker10k` is already repo-relative |

Two defects are worth fixing explicitly in the new path:

1. Root detection depends on the checkout being *named* `LLaMA-Factory*`. Renaming
   the directory breaks the job before it starts.
2. Line 280 runs `pushd /scratch/indrisch/LLaMA-Factory`, executing a **different
   tree** than the one submitted from.

## Chosen approach

Approach A: a portable wrapper and body over a **shared resolver**, plus a
`${PROJECT_DIR}`-aware `sysconfig` section. Rejected alternatives:

- **B, one self-contained script.** Smallest blast radius, but duplicates ~200
  lines of the existing body (drift) and leaves layer 5 unsolved.
- **C, refactor existing cluster scripts in place.** Fixes the coupling repo-wide
  but is a large diff against scripts with jobs in flight, and contradicts both
  the "new wrapper" framing and the repo's minimal-change guardrail.

A is chosen because it mirrors the Trillium strategy being copied and lets future
cluster wrappers inherit portability without edits.

## Architecture

Five new files and two small edits:

```text
scripts/utils/portable_env.sh                 NEW  resolver: root, site.env, defaults, preflight
scripts/site.env.example                      NEW  documented override template
scripts/sysconfig.json                        EDIT add "PORTABLE" section using ${PROJECT_DIR}
scripts/sysconfigtool.py                      EDIT expand ${VAR} in read()/read_all()
models/qwen2_5vl_lora_sft_CoT/
  portable_slurm_qwen2_5vl_lora_sft_CoT_traineval.sh   NEW  thin SBATCH wrapper
  portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh    NEW  portable job body
examples/train_lora/
  portable_qwen2_5vl_lora_sft_CoT_traineval.yaml       NEW  no absolute paths
```

### Components and boundaries

**`scripts/utils/portable_env.sh`** — the only component that knows how to turn a
machine into a set of paths. Sourced, not executed. Contract:

- Input: optional env overrides, optional `scripts/site.env`.
- Output: exported `PROJECT_DIR`, `CLUSTER`, `RUNNING_MODE`, and every path
  variable in the table below; plus `preflight_check()`.
- Depends on: `bash`, `git` (optional fallback), `python3` (only for sysconfig).

It must not know about any specific experiment. Existing
`scripts/utils/env.sh` is left alone; the new file is a portable sibling so the
two can coexist while old scripts still source the old one.

**`portable_slurm_*.sh`** — `#SBATCH` resources only, then `exec` the body. No
path logic beyond locating the body next to itself.

**`portable_body_*.sh`** — sources the resolver, runs preflight, then dispatches
to `APPTAINER` / `VENV` / `SHELL`. Knows about this experiment's YAML and output
directory; knows nothing about site layout.

## Path resolution contract

Every path gets a repo-relative default and an override. Nothing is tied to a
username.

| Concern | Today | Portable default | Override |
|---|---|---|---|
| Repo root | `$PWD` name match | `${BASH_SOURCE[0]}`, falling back to `git rev-parse --show-toplevel` | `LFS_PROJECT_DIR` |
| HF cache | `/scratch/indrisch/huggingface/hub` | `$PROJECT_DIR/.cache/huggingface` | `HF_HOME`, `HF_HUB_CACHE` |
| SIF image | sysconfig absolute | `$PROJECT_DIR/containers/llamafactory.sif` | `SIF_FILE` |
| Apptainer overlay | `/scratch/indrisch/LLaMA-Factory/apptainer/overlay.img` | `$PROJECT_DIR/apptainer/overlay.img` | `APPTAINER_OVERLAY` |
| Venv | `/scratch/indrisch/venv_llamafactory_cu126` | `$PROJECT_DIR/.venv` | `VENV_LLAMAFACTORY` |
| ScanNet H5 | `/scratch/indrisch/ScanNet_h5/scans` | `$PROJECT_DIR/data/h5/ScanNet_h5/scans` | `SCANNET_H5_DIR` |
| Spatial-SSRL H5 | `/scratch/indrisch/Spatial-SSRL_images_h5` | `$PROJECT_DIR/data/h5/Spatial-SSRL_images_h5` | `SPATIALSSRL_H5_DIR` |
| 3DThinker H5 | `/scratch/indrisch/3DThinker10K_images_h5` | `$PROJECT_DIR/data/h5/3DThinker10K_images_h5` | `THINKER10K_H5_DIR` |
| Home bind | `-B /home/indrisch` | `-B "$HOME"` | `EXTRA_BINDS` (space-separated `-B src[:dst[:ro]]` args, appended after the defaults) |
| Scratch caches | `/scratch/indrisch/.triton_cache` | `${SLURM_TMPDIR:-$PROJECT_DIR/.cache}/.triton_cache` | `TRITON_CACHE_DIR` |
| Torch extensions | `/scratch/indrisch/.cache/torch_extensions` | `${SLURM_TMPDIR:-$PROJECT_DIR/.cache}/torch_extensions` | `TORCH_EXTENSIONS_DIR` |
| W&B dir | `$PROJECT_DIR/wandb/` | unchanged (already relative) | `WANDB_DIR` |
| Working dir | `pushd /scratch/indrisch/LLaMA-Factory` | `cd "$PROJECT_DIR"` | — |
| Mail / account | hardcoded `--mail-user` | omitted from `#SBATCH` | `sbatch --mail-user=`, `sbatch -A` |

Deliberately still absolute, because they are machine-level rather than
user-level: cvmfs module library paths (`MPI_LIB_PATH`, `HWLOC_LIB_PATH`),
`/etc/ssl/certs`, `/etc/pki`, `/dev/shm`, `$SLURM_TMPDIR`.

Two settings are **YAML keys, not environment variables**, so the resolver cannot
export them. LLaMA-Factory does not read `MEDIA_DIR` or `DATASET_DIR` from the
environment. They are therefore written as repo-relative values in the portable
YAML and resolve against the working directory, which the body sets to
`$PROJECT_DIR`:

- `media_dir: data/h5/ScanNet_h5`
- `dataset_dir: data/annotations`

Both can still be overridden per-run using the CLI `key=value` form that
`llamafactory-cli train` accepts after the YAML path, e.g.
`... train <yaml> media_dir=/some/other/root`. The body forwards `"$@"` so this
works through `sbatch`.

### Override precedence

Highest wins:

1. Environment already set when the job starts (e.g. `sbatch --export`).
2. `scripts/site.env` if present.
3. `sysconfig.json` values for the detected `CLUSTER`, when not `"None"`.
4. Repo-relative defaults from the table.

## Data staging

Large artifacts (H5 stores, hub snapshots, SIF) stay where they physically live.
They are made reachable repo-relative with **symlinks**, so nothing is copied:

```text
data/h5/ScanNet_h5            -> /path/to/real/ScanNet_h5
data/h5/Spatial-SSRL_images_h5 -> ...
data/h5/3DThinker10K_images_h5 -> ...
data/annotations/             staged annotation files (or symlinks)
containers/llamafactory.sif   -> /path/to/real/llamafactory.sif
.cache/huggingface            -> /path/to/real/hub
```

Staging is performed by a `stage_portable_assets()` function in
`scripts/utils/portable_env.sh`, invoked explicitly by running the body with
`PORTABLE_STAGE=1` (never implicitly during a training run). It creates the
directories above, creates symlinks only where the target is provided via
`scripts/site.env`, and is idempotent: existing correct symlinks are left alone
and it never overwrites a real directory.

This is what removes the two absolute `data/dataset_info.json` entries. The same
staging step writes `data/annotations/dataset_info.json` as a full copy of the
original registry in which **every** `file_name` is normalized to be relative to
`data/annotations/`:

- `Scene30k` and `SpatialSSRL_coldstart`: absolute hub-snapshot paths become
  relative paths into `data/annotations/`, where the staging step symlinks the
  parquet and JSON.
- Already-relative entries such as `3DThinker10k`
  (`3DThinker-10K/out/3dthinker10k_cot.jsonl`) are prefixed with `../` so they
  still resolve, because `dataset_dir` moves from `data` to `data/annotations`
  and the loader joins `dataset_dir` with `file_name`.

The portable YAML sets `dataset_dir: data/annotations`. The original
`data/dataset_info.json` is never modified, so existing jobs are unaffected.

`.gitignore` already covers `*.sif`, `saves/`, `wandb/`, `.cache`, `.env`. The
plan adds `scripts/site.env`, `containers/`, `data/h5/`, and
`data/annotations/` so staged artifacts stay untracked.

## Cluster and runtime detection

Keep the existing hostname/prompt detection and the `RUNNING_MODE` values
(`APPTAINER`, `VENV`, `SHELL`), including the `TAMIA` branch. The change is that
detection selects *behaviour* only; **paths** come from the resolver. The same
script therefore runs on Trillium, Tamia, Nibi, Rorqual, Killarney, or a
workstation.

Offline behaviour is preserved verbatim from the Trillium job:
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`,
`WANDB_MODE=offline`, `DISABLE_VERSION_CHECK=1`, `FORCE_TORCHRUN=1`.

## Error handling

`preflight_check()` resolves every path, prints a pass/fail table, and returns
nonzero on any missing required artifact. The body runs it before touching a GPU,
so a missing snapshot fails in seconds instead of wasting an allocation — which
matters because compute nodes have no network and cannot self-heal.

`PREFLIGHT=1` runs only the check and exits, making it usable on a login node.

Required vs optional: model snapshot, the three annotation sources, YAML,
DeepSpeed JSON, and (in `APPTAINER` mode) SIF are required. The three H5 roots
are required for training but reported individually so a partial stage is
diagnosable. Overlay is optional.

## Testing

No GPU is available in the authoring environment, so verification is staged:

1. `bash -n` on all new scripts, plus `shellcheck` if available.
2. `PREFLIGHT=1` on a login node: prints the resolved table and exits nonzero
   when artifacts are absent.
3. **Relocation test** (the real regression guard): copy the repo to a
   differently-named temporary path, run `PREFLIGHT=1` in both, and assert every
   resolved path differs only by the root prefix. This is what proves the
   `$PWD`-name-matching bug is gone.
4. `RUNNING_MODE=SHELL` to enter the container and confirm binds and
   `PYTHONPATH` resolve to this tree.
5. A `max_samples` smoke run before submitting the full job.

## Risks

- `sysconfigtool.py` is shared. The `${VAR}` expansion must be additive: existing
  sections contain no `${` tokens, so their behaviour is unchanged. This will be
  asserted by a relocation-test assertion on an existing cluster key.
- `make license` is **already failing** on `scripts/assign_question_ids.py`
  (pre-existing, unrelated). The plan adds no new `.py` files. Since
  `scripts/sysconfigtool.py` is being edited and currently has no license header,
  adding one is a cheap optional improvement, not a requirement.
- Repo-relative defaults could tempt someone to copy hundreds of GB into the
  checkout. The staging step uses symlinks and the docs must say so explicitly.
