# LLaMA Factory (LFS fork)

## Overview

Unified LLM/VLM fine-tuning toolkit (`llamafactory` 0.9.4.dev0): YAML/CLI training, Gradio LlamaBoard, OpenAI-style API, and export. This fork of [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) adds AllianceCan (Compute Canada) SLURM jobs, H5-backed 3D/VL datasets, and eval prediction dumps. Package code lives under `src/llamafactory`; install editable from repo root.

## Tech Stack

- Python ≥3.9 (CI: 3.9–3.12). Core: PyTorch ≥2.0, transformers (see `requirements.txt`; avoid 4.52.0), datasets, accelerate, peft, trl, Gradio, tyro, omegaconf. `numpy<2.0.0`.
- Optional extras in `setup.py`: `torch`, `deepspeed`, `vllm`, `sglang`, `bitsandbytes`, `liger-kernel`, `dev` (`pre-commit`, `ruff`, `pytest`, `build`), etc.
- Cluster: AllianceCan SLURM (login vs compute). Docker: `docker/docker-cuda|rocm|npu`. Logging: WandB / SwanLab / TensorBoard / MLflow (examples).

## Key Concepts & Terminology

- **stage**: `pt` | `sft` | `rm` | `ppo` | `dpo` | `kto` (not `eval`).
- **finetuning_type**: `lora` | `oft` | `freeze` | `full`.
- **template**: chat/VL formatting (`qwen2_vl`, `llama3`, …); must match the base model.
- **dataset_info.json**: registry in `dataset_dir` (default `./data`); custom data must be registered before use.
- **FORCE_TORCHRUN**: multi-GPU / DeepSpeed launch via `torchrun` (`NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`).
- **H5 image store**: lazy decode via `src/llamafactory/data/data_packing/h5_image_store.py`; env `SCANNET_H5_DIR`, `SPATIALSSRL_H5_DIR`, `THINKER10K_H5_DIR`.
- **USE_V1**: `llamafactory-cli` loads `src/llamafactory/v1` instead of the default launcher.

## Environment & Dependencies

- Local/dev: `pip install -e ".[torch,metrics]"` (README) or `pip install -e ".[dev]"` (CONTRIBUTING). CI uses `".[torch,dev]"`. Optional: `uv sync --extra torch --extra metrics --prerelease=allow`.
- Hugging Face login may be required for gated datasets (`huggingface-cli login`).
- **AllianceCan login node**: no GPU, no `$SLURM_TMPDIR`. **Compute nodes**: no internet — stage models/data on `/scratch` or `/project`; do not `pip install` from PyPI on compute. GPU jobs belong in SLURM, not on the login node.
- Shared venv paths used in this fork (cluster-specific): `/scratch/indrisch/venv_llamafactory_cu126` and `..._cu126_qwen35`; Killarney often `/project/aip-wangcs/indrisch/venv_llamafactory_cu126_qwen35`. Typical modules: `StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.12 cuda/12.6 opencv/4.12.0 arrow`. Details: `.agents/rules/alliancecan-*.md`, `.github/copilot-instructions.md`.
- HF cache / tokens: CI sets `HF_TOKEN`; jobs may use `DISABLE_VERSION_CHECK=1`.

## Commands

Verified from `Makefile`, `setup.py`, CI, README, `launcher.py`:

```bash
pip install -e ".[torch,dev]"          # CONTRIBUTING / CI
make commit                            # pre-commit install + run --all-files
make style                             # ruff check --fix + ruff format (scripts src tests tests_v1 setup.py)
make quality                           # ruff check + ruff format --check
make license                           # tests/check_license.py on those dirs
make test                              # CUDA_VISIBLE_DEVICES= WANDB_DISABLED=true pytest -vv tests/
make build                             # python -m build
llamafactory-cli train <yaml>          # train or eval-only (see Guardrails)
FORCE_TORCHRUN=1 llamafactory-cli train <yaml>
llamafactory-cli chat|export|webui|api|env|version
# lmf is an alias when ENABLE_SHORT_CONSOLE is not disabled
```

NPU device selection: `ASCEND_RT_VISIBLE_DEVICES` (not CUDA). DeepSpeed YAML must be launched with `FORCE_TORCHRUN=1` (`hparams/parser.py`).

## Project Layout

- `src/llamafactory/` — library: `hparams/`, `data/`, `model/`, `train/` (per-stage workflows), `chat/`, `webui/`, `api/`, `eval/` (legacy templates), `v1/` (opt-in), `launcher.py`.
- `src/train.py` — `run_exp()` entry for torchrun/accelerate.
- `examples/` — YAML/scripts (LoRA/QLoRA/full, DeepSpeed JSON, accelerate FSDP, inference, merge). Fork cluster YAMLs: `examples/train_lora/{killarney,nibi,rorqual,trillium}_*.yaml`.
- `data/` — `dataset_info.json`, demos, Scene30k / Spatial-SSRL / 3DThinker-10K.
- `models/`, `experiments/`, `preliminaries/` — SLURM jobs, logs (`*.out`), checkpoints; treat as experiment artifacts.
- `tests/` — pytest suite run by `make test`. `tests_v1/` — v1 unit tests (linted, not in `make test`).
- `scripts/` — helpers (`assign_question_ids.py`, merge/export). `docker/`. `.grok/skills/` — train/eval/distributed agent skills.

## Code Style & Patterns

- Google Python Style; Ruff line length 119, double quotes, isort `llamafactory` first-party (`pyproject.toml`).
- New `.py` under `scripts/`, `src/`, `tests/`, `tests_v1/`: first line must include `Copyright`, `2025`, `LlamaFactory` (`tests/check_license.py`).
- Training configs are YAML; CLI flags override. Prefer copying a nearest `examples/` or `models/**` YAML over inventing flags.
- Datasets: alpaca or sharegpt JSON/JSONL/CSV/parquet/arrow; register in `dataset_info.json`.

## Making Changes

* Make minimal, focused changes; avoid broad refactors unless requested.
* Preserve existing architecture and patterns.
* Don't introduce new dependencies without justification.
* Update tests when behavior changes; update docs when user-visible behavior, configuration, or workflows change.

Upstream contribution path is fork + PR (`.github/CONTRIBUTING.md`). Pre-commit `no-commit-to-branch` blocks commits on `main`. CI on `pull_request`/`push` to `main` for `**/*.py` and requirements: `make style && make quality`, `make license`, `make build`, `make test`.

## Guardrails

### Always

- Register custom datasets in `data/dataset_info.json` (or the configured `dataset_dir`).
- Match `template` (and VL `media_dir` / pixel/fps caps) to the checkpoint.
- For eval: `llamafactory-cli train` with `do_train: false` and `do_eval` and/or `do_predict`; supply `eval_dataset` or `val_size > 0`. `predict_with_generate` requires `eval_dataset` and is incompatible with DeepSpeed ZeRO-3.
- Multi-GPU: `FORCE_TORCHRUN=1`. Multi-node: set `NNODES` / `NODE_RANK` / `MASTER_*` on every rank.
- On AllianceCan: run GPU work in SLURM; copy venv or use Apptainer on compute; stage weights/data before submit.

### Never

- Do not run `llamafactory-cli eval` — it always raises `NotImplementedError` (`launcher.py`). Ignore docs/YAMLs that still use that subcommand (`examples/train_lora/llama3_lora_eval.yaml` is the dead MMLU-style file).
- Do not commit to `main` (pre-commit hook).
- Do not use GPTQ/AWQ with FSDP+QLoRA (`examples/extras/fsdp_qlora` / distributed skill).
- Do not pick DeepSpeed GPUs with `CUDA_VISIBLE_DEVICES` when using the `deepspeed` CLI (use `--include`); this fork prefers torchrun on AllianceCan.

### Use Extra Caution

- `data/dataset_info.json`, DeepSpeed JSON under `examples/deepspeed/`, accelerate YAML under `examples/accelerate/`.
- Cluster venv paths and SLURM scripts under `models/` / `experiments/` (machine-specific).
- Lockfiles / `requirements*.txt`; generated `models/**/out/` and `*.out` logs; H5 stores (do not unpack JPEG trees for Spatial-SSRL / 3DThinker unless the spec says so).
- Secrets: HF/MS/OM hub tokens, WandB keys.

### Deprecated

- `llamafactory-cli eval` and MMLU `task` / `save_dir` eval schema.

## Troubleshooting

- Dataset missing: `dataset_dir` + `dataset_info.json` name vs YAML `dataset`.
- `Please specify dataset for evaluation`: set `eval_dataset` or `val_size > 0`.
- OOM: lower `per_device_*_batch_size`, raise `gradient_accumulation_steps`, shrink `cutoff_len` / VL pixels; see `models/**/cursor_adjust_variables_to_prevent_oom.md`.
- Wrong generations: bad `template` / H5 env not set. CoT mix does not need a single `media_dir` tree — `h5_image_store` routes by path (`data/README.md`).
- `USE_MCA` forces `FORCE_TORCHRUN=1`.

Train/eval/distributed details: `.grok/skills/llamafactory-cli-train`, `llamafactory-cli-eval`, `alliancecan/llamafactory-distributed`.

## Agent Notes

This file is symlinked to CLAUDE.md and GEMINI.md; keep all instructions tool-neutral.
