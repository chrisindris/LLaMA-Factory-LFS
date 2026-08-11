#!/usr/bin/env bash
# Scratchpad / helpers for summarizing Killarney CoT SFT .out logs.
# Primary work is done by the embedded Python extractor.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/out}"
# Prefer aip-wangcs path if present (matches log YAML_FILE paths); fall back to this tree.
if [[ -d /project/aip-wangcs/indrisch/LLaMA-Factory/models/qwen2_5vl_lora_sft_CoT/out ]]; then
  OUT_DIR="${OUT_DIR_OVERRIDE:-/project/aip-wangcs/indrisch/LLaMA-Factory/models/qwen2_5vl_lora_sft_CoT/out}"
fi
JOB_ID_MIN="${JOB_ID_MIN:-4667851}"   # keep files with trailing job id STRICTLY greater than this
DATE_MIN="${DATE_MIN:-2026-08-08}"    # mtime on or after this date (local)
TSV_OUT="${TSV_OUT:-${OUT_DIR}/run_summary_jobid_gt_${JOB_ID_MIN}.tsv}"

list_target_logs() {
  python3 - "$OUT_DIR" "$JOB_ID_MIN" "$DATE_MIN" <<'PY'
import os, re, sys
from datetime import datetime
from pathlib import Path

out_dir = Path(sys.argv[1])
job_min = int(sys.argv[2])
date_min = datetime.strptime(sys.argv[3], "%Y-%m-%d").timestamp()

for p in sorted(out_dir.glob("kn*.out")):
    nums = re.findall(r"\d+", p.name)
    if not nums:
        continue
    jid = int(nums[-1])
    if jid <= job_min:
        continue
    if p.stat().st_mtime < date_min:
        continue
    print(p)
PY
}

run_table() {
  python3 - "$OUT_DIR" "$JOB_ID_MIN" "$DATE_MIN" "$TSV_OUT" <<'PY'
#!/usr/bin/env python3
"""Extract config + free_gb + outcome summary from Killarney kn*.out logs."""
from __future__ import annotations

import collections
import csv
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

OUT_DIR = Path(sys.argv[1])
JOB_MIN = int(sys.argv[2])
DATE_MIN = datetime.strptime(sys.argv[3], "%Y-%m-%d").timestamp()
TSV_OUT = Path(sys.argv[4])

# --- regexes ---
YAML_RE = re.compile(r"^YAML_FILE:\s*(\S+)")
FREE_GB_RE = re.compile(r"free_gb=([0-9]+(?:\.[0-9]+)?)")
# Direct config keys in log (rare) — preprocessing workers come only from num_proc=
CFG_PATTERNS = {
    "image_sample_count": re.compile(
        r"(?:image_sample_count|IMAGE_SAMPLE_COUNT)\s*[:=]\s*(-?\d+)", re.I
    ),
    "cutoff_len": re.compile(r"(?:cutoff_len|CUTOFF_LEN)\s*[:=]\s*(-?\d+)", re.I),
    "dataloader_num_workers": re.compile(
        r"(?:dataloader_num_workers|DATALOADER_NUM_WORKERS)\s*[:=]\s*(-?\d+)", re.I
    ),
}
# preprocessing_num_workers := first number after num_proc=
NUM_PROC_RE = re.compile(r"num_proc=(\d+)")
# MAX_JOBS := number immediately after "Using envvar MAX_JOBS ("
MAX_JOBS_RE = re.compile(r"Using envvar MAX_JOBS \((\d+)")
YAML_KEY_RE = {
    "image_sample_count": re.compile(r"^image_sample_count:\s*(-?\d+)"),
    "cutoff_len": re.compile(r"^cutoff_len:\s*(-?\d+)"),
    "dataloader_num_workers": re.compile(r"^dataloader_num_workers:\s*(-?\d+)"),
}

# Outcome markers (checked per line)
TIME_LIMIT_RE = re.compile(r"DUE TO TIME LIMIT|TIME LIMIT \*\*\*", re.I)
CANCEL_SIGNAL_RE = re.compile(r"CANCELLED AT .* DUE to SIGNAL", re.I)
CUDA_OOM_RE = re.compile(
    r"CUDA out of memory|torch\.cuda\.OutOfMemoryError|OutOfMemoryError:\s*CUDA", re.I
)
# Dataloader / shm bus errors (often without elastic "Signal 7")
BUS_SHM_RE = re.compile(
    r"Bus error|insufficient shared memory|workers are out of shared memory|"
    r"Unexpected bus error encountered in worker",
    re.I,
)
SIGBUS_RE = re.compile(r"\bSIGBUS\b|Signal 7\b", re.I)
SIGKILL_RE = re.compile(r"\bSIGKILL\b|Signal 9\b", re.I)
# Real DeepSpeed extension build failures (not routine "Emitting ninja build file")
COMPILE_FAIL_RE = re.compile(
    r"target specific option mismatch|Error building extension|"
    r"cpu_adam\.so: cannot open|Command \['ninja'",
    re.I,
)
MAP_DIED_RE = re.compile(r"abruptly died during map operation", re.I)
IMPORT_ERR_RE = re.compile(r"ImportError: cannot import name", re.I)
CHILD_FAIL_RE = re.compile(r"ChildFailedError|CalledProcessError", re.I)
SRUN_FAIL_RE = re.compile(r"srun: error:.*Exited with exit code", re.I)
SUCCESS_RE = re.compile(
    r"\*{3,}\s*train metrics\s*\*{3,}|Training completed|train_runtime\s*=", re.I
)
TOKENIZE_RE = re.compile(r"Running tokenizer on dataset|Converting format of dataset", re.I)


def job_id_from_name(name: str) -> Optional[int]:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def resolve_path(path: str) -> Optional[Path]:
    """Try aip-wangcs and 6110552 project prefixes."""
    candidates = [Path(path)]
    if "/project/aip-wangcs/" in path:
        candidates.append(Path(path.replace("/project/aip-wangcs/", "/project/6110552/", 1)))
    if "/project/6110552/" in path:
        candidates.append(Path(path.replace("/project/6110552/", "/project/aip-wangcs/", 1)))
    for c in candidates:
        if c.is_file():
            return c
    return None


def read_yaml_config(yaml_path: Optional[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "image_sample_count": None,
        "cutoff_len": None,
        "dataloader_num_workers": None,
    }
    if not yaml_path:
        return out
    p = resolve_path(yaml_path)
    if p is None:
        return out
    try:
        with p.open("r", errors="replace") as f:
            for line in f:
                s = line.strip()
                for key, rx in YAML_KEY_RE.items():
                    m = rx.match(s)
                    if m:
                        out[key] = int(m.group(1))
    except OSError:
        pass
    return out


def typical_free_gb(values: list[float]) -> str:
    if not values:
        return "N/A"
    rounded = [round(v, 3) for v in values]
    mode_val, mode_n = collections.Counter(rounded).most_common(1)[0]
    # Prefer mode; if all unique, fall back to median
    if mode_n == 1 and len(rounded) > 3:
        return f"{statistics.median(values):.3f}"
    return f"{mode_val:.3f}"


def classify_result(flags: dict[str, bool], saw_tokenize: bool) -> str:
    if flags["success"]:
        return "SUCCESS"
    if flags["time_limit"]:
        return "TIME_LIMIT"
    if flags["cancel_signal"]:
        if saw_tokenize and not flags["child_failed"]:
            return "CANCELLED_SIGNAL (during tokenization)"
        return "CANCELLED_SIGNAL"
    if flags["cuda_oom"]:
        return "CUDA_OOM"
    # Bus / shared-memory failures (dataloader workers or elastic SIGBUS)
    if flags["bus_shm"] or flags["sigbus"]:
        if flags["bus_shm"]:
            return "CRASH (SIGBUS/shm — dataloader workers)"
        return "CRASH (SIGBUS)"
    if flags["sigkill"]:
        return "CRASH (SIGKILL)"
    if flags["compile_fail"]:
        return "CRASH (DeepSpeed cpu_adam compile)"
    if flags["import_err"]:
        return "CRASH (ImportError)"
    if flags["map_died"]:
        return "CRASH (HF map worker died)"
    if flags["child_failed"] or flags["srun_fail"]:
        return "CRASH"
    return "INCOMPLETE/UNKNOWN"


def parse_log(path: Path) -> dict[str, Any]:
    yaml_file: Optional[str] = None
    free_vals: list[float] = []
    cfg: dict[str, Any] = {
        "image_sample_count": None,
        "cutoff_len": None,
        "dataloader_num_workers": None,
    }
    cfg_source = {k: None for k in cfg}
    num_proc: Optional[int] = None  # -> preprocessing_num_workers column
    max_jobs: Optional[int] = None  # -> MAX_JOBS column
    flags = {
        "time_limit": False,
        "cancel_signal": False,
        "cuda_oom": False,
        "bus_shm": False,
        "sigbus": False,
        "sigkill": False,
        "compile_fail": False,
        "import_err": False,
        "map_died": False,
        "child_failed": False,
        "srun_fail": False,
        "success": False,
    }
    saw_tokenize = False

    with path.open("r", errors="replace") as f:
        for line in f:
            if yaml_file is None:
                m = YAML_RE.match(line)
                if m:
                    yaml_file = m.group(1)

            for m in FREE_GB_RE.finditer(line):
                free_vals.append(float(m.group(1)))

            for key, rx in CFG_PATTERNS.items():
                if cfg[key] is None:
                    m = rx.search(line)
                    if m:
                        cfg[key] = int(m.group(1))
                        cfg_source[key] = "log"

            if num_proc is None:
                m = NUM_PROC_RE.search(line)
                if m:
                    num_proc = int(m.group(1))

            if max_jobs is None:
                m = MAX_JOBS_RE.search(line)
                if m:
                    max_jobs = int(m.group(1))

            if not saw_tokenize and TOKENIZE_RE.search(line):
                saw_tokenize = True

            if not flags["time_limit"] and TIME_LIMIT_RE.search(line):
                flags["time_limit"] = True
            if not flags["cancel_signal"] and CANCEL_SIGNAL_RE.search(line):
                flags["cancel_signal"] = True
            if not flags["cuda_oom"] and CUDA_OOM_RE.search(line):
                flags["cuda_oom"] = True
            if not flags["bus_shm"] and BUS_SHM_RE.search(line):
                flags["bus_shm"] = True
            if not flags["sigbus"] and SIGBUS_RE.search(line):
                flags["sigbus"] = True
            if not flags["sigkill"] and SIGKILL_RE.search(line):
                flags["sigkill"] = True
            if not flags["compile_fail"] and COMPILE_FAIL_RE.search(line):
                flags["compile_fail"] = True
            if not flags["import_err"] and IMPORT_ERR_RE.search(line):
                flags["import_err"] = True
            if not flags["map_died"] and MAP_DIED_RE.search(line):
                flags["map_died"] = True
            if not flags["child_failed"] and CHILD_FAIL_RE.search(line):
                flags["child_failed"] = True
            if not flags["srun_fail"] and SRUN_FAIL_RE.search(line):
                flags["srun_fail"] = True
            if not flags["success"] and SUCCESS_RE.search(line):
                flags["success"] = True

    # YAML fallback only for image_sample_count / cutoff_len / dataloader_num_workers
    ycfg = read_yaml_config(yaml_file)
    for key in cfg:
        if cfg[key] is None and ycfg.get(key) is not None:
            cfg[key] = ycfg[key]
            cfg_source[key] = "yaml"

    result = classify_result(flags, saw_tokenize)

    return {
        "filename": path.name,
        "job_id": job_id_from_name(path.name),
        "yaml_file": yaml_file or "",
        "image_sample_count": cfg["image_sample_count"] if cfg["image_sample_count"] is not None else "N/A",
        "cutoff_len": cfg["cutoff_len"] if cfg["cutoff_len"] is not None else "N/A",
        # Log-only: number after num_proc=
        "preprocessing_num_workers": num_proc if num_proc is not None else "N/A",
        # Log-only: number after "Using envvar MAX_JOBS ("
        "MAX_JOBS": max_jobs if max_jobs is not None else "N/A",
        "dataloader_num_workers": (
            cfg["dataloader_num_workers"] if cfg["dataloader_num_workers"] is not None else "N/A"
        ),
        "typical_free_gb": typical_free_gb(free_vals),
        "free_gb_n": len(free_vals),
        "overall_result": result,
        "cfg_source": ",".join(
            f"{k}={v}" for k, v in cfg_source.items() if v is not None
        ),
    }


def main() -> None:
    rows: list[dict[str, Any]] = []
    for p in sorted(OUT_DIR.glob("kn*.out")):
        jid = job_id_from_name(p.name)
        if jid is None or jid <= JOB_MIN:
            continue
        if p.stat().st_mtime < DATE_MIN:
            continue
        rows.append(parse_log(p))

    rows.sort(key=lambda r: (r["job_id"] or 0, r["filename"]))

    fieldnames = [
        "filename",
        "image_sample_count",
        "cutoff_len",
        "preprocessing_num_workers",
        "MAX_JOBS",
        "dataloader_num_workers",
        "typical_free_gb",
        "overall_result",
        "yaml_file",
        "job_id",
        "free_gb_n",
        "cfg_source",
    ]
    TSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Markdown to stdout
    md_cols = [
        "filename",
        "image_sample_count",
        "cutoff_len",
        "preprocessing_num_workers",
        "MAX_JOBS",
        "dataloader_num_workers",
        "typical_free_gb",
        "overall_result",
    ]
    print(f"# rows: {len(rows)}")
    print(f"# tsv: {TSV_OUT}")
    print()
    print("| " + " | ".join(md_cols) + " |")
    print("| " + " | ".join(["---"] * len(md_cols)) + " |")
    for r in rows:
        cells = []
        for c in md_cols:
            val = str(r[c])
            # keep table compact
            if c == "filename":
                val = val  # full name
            cells.append(val.replace("|", "\\|"))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
PY
}

usage() {
  cat <<EOF
Usage: $0 [list|run|help]

  list   Print matching kn*.out paths (job_id > ${JOB_ID_MIN}, mtime >= ${DATE_MIN})
  run    Parse logs, write TSV to ${TSV_OUT}, print markdown table
  help   This message

Env overrides: OUT_DIR, JOB_ID_MIN, DATE_MIN, TSV_OUT, OUT_DIR_OVERRIDE
EOF
}

case "${1:-run}" in
  list) list_target_logs ;;
  run)  run_table ;;
  help|-h|--help) usage ;;
  *) echo "Unknown command: $1" >&2; usage; exit 1 ;;
esac
