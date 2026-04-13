#!/bin/bash

# Read-only job monitor for offline W&B runs.
# Default: login-node tmux dashboard over shared scratch logs.
# Optional: srun attach telemetry for an active batch job allocation.

WATCH_INTERVAL=15
GPU_INTERVAL=30
USE_RG=1
UNSAFE_INTERVALS=0
PRINT_COMMANDS=0

RUN_DIR=""
RUN_ID=""
OUT_FILE=""
JOB_ID=""
SESSION_NAME=""

ATTACH_GPU=0
ATTACH_NVIDIA_SMI=0
ATTACH_HTOP=0
ALLOW_MULTIPANE_ATTACH=0
CHECK_OVERHEAD=0

PROJECT_DIR=""
WANDB_ROOT=""
DEBUG_LOG=""
INTERNAL_LOG=""
RUN_FILE=""
LOW_PRIO_PREFIX=""

usage() {
  cat <<'EOF'
Usage:
  monitor_job.sh [options]

Default mode (login-node tmux dashboard, read-only):
  monitor_job.sh [--run-dir <PATH> | --run-id <ID>] [--out-file <PATH>] [--session-name <NAME>]

Attach mode (optional telemetry on job allocation):
  monitor_job.sh --job-id <JOBID> [--attach-gpu | --attach-nvidia-smi | --attach-htop]
  monitor_job.sh --job-id <JOBID> --allow-multipane-attach --attach-nvidia-smi --attach-htop

Options:
  --job-id <JOBID>                Slurm job id (required for attach modes)
  --run-dir <PATH>                Full path to offline-run directory
  --run-id <ID>                   Run id suffix (e.g., n9a3kkgw) or full offline-run-* name
  --out-file <PATH>               Training output file for metrics pane
  --session-name <NAME>           tmux session name
  --watch-interval <SEC>          Heartbeat interval seconds (default: 15)
  --gpu-interval <SEC>            GPU telemetry interval seconds (default: 30)
  --no-rg                         Use grep instead of rg for metrics filtering
  --unsafe-intervals              Allow intervals below safe minima (warns)
  --attach-gpu                    Attach mode: run nvtop in allocation
  --attach-nvidia-smi             Attach mode: run watch nvidia-smi in allocation
  --attach-htop                   Attach mode: run htop -u $USER in allocation
  --allow-multipane-attach        Allow multiple attach monitors in one job-side tmux session
  --check-overhead                Print baseline/post telemetry snapshots for attach mode
  --print-commands                Print commands and exit without launching monitors
  -h, --help                      Show this message

Safety:
  - This script is read-only: it never runs wandb sync or changes WANDB mode.
  - Default mode runs on login node to minimize impact on training allocation.
  - Attach mode shares job resources; keep monitor count and refresh rates conservative.
EOF
}

log() {
  echo "[monitor] $*"
}

warn() {
  echo "[monitor][warn] $*" >&2
}

die() {
  echo "[monitor][error] $*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

resolve_project_dir() {
  if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
    PROJECT_DIR="${PWD%%LLaMA-Factory-LFS*}/LLaMA-Factory-LFS"
    return 0
  fi

  if [[ "$PWD" == *LLaMA-Factory* ]]; then
    PROJECT_DIR="${PWD%%LLaMA-Factory*}/LLaMA-Factory"
    return 0
  fi

  local script_root
  script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [[ -d "$script_root" ]] && [[ -f "$script_root/pyproject.toml" ]]; then
    PROJECT_DIR="$script_root"
    return 0
  fi

  die "Could not infer project directory. Run from inside LLaMA-Factory or provide expected layout."
}

resolve_run_dir() {
  WANDB_ROOT="$PROJECT_DIR/wandb/wandb"
  [[ -d "$WANDB_ROOT" ]] || die "W&B root not found: $WANDB_ROOT"

  if [[ -n "$RUN_DIR" ]]; then
    if [[ "$RUN_DIR" != /* ]]; then
      RUN_DIR="$PWD/$RUN_DIR"
    fi
    [[ -d "$RUN_DIR" ]] || die "--run-dir does not exist: $RUN_DIR"
    return 0
  fi

  if [[ -n "$RUN_ID" ]]; then
    if [[ "$RUN_ID" == offline-run-* ]]; then
      RUN_DIR="$WANDB_ROOT/$RUN_ID"
      [[ -d "$RUN_DIR" ]] || die "Run dir for --run-id not found: $RUN_DIR"
      return 0
    fi

    local matches=()
    shopt -s nullglob
    matches=("$WANDB_ROOT"/offline-run-*-${RUN_ID})
    shopt -u nullglob

    if [[ "${#matches[@]}" -eq 1 ]]; then
      RUN_DIR="${matches[0]}"
      return 0
    fi
    if [[ "${#matches[@]}" -gt 1 ]]; then
      die "Multiple run directories matched --run-id ${RUN_ID}. Please provide --run-dir explicitly."
    fi
    die "No run directory matched --run-id ${RUN_ID}."
  fi

  if [[ -e "$WANDB_ROOT/latest-run" ]]; then
    RUN_DIR="$(readlink -f "$WANDB_ROOT/latest-run")"
    [[ -d "$RUN_DIR" ]] || die "latest-run exists but target is invalid: $RUN_DIR"
    return 0
  fi

  die "Could not resolve run directory. Provide --run-dir or --run-id."
}

find_out_file_by_jobid() {
  local job_id="$1"
  local newest
  newest="$(find "$PROJECT_DIR/models" "$PROJECT_DIR/experiments" -type f -name "*-${job_id}.out" -print0 2>/dev/null | xargs -0 -r ls -1t 2>/dev/null | head -n 1)"
  if [[ -n "$newest" ]]; then
    echo "$newest"
    return 0
  fi
  return 1
}

find_recent_out_file() {
  local candidate
  if [[ -d "$PWD/out" ]]; then
    candidate="$(find "$PWD/out" -maxdepth 1 -type f -name "*.out" -print0 2>/dev/null | xargs -0 -r ls -1t 2>/dev/null | head -n 1)"
    if [[ -n "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  fi

  candidate="$(find "$PROJECT_DIR/models" "$PROJECT_DIR/experiments" -type f -path "*/out/*.out" -print0 2>/dev/null | xargs -0 -r ls -1t 2>/dev/null | head -n 1)"
  if [[ -n "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  return 1
}

resolve_out_file() {
  if [[ -n "$OUT_FILE" ]]; then
    if [[ "$OUT_FILE" != /* ]]; then
      OUT_FILE="$PWD/$OUT_FILE"
    fi
    [[ -r "$OUT_FILE" ]] || die "--out-file is not readable: $OUT_FILE"
    return 0
  fi

  if [[ -n "$JOB_ID" ]]; then
    OUT_FILE="$(find_out_file_by_jobid "$JOB_ID")"
    if [[ -n "$OUT_FILE" ]]; then
      return 0
    fi
  fi

  OUT_FILE="$(find_recent_out_file)"
  if [[ -n "$OUT_FILE" ]]; then
    return 0
  fi

  OUT_FILE=""
}

set_low_priority_prefix() {
  if command_exists ionice && command_exists nice; then
    LOW_PRIO_PREFIX="ionice -c3 nice -n 10"
    return
  fi

  if command_exists nice; then
    LOW_PRIO_PREFIX="nice -n 10"
    return
  fi

  LOW_PRIO_PREFIX=""
}

print_overhead_snapshot() {
  local phase="$1"
  if [[ -z "$JOB_ID" ]]; then
    warn "Cannot capture overhead snapshot without --job-id."
    return
  fi

  log "Overhead snapshot (${phase})"
  if command_exists sstat; then
    sstat -j "${JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P 2>/dev/null || true
  elif command_exists sacct; then
    sacct -j "$JOB_ID" --format=JobID,AveCPU,MaxRSS,MaxVMSize,Elapsed,State -P -n 2>/dev/null || true
  else
    warn "Neither sstat nor sacct is available for overhead reporting."
  fi
}

print_overhead_commands() {
  log "Print-only: overhead snapshots would run before and after monitor start."
  printf '%s\n' "sstat -j \"${JOB_ID}.batch\" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P"
  printf '%s\n' "sacct -j \"$JOB_ID\" --format=JobID,AveCPU,MaxRSS,MaxVMSize,Elapsed,State -P -n"
}

validate_login_mode_inputs() {
  resolve_run_dir

  DEBUG_LOG="$RUN_DIR/logs/debug.log"
  INTERNAL_LOG="$RUN_DIR/logs/debug-internal.log"

  [[ -r "$DEBUG_LOG" ]] || die "Missing or unreadable debug log: $DEBUG_LOG"
  [[ -r "$INTERNAL_LOG" ]] || die "Missing or unreadable internal log: $INTERNAL_LOG"

  RUN_FILE="$(ls -1t "$RUN_DIR"/run-*.wandb 2>/dev/null | head -n 1)"
  [[ -n "$RUN_FILE" ]] || die "No run-*.wandb file found under: $RUN_DIR"
  [[ -r "$RUN_FILE" ]] || die "Run file is not readable: $RUN_FILE"

  resolve_out_file
}

print_preflight() {
  local mode="$1"
  log "Read-only monitor preflight"
  log "Mode: $mode"
  if [[ "$PRINT_COMMANDS" -eq 1 ]]; then
    log "Print-only mode: enabled"
  fi
  if [[ "$mode" == "login" ]]; then
    log "Run dir: $RUN_DIR"
    log "Debug log: $DEBUG_LOG"
    log "Internal log: $INTERNAL_LOG"
    if [[ -n "$OUT_FILE" ]]; then
      log "Out file: $OUT_FILE"
    else
      warn "Could not resolve output file automatically. Metrics pane will show a guidance message."
    fi
    log "Run heartbeat file: $RUN_FILE"
    log "watch interval: ${WATCH_INTERVAL}s"
  else
    log "Job id: $JOB_ID"
    log "GPU interval: ${GPU_INTERVAL}s"
  fi

  log "Guardrails: no wandb sync, no WANDB mode changes, no socket manipulation."
}

build_metrics_filter_cmd() {
  if [[ "$USE_RG" -eq 1 ]]; then
    if command_exists rg; then
      echo "rg --line-buffered 'loss|grad_norm|learning_rate|epoch|eval'"
      return
    fi
    warn "rg not found; falling back to grep."
  fi
  echo "grep --line-buffered -E 'loss|grad_norm|learning_rate|epoch|eval'"
}

launch_login_tmux_dashboard() {
  set_low_priority_prefix

  local metrics_filter
  metrics_filter="$(build_metrics_filter_cmd)"

  local run_label
  run_label="$(basename "$RUN_DIR")"
  if [[ -z "$SESSION_NAME" ]]; then
    SESSION_NAME="monitor-${run_label##*-}"
  fi

  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    warn "tmux session already exists: $SESSION_NAME. Attaching to existing session."
    tmux attach -t "$SESSION_NAME"
    return
  fi

  local cmd_debug cmd_internal cmd_metrics cmd_heartbeat
  cmd_debug="$LOW_PRIO_PREFIX tail -n 0 -F '$DEBUG_LOG'"
  cmd_internal="$LOW_PRIO_PREFIX tail -n 0 -F '$INTERNAL_LOG'"

  if [[ -n "$OUT_FILE" ]]; then
    cmd_metrics="$LOW_PRIO_PREFIX tail -n 0 -F '$OUT_FILE' | $metrics_filter"
  else
    cmd_metrics="echo 'No output file resolved. Re-run with --out-file <PATH>.'; tail -f /dev/null"
  fi

  cmd_heartbeat="$LOW_PRIO_PREFIX watch -n ${WATCH_INTERVAL} \"stat -c '%y %s bytes' '$RUN_FILE'\""

  if [[ "$PRINT_COMMANDS" -eq 1 ]]; then
    log "Print-only mode enabled. No monitor process will be started."
    local q_cmd_debug q_cmd_internal q_cmd_metrics q_cmd_heartbeat
    printf -v q_cmd_debug '%q' "$cmd_debug"
    printf -v q_cmd_internal '%q' "$cmd_internal"
    printf -v q_cmd_metrics '%q' "$cmd_metrics"
    printf -v q_cmd_heartbeat '%q' "$cmd_heartbeat"

    printf '%s\n' "tmux new-session -d -s \"$SESSION_NAME\" $q_cmd_debug"
    printf '%s\n' "tmux split-window -h -t \"$SESSION_NAME:0\" $q_cmd_internal"
    printf '%s\n' "tmux split-window -v -t \"$SESSION_NAME:0.0\" $q_cmd_metrics"
    printf '%s\n' "tmux split-window -v -t \"$SESSION_NAME:0.1\" $q_cmd_heartbeat"
    printf '%s\n' "tmux select-layout -t \"$SESSION_NAME:0\" tiled"
    printf '%s\n' "tmux set-option -t \"$SESSION_NAME:0\" remain-on-exit on"
    printf '%s\n' "tmux attach -t \"$SESSION_NAME\""
    return
  fi

  command_exists tmux || die "tmux is required for default mode."

  tmux new-session -d -s "$SESSION_NAME" "$cmd_debug"
  tmux split-window -h -t "$SESSION_NAME:0" "$cmd_internal"
  tmux split-window -v -t "$SESSION_NAME:0.0" "$cmd_metrics"
  tmux split-window -v -t "$SESSION_NAME:0.1" "$cmd_heartbeat"
  tmux select-layout -t "$SESSION_NAME:0" tiled
  tmux set-option -t "$SESSION_NAME:0" remain-on-exit on >/dev/null

  log "tmux session created: $SESSION_NAME"
  log "Attach command: tmux attach -t $SESSION_NAME"
  tmux attach -t "$SESSION_NAME"
}

launch_attach_mode() {
  local attach_count=0
  ((attach_count += ATTACH_GPU))
  ((attach_count += ATTACH_NVIDIA_SMI))
  ((attach_count += ATTACH_HTOP))

  [[ -n "$JOB_ID" ]] || die "--job-id is required for attach modes."
  [[ "$attach_count" -gt 0 ]] || die "Attach mode selected but no attach monitor flag provided."

  if [[ "$attach_count" -gt 1 ]] && [[ "$ALLOW_MULTIPANE_ATTACH" -eq 0 ]]; then
    die "Multiple attach monitors requested. Add --allow-multipane-attach to proceed."
  fi

  local single_attach_cmd=""
  if [[ "$attach_count" -eq 1 ]]; then
    if [[ "$ATTACH_GPU" -eq 1 ]]; then
      single_attach_cmd="srun --pty --overlap --jobid \"$JOB_ID\" nvtop"
    elif [[ "$ATTACH_NVIDIA_SMI" -eq 1 ]]; then
      single_attach_cmd="srun --jobid \"$JOB_ID\" --overlap --pty watch -n $GPU_INTERVAL nvidia-smi"
    else
      single_attach_cmd="srun --jobid \"$JOB_ID\" --overlap --pty htop -u \"$USER\""
    fi
  fi

  if [[ "$PRINT_COMMANDS" -eq 1 ]]; then
    log "Print-only mode enabled. No monitor process will be started."
    if [[ "$CHECK_OVERHEAD" -eq 1 ]]; then
      print_overhead_commands
    fi

    if [[ "$attach_count" -eq 1 ]]; then
      printf '%s\n' "$single_attach_cmd"
      return
    fi

    warn "Multipane attach preview. This mode has higher overhead than single monitor attach."

    local preview_cmds=()
    if [[ "$ATTACH_GPU" -eq 1 ]]; then
      preview_cmds+=("nvtop")
    fi
    if [[ "$ATTACH_NVIDIA_SMI" -eq 1 ]]; then
      preview_cmds+=("watch -n ${GPU_INTERVAL} nvidia-smi")
    fi
    if [[ "$ATTACH_HTOP" -eq 1 ]]; then
      preview_cmds+=("htop -u $USER")
    fi

    local preview
    preview="srun --jobid \"$JOB_ID\" --overlap --pty env -u TMUX tmux new-session -d '${preview_cmds[0]}'"
    local p
    for ((p = 1; p < ${#preview_cmds[@]}; p++)); do
      preview+=" \\; split-window -t 0 '${preview_cmds[$p]}'"
    done
    preview+=" \\; select-layout tiled \\; attach"
    printf '%s\n' "$preview"
    return
  fi

  command_exists srun || die "srun is required for attach modes."

  if [[ "$CHECK_OVERHEAD" -eq 1 ]]; then
    print_overhead_snapshot "baseline"
  fi

  if [[ "$attach_count" -eq 1 ]]; then
    if [[ "$ATTACH_GPU" -eq 1 ]]; then
      warn "Attach monitor runs inside allocation and shares job resources."
      srun --pty --overlap --jobid "$JOB_ID" nvtop
    elif [[ "$ATTACH_NVIDIA_SMI" -eq 1 ]]; then
      warn "Attach monitor runs inside allocation and shares job resources."
      srun --jobid "$JOB_ID" --overlap --pty watch -n "$GPU_INTERVAL" nvidia-smi
    else
      warn "Attach monitor runs inside allocation and shares job resources."
      srun --jobid "$JOB_ID" --overlap --pty htop -u "$USER"
    fi

    if [[ "$CHECK_OVERHEAD" -eq 1 ]]; then
      print_overhead_snapshot "post"
    fi
    return
  fi

  warn "Multipane attach requested. This increases monitoring overhead in allocation."

  local cmds=()
  if [[ "$ATTACH_GPU" -eq 1 ]]; then
    cmds+=("nvtop")
  fi
  if [[ "$ATTACH_NVIDIA_SMI" -eq 1 ]]; then
    cmds+=("watch -n ${GPU_INTERVAL} nvidia-smi")
  fi
  if [[ "$ATTACH_HTOP" -eq 1 ]]; then
    cmds+=("htop -u $USER")
  fi

  local tmux_args=(new-session -d "${cmds[0]}")
  local i
  for ((i = 1; i < ${#cmds[@]}; i++)); do
    tmux_args+=(\; split-window -t 0 "${cmds[$i]}")
  done
  tmux_args+=(\; select-layout tiled \; attach)

  srun --jobid "$JOB_ID" --overlap --pty env -u TMUX tmux "${tmux_args[@]}"

  if [[ "$CHECK_OVERHEAD" -eq 1 ]]; then
    print_overhead_snapshot "post"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id)
      [[ $# -ge 2 ]] || die "--job-id requires a value."
      JOB_ID="$2"
      shift 2
      ;;
    --run-dir)
      [[ $# -ge 2 ]] || die "--run-dir requires a value."
      RUN_DIR="$2"
      shift 2
      ;;
    --run-id)
      [[ $# -ge 2 ]] || die "--run-id requires a value."
      RUN_ID="$2"
      shift 2
      ;;
    --out-file)
      [[ $# -ge 2 ]] || die "--out-file requires a value."
      OUT_FILE="$2"
      shift 2
      ;;
    --session-name)
      [[ $# -ge 2 ]] || die "--session-name requires a value."
      SESSION_NAME="$2"
      shift 2
      ;;
    --watch-interval)
      [[ $# -ge 2 ]] || die "--watch-interval requires a value."
      WATCH_INTERVAL="$2"
      shift 2
      ;;
    --gpu-interval)
      [[ $# -ge 2 ]] || die "--gpu-interval requires a value."
      GPU_INTERVAL="$2"
      shift 2
      ;;
    --no-rg)
      USE_RG=0
      shift
      ;;
    --unsafe-intervals)
      UNSAFE_INTERVALS=1
      shift
      ;;
    --attach-gpu)
      ATTACH_GPU=1
      shift
      ;;
    --attach-nvidia-smi)
      ATTACH_NVIDIA_SMI=1
      shift
      ;;
    --attach-htop)
      ATTACH_HTOP=1
      shift
      ;;
    --allow-multipane-attach)
      ALLOW_MULTIPANE_ATTACH=1
      shift
      ;;
    --check-overhead)
      CHECK_OVERHEAD=1
      shift
      ;;
    --print-commands)
      PRINT_COMMANDS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

is_positive_int "$WATCH_INTERVAL" || die "--watch-interval must be a positive integer."
is_positive_int "$GPU_INTERVAL" || die "--gpu-interval must be a positive integer."

if [[ "$UNSAFE_INTERVALS" -eq 0 ]]; then
  if [[ "$WATCH_INTERVAL" -lt 15 ]]; then
    die "--watch-interval below safe minimum (15s). Use --unsafe-intervals to override."
  fi
  if [[ "$GPU_INTERVAL" -lt 30 ]]; then
    die "--gpu-interval below safe minimum (30s). Use --unsafe-intervals to override."
  fi
else
  warn "Unsafe intervals enabled. Aggressive polling may impact job performance."
fi

resolve_project_dir

attach_count=$((ATTACH_GPU + ATTACH_NVIDIA_SMI + ATTACH_HTOP))

if [[ "$attach_count" -gt 0 ]]; then
  [[ -n "$JOB_ID" ]] || die "--job-id is required for attach modes."
  print_preflight "attach"
  launch_attach_mode
  exit 0
fi

validate_login_mode_inputs
print_preflight "login"
launch_login_tmux_dashboard
