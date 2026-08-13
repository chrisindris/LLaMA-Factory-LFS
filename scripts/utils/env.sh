#!/usr/bin/env bash

get_project_dir() {
	if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
		export PROJECT_DIR="${PWD%%LLaMA-Factory-LFS*}/LLaMA-Factory-LFS"
	elif [[ "$PWD" == *LLaMA-Factory-copy* ]]; then
		export PROJECT_DIR="${PWD%%LLaMA-Factory-copy*}/LLaMA-Factory-copy"
	elif [[ "$PWD" == *LLaMA-Factory* ]]; then
		export PROJECT_DIR="${PWD%%LLaMA-Factory*}/LLaMA-Factory"
	else
		echo "Error: Could not find 'LLaMA-Factory' or 'LLaMA-Factory-LFS' or 'LLaMA-Factory-copy' in the current path."
		exit 1
	fi
	export SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
	export UTILS_DIR_PATH="$PROJECT_DIR/scripts/utils"
	export PYTHONPATH="$PYTHONPATH:$SYSCONFIG_DIR_PATH"
	export PYTHONPATH="$PYTHONPATH:$UTILS_DIR_PATH"
	export WANDB_DIR="${PROJECT_DIR}/wandb/"
	echo "PROJECT_DIR: $PROJECT_DIR"
	echo "SYSCONFIG_DIR_PATH: $SYSCONFIG_DIR_PATH"
	echo "UTILS_DIR_PATH: $UTILS_DIR_PATH"
	echo "PYTHONPATH: $PYTHONPATH"
	echo "WANDB_DIR: $WANDB_DIR"
}

get_cluster_settings() {

	# if the variables are set already, make sure they are in upper case!
	export CLUSTER="${CLUSTER^^}"
	export RUNNING_MODE="${RUNNING_MODE^^}"

	# Detect cluster based on terminal prompt or hostname
	if [[ "${PS1:-}" == *"rorqual"* ]] || [[ "$HOSTNAME" == *"rorqual"* ]] || [[ "${PS1:-}" == *"rg"* ]] || [[ "$HOSTNAME" == *"rg"* ]] || [[ "${PS1:-}" == *"rc"* ]] || [[ "$HOSTNAME" == *"rc"* ]]; then
		export CLUSTER="${CLUSTER:-RORQUAL}"
		export RUNNING_MODE="${RUNNING_MODE:-APPTAINER}"
	elif [[ "${PS1:-}" == *"trig"* ]] || [[ "$HOSTNAME" == *"trig"* ]] || [[ "${PS1:-}" == *"tri"* ]] || [[ "$HOSTNAME" == *"tri"* ]]; then
		export CLUSTER="${CLUSTER:-TRILLIUM}"
		export RUNNING_MODE="${RUNNING_MODE:-APPTAINER}"
	elif [[ "${PS1:-}" == *"klogin"* ]] || [[ "$HOSTNAME" == *"klogin"* ]] || [[ "${PS1:-}" == *"kn"* ]] || [[ "$HOSTNAME" == *"kn"* ]]; then
		export CLUSTER="${CLUSTER:-KILLARNEY}"
		export RUNNING_MODE="${RUNNING_MODE:-APPTAINER}"
	elif [[ "$HOSTNAME" == *"nibi"* ]] || [[ "${PS1:-}" == *"nibi"* ]] || [[ "${PS1:-}" == *"g"* ]] || [[ "$HOSTNAME" == *"g"* ]] || [[ "${PS1:-}" == *"c"* ]] || [[ "$HOSTNAME" == *"c"* ]]; then
		export CLUSTER="${CLUSTER:-NIBI}"
		export RUNNING_MODE="${RUNNING_MODE:-APPTAINER}"
	else
		echo "Warning: Could not detect cluster from PS1 or HOSTNAME. Defaulting to NIBI."
		export CLUSTER="${CLUSTER:-NIBI}"
		export RUNNING_MODE="${RUNNING_MODE:-APPTAINER}"
	fi

	if [[ "$RUNNING_MODE" == "SHELL" ]]; then
		export SLURM_TMPDIR="/tmp"
	fi

	# Arch list: L40S is Ada (8.9). Do not trust BEST_GPU=h100 on Killarney sysconfig for L40S jobs.
	if [[ "$CLUSTER" == "KILLARNEY" ]]; then
		export TORCH_CUDA_ARCH_LIST="8.9"
	elif [[ "$BEST_GPU" == "h100" ]]; then
		export TORCH_CUDA_ARCH_LIST="9.0"
	else
		export TORCH_CUDA_ARCH_LIST="8.0"
	fi

	echo "CLUSTER: $CLUSTER"
	echo "RUNNING_MODE: $RUNNING_MODE"
	echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
}

get_sysconfig_settings() {
	# Export every key/value from sysconfig.json for the current CLUSTER.
	# Keys are uppercased so e.g. media_dir becomes MEDIA_DIR.
	eval "$(python3 -c "
import shlex
import sysconfigtool
for key, value in sysconfigtool.read_all('${CLUSTER}').items():
    env_key = key.upper()
    print(f'export {env_key}={shlex.quote(str(value))}')
    print(f'echo {shlex.quote(env_key + \": \" + str(value))}')
")"
}

get_project_dir

get_cluster_settings

get_sysconfig_settings
