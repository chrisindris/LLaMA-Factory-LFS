#!/bin/bash

# RUN ON: Login Node
# RUN AS: ./load_model.sh <cluster_name>

export HF_TOKEN=$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)

# vLLM models typically come from the huggingface hub. 
module load python/3.12 git-lfs/3.4.0 && git-lfs install
virtualenv --no-download temp_env && source temp_env/bin/activate
pip install --no-index huggingface_hub
# --- default location: $HOME/.cache/huggingface/hub; modify with HF_HOME and HFHUB_CACHE. ---
#huggingface-cli download moonshotai/Kimi-VL-A3B-Thinking-2506 # default location: $HOME/.cache/huggingface/hub
#HF_HUB_DISABLE_XET=1 hf download --max-workers=4 moonshotai/Kimi-VL-A3B-Thinking-2506 # using --local-dir and --cache-dir; default location: $HOME/.cache/huggingface/hub

if [[ "$PWD" == *LLaMA-Factory-LFS* ]]; then
    PROJECT_DIR="${PWD%%LLaMA-Factory-LFS*}/LLaMA-Factory-LFS"
elif [[ "$PWD" == *LLaMA-Factory* ]]; then
    PROJECT_DIR="${PWD%%LLaMA-Factory*}/LLaMA-Factory"
else
    echo "Error: Could not find 'LLaMA-Factory' or 'LLaMA-Factory-LFS' in the current path."
    exit 1
fi
SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
export PYTHONPATH="$PYTHONPATH:$SYSCONFIG_DIR_PATH"

echo "PROJECT_DIR: $PROJECT_DIR"
echo "SYSCONFIG_DIR_PATH: $SYSCONFIG_DIR_PATH"
echo "PWD: $PWD"
echo "PYTHONPATH: $PYTHONPATH"

export HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('$1', 'HF_HOME'))")" 
export HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('$1', 'HF_HUB_CACHE'))")" 
export HF_HUB_DISABLE_XET="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('$1', 'HF_HUB_DISABLE_XET'))")" 
unset PYTHONPATH

echo "Cluster name: $1"
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "HF_HUB_DISABLE_XET: $HF_HUB_DISABLE_XET"

# hf download --max-workers=7 moonshotai/Kimi-VL-A3B-Thinking-2506 # model for generating traces (the "teacher")
#hf download --max-workers=5 Qwen/Qwen2.5-VL-7B-Instruct # model we will use as a student (in addition to LLaVa-3D)
# hf download --max-workers=4 Qwen/Qwen2.5-7B-Instruct-1M # text-only long-context model used to assess the traces
#hf download --max-workers=4 Qwen/Qwen2.5-7B-Instruct # text-only model used to assess the traces; the 1M version doesn't seem to load
#hf download --max-workers=4 Qwen/Qwen2.5-0.5B-Instruct # text-only model used to assess the traces; the 1M version doesn't seem to load
#hf download --max-workers=5 Qwen/Qwen3-Reranker-8B # could be useful for reranking the traces
#hf download --max-workers=3 Qwen/Qwen3-4B-Thinking-2507 # Qwen3 (latest qwen) with thinking capabilities; 4B is small enough to fit on one GPU; note that this is also language only.
#hf download --max-workers=8 Qwen/Qwen3-30B-A3B-Thinking-2507 # Same as the line above, but if we can use multiple GPUs, we can use this one to get even better decisions about the best traces.
# hf download --max-workers=4 FacebookAI/roberta-large-mnli # for MNLI, deterimining the X portion of the reranker.
# hf download --max-workers=4 MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli # for MNLI, deterimining the X portion of the reranker.

hf download --max-workers=4 Video-R1/Video-R1-7B # Video-R1 is Qwen2.5VL-7B that was used by Video-R1.
#hf download --max-workers=4 Video-R1/Qwen2.5-VL-7B-COT-SFT # this is the video-R1, but with 1 epoch of SFT training on their dataset. This should work perfectly with LLaMA-Factory out of box.
#hf download --max-workers=4 Qwen/Qwen3-VL-8B-Instruct # The Qwen3 counterpart to Qwen2.5VL-7B, which is what we have been using.
#hf download --max-workers=4 Qwen/Qwen3-VL-8B-Thinking # Same as above but with extra thinking capabilities.
#hf download --max-workers=8 zd11024/Video3D-LLM-LLaVA-Qwen-Uniform-32 # Resulting model of Video3D-LLM

# SPECIAL NOTE: there is currently no Qwen3-VL that is less than 235B, which is way too big for our purposes (training it); even the smallest Qwen3-Omni (also vision-language) is 30GB. Hence, we will still try to enhance Qwen2.5 VL.
# we should try VL and non-VL versions of Qwen3 to assess traces.

hf download --max-workers=12 cvis-tmu/easyr1_verl_sif --repo-type=dataset
hf download --max-workers=4 cvis-tmu/llamafactory-sqa3d-traces-multiimage-vqa_R.12_C.12_F.12_X.62 --repo-type=dataset
hf download --max-workers=4 cvis-tmu/llamafactory-sqa3d-traces-multiimage-vqa --repo-type=dataset

deactivate
rm -r temp_env

# once that's done, you can run sbatch vllm_example.sh