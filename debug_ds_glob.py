import os
import glob

load_dir = "/project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval_native8gpu/checkpoint-233"
tag = "global_step233"

ckpt_file_pattern_model = os.path.join(load_dir, tag, "*_model_states.pt")
ckpt_files_model = glob.glob(ckpt_file_pattern_model)
print(f"Model files found: {len(ckpt_files_model)}")

ckpt_file_pattern_optim = os.path.join(load_dir, tag, "*_optim_states.pt")
ckpt_files_optim = glob.glob(ckpt_file_pattern_optim)
print(f"Optim files found: {len(ckpt_files_optim)}")

all_files = ckpt_files_model + ckpt_files_optim
print(f"Total files: {len(all_files)}")
for f in all_files[:5]:
    print(f)

