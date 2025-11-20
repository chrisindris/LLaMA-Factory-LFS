# to save outputs:
find . -type f -size +50M ! -name *.dar ! -name "*pt_data_worker*" ! -wholename "*saves*" ! -name *.whl ! -name *.pack -exec sh -c "module load git-lfs/3.4.0 StdEnv/2023 && git-lfs track {} && echo {}" sh {} \;
# then, upload saves to GCP


# 100 examples (per_device_train_batch_size: 2) takes 25 minutes; used 14.49% of CPU memory (108.97 of 751.95 GB), no GPU OOM errors
# 500 examples (per_device_train_batch_size: 2) on a full node (96 CPUs, 4 GPUs, 32 preproc workers and 4 dataloader workers) will take about 3 hours; used 13.14% of CPU memory (98.82 of 751.95 GB) - [TBD]
# 500 examples (per_device_train_batch_size: 2) on a full node (96 CPUs, 4 GPUs, 32 preproc workers and 32 dataloader workers) will take about 3 hours; used 13.14% of CPU memory (98.82 of 751.95 GB)
# -- if we try it with 32 dataloader workers: (JOBID 96615)(queue time: ~2 hours)[Completed, woohoo! (took 1.75 hours)]
# ---- same but 15 minutes instead of 3 hours (check for initial issues): (JOBID 96616)(queue time: <2 hours)[Crashed due to NODE_FAIL i.e. not my fault]
# -- quarter-node job (24 CPUs, 1 GPU): (JOBID 96618)(queue time: 10 minutes)[OUT OF SHARED MEMORY (single-gpu jobs on Trillium give you 188GB mem, which was split among 32 data loaders); need to try with fewer dataloader workers if using 1 GPU]
# ---- same but with 16 workers: [TBD]
# ---- same but 15 minutes instead of 3 hours (check for initial issues): (96617)(queue time: few minutes)[finished tokenization]


[indrisch@trig-login01 qwen2_5vl_lora_sft_SQA3D]$ ( squeue | wc -l ) && ( squeue | grep -n indrisch ) 
316
8:          96617        indrisch     def-wangcs slurm_qwen2_5v   R      14:43           compute     1 gres/gpu:h100:1 trig0062 (None)
254:          96618        indrisch     def-wangcs slurm_qwen2_5v  PD    3:00:00           compute     1 gres/gpu:h100:1  (Priority)
255:          96616        indrisch     def-wangcs slurm_qwen2_5v  PD      15:00 compute_full_node     1 gres/gpu:h100:4  (Priority)
258:          96615        indrisch     def-wangcs slurm_qwen2_5v  PD    3:00:00 compute_full_node     1 gres/gpu:h100:4  (Priority)