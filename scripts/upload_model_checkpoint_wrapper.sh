#!/bin/bash

# ./upload_model_checkpoint.sh \
#     --id qwen2_5vl-7b-lora-sft-Scene30k_traineval_1278steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/Scene30k_traineval_5epochs/checkpoint-1278/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260406_075716-4qql81z4 \
#     --commit-message "Qwen2.5-VL-7B LoRA adapter checkpoint trained for 1278 steps (3 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id qwen2_5vl-7b-lora-sft-Scene30k_traineval_1704steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/Scene30k_traineval_5epochs/checkpoint-1704/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260406_075716-4qql81z4 \
#     --commit-message "Qwen2.5-VL-7B LoRA adapter checkpoint trained for 1704 steps (4 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id qwen2_5vl-7b-lora-sft-Scene30k_traineval_2130steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft/Scene30k_traineval_5epochs/checkpoint-2130/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260406_075716-4qql81z4 \
#     --commit-message "Qwen2.5-VL-7B LoRA adapter checkpoint trained for 2130 steps (5 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-7b-lora-sft-Scene30k_traineval_426steps \
#     --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/Scene30k_traineval_5epochs/checkpoint-426/ \
#     --wandb-log /project/aip-wangcs/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260414_121240-exbumxo3 \
#     --commit-message "Qwen3-VL-8B LoRA adapter checkpoint trained for 426 steps (1 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id videor1-lora-sft-Scene30k_traineval_852steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/videor1/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260413_074026-4mg8bdpv \
#     --commit-message "VideoR1 LoRA adapter checkpoint trained for 852 steps (2 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id videor1-lora-sft-Scene30k_traineval_852steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/videor1/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260413_074026-4mg8bdpv \
#     --commit-message "VideoR1 LoRA adapter checkpoint trained for 852 steps (2 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-8b-lora-sft-Scene30k_traineval_426steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen3vl/lora/sft/Scene30k_traineval_5epochs/checkpoint-426/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_084057-2ahgooa3 \
#     --commit-message "Qwen3VL LoRA adapter checkpoint trained for 426 steps (1 epoch) total on the Scene30k dataset (corrected)" \
#     --no-wandb-upload

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-8b-lora-sft-Scene30k_traineval_852steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen3vl/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_084057-2ahgooa3 \
#     --commit-message "Qwen3VL LoRA adapter checkpoint trained for 852 steps (2 epoch) total on the Scene30k dataset (corrected)" \
#     --no-wandb-upload

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-8b-lora-sft-Scene30k_traineval_1278steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen3vl/lora/sft/Scene30k_traineval_5epochs/checkpoint-1278/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_084057-2ahgooa3 \
#     --commit-message "Qwen3VL LoRA adapter checkpoint trained for 1278 steps (3 epoch) total on the Scene30k dataset (corrected)" \
#     --no-wandb-upload

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-8b-lora-sft-Scene30k_traineval_1704steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen3vl/lora/sft/Scene30k_traineval_5epochs/checkpoint-1704/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_084057-2ahgooa3 \
#     --commit-message "Qwen3VL LoRA adapter checkpoint trained for 1704 steps (4 epoch) total on the Scene30k dataset (corrected)" \
#     --no-wandb-upload

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-8b-lora-sft-Scene30k_traineval_2130steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen3vl/lora/sft/Scene30k_traineval_5epochs/checkpoint-2130/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_084057-2ahgooa3 \
#     --commit-message "Qwen3VL LoRA adapter checkpoint trained for 2130 steps (5 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id videor1sft-lora-sft-Scene30k_traineval_426steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/videor1sft/lora/sft/Scene30k_traineval_5epochs/checkpoint-426/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_102752-emyhx9u4 \
#     --commit-message "VideoR1 (SFT-only, no RL) LoRA adapter checkpoint trained for 426 steps (1 epoch) total on the Scene30k dataset (corrected)" \
#     --no-wandb-upload

# ./upload_model_checkpoint.sh \
#     --id videor1sft-lora-sft-Scene30k_traineval_852steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/videor1sft/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_102752-emyhx9u4 \
#     --commit-message "VideoR1 (SFT-only, no RL) LoRA adapter checkpoint trained for 852 steps (2 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-7b-lora-sft-Scene30k_traineval_426steps \
#     --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/Scene30k_traineval_5epochs/checkpoint-426/ \
#     --wandb-log /project/aip-wangcs/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260414_121240-exbumxo3 \
#     --commit-message "Qwen3-VL-8B LoRA adapter checkpoint trained for 426 steps (1 epoch) total on the Scene30k dataset (corrected)"


# ./upload_model_checkpoint.sh \
#     --id qwen3vl-7b-lora-sft-Scene30k_traineval_852steps \
#     --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
#     --wandb-log /project/aip-wangcs/indrisch//LLaMA-Factory/wandb/wandb/offline-run-20260414_121240-exbumxo3 \
#     --commit-message "Qwen3-VL-8B LoRA adapter checkpoint trained for 852 steps (2 epoch) total on the Scene30k dataset (corrected)"


# ./upload_model_checkpoint.sh \
#     --id qwen3vl-7b-lora-sft-Scene30k_traineval_1278steps \
#     --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/Scene30k_traineval_5epochs/checkpoint-1278/ \
#     --wandb-log /project/aip-wangcs/indrisch//LLaMA-Factory/wandb/wandb/offline-run-20260414_121240-exbumxo3 \
#     --commit-message "Qwen3-VL-8B LoRA adapter checkpoint trained for 1278 steps (3 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-7b-lora-sft-Scene30k_traineval_852steps \
#     --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
#     --wandb-log /project/aip-wangcs/indrisch//LLaMA-Factory/wandb/wandb/offline-run-20260414_121240-exbumxo3 \
#     --commit-message "Qwen3-VL-8B LoRA adapter checkpoint trained for 852 steps (2 epoch) total on the Scene30k dataset (corrected)"


# ./upload_model_checkpoint.sh \
#     --id qwen3vl-7b-lora-sft-Scene30k_traineval_1278steps \
#     --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/Scene30k_traineval_5epochs/checkpoint-1278/ \
#     --wandb-log /project/aip-wangcs/indrisch//LLaMA-Factory/wandb/wandb/offline-run-20260414_121240-exbumxo3 \
#     --commit-message "Qwen3-VL-8B LoRA adapter checkpoint trained for 1278 steps (3 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id videor1-lora-sft-Scene30k_traineval_426steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/videor1/lora/sft/Scene30k_traineval_5epochs/checkpoint-426/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260413_074026-4mg8bdpv \
#     --commit-message "VideoR1 LoRA adapter checkpoint trained for 426 steps (1 epoch) total on the Scene30k dataset (corrected)" \
#     --no-wandb-upload

# ./upload_model_checkpoint.sh \
#     --id videor1-lora-sft-Scene30k_traineval_852steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/videor1/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260413_074026-4mg8bdpv \
#     --commit-message "VideoR1 LoRA adapter checkpoint trained for 852 steps (2 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id videor1-lora-sft-Scene30k_traineval_852steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/videor1/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260413_074026-4mg8bdpv \
#     --commit-message "VideoR1 LoRA adapter checkpoint trained for 852 steps (2 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-8b-lora-sft-Scene30k_traineval_426steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen3vl/lora/sft/Scene30k_traineval_5epochs/checkpoint-426/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_084057-2ahgooa3 \
#     --commit-message "Qwen3VL LoRA adapter checkpoint trained for 426 steps (1 epoch) total on the Scene30k dataset (corrected)" \
#     --no-wandb-upload

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-8b-lora-sft-Scene30k_traineval_852steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen3vl/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_084057-2ahgooa3 \
#     --commit-message "Qwen3VL LoRA adapter checkpoint trained for 852 steps (2 epoch) total on the Scene30k dataset (corrected)" \
#     --no-wandb-upload

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-8b-lora-sft-Scene30k_traineval_1278steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen3vl/lora/sft/Scene30k_traineval_5epochs/checkpoint-1278/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_084057-2ahgooa3 \
#     --commit-message "Qwen3VL LoRA adapter checkpoint trained for 1278 steps (3 epoch) total on the Scene30k dataset (corrected)" \
#     --no-wandb-upload

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-8b-lora-sft-Scene30k_traineval_1704steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen3vl/lora/sft/Scene30k_traineval_5epochs/checkpoint-1704/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_084057-2ahgooa3 \
#     --commit-message "Qwen3VL LoRA adapter checkpoint trained for 1704 steps (4 epoch) total on the Scene30k dataset (corrected)" \
#     --no-wandb-upload

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-8b-lora-sft-Scene30k_traineval_2130steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/qwen3vl/lora/sft/Scene30k_traineval_5epochs/checkpoint-2130/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_084057-2ahgooa3 \
#     --commit-message "Qwen3VL LoRA adapter checkpoint trained for 2130 steps (5 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id videor1sft-lora-sft-Scene30k_traineval_426steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/videor1sft/lora/sft/Scene30k_traineval_5epochs/checkpoint-426/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_102752-emyhx9u4 \
#     --commit-message "VideoR1 (SFT-only, no RL) LoRA adapter checkpoint trained for 426 steps (1 epoch) total on the Scene30k dataset (corrected)" \
#     --no-wandb-upload

# ./upload_model_checkpoint.sh \
#     --id videor1sft-lora-sft-Scene30k_traineval_852steps \
#     --checkpoint /scratch/indrisch/LLaMA-Factory/saves/videor1sft/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
#     --wandb-log /scratch/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260418_102752-emyhx9u4 \
#     --commit-message "VideoR1 (SFT-only, no RL) LoRA adapter checkpoint trained for 852 steps (2 epoch) total on the Scene30k dataset (corrected)"

# ./upload_model_checkpoint.sh \
#     --id qwen3vl-7b-lora-sft-Scene30k_traineval_426steps \
#     --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/Scene30k_traineval_5epochs/checkpoint-426/ \
#     --wandb-log /project/aip-wangcs/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260414_121240-exbumxo3 \
#     --commit-message "Qwen3-VL-8B LoRA adapter checkpoint trained for 426 steps (1 epoch) total on the Scene30k dataset (corrected)"


# ./upload_model_checkpoint.sh \
#     --id qwen3vl-7b-lora-sft-Scene30k_traineval_852steps \
#     --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
#     --wandb-log /project/aip-wangcs/indrisch//LLaMA-Factory/wandb/wandb/offline-run-20260414_121240-exbumxo3 \
#     --commit-message "Qwen3-VL-8B LoRA adapter checkpoint trained for 852 steps (2 epoch) total on the Scene30k dataset (corrected)"


# ./upload_model_checkpoint.sh \
#     --id qwen3vl-7b-lora-sft-Scene30k_traineval_1278steps \
#     --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/qwen3vl-8b/lora/sft/Scene30k_traineval_5epochs/checkpoint-1278/ \
#     --wandb-log /project/aip-wangcs/indrisch//LLaMA-Factory/wandb/wandb/offline-run-20260414_121240-exbumxo3 \
#     --commit-message "Qwen3-VL-8B LoRA adapter checkpoint trained for 1278 steps (3 epoch) total on the Scene30k dataset (corrected)"

./upload_model_checkpoint.sh \
    --id videor1sft-lora-sft-Scene30k_traineval_2epochs \
    --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/videor1sft/lora/sft/Scene30k_traineval_5epochs/checkpoint-426/ \
    --wandb-log /project/aip-wangcs/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260427_072733-1abe7g77 \
    --commit-message "VideoR1 (SFT-only, no RL) LoRA adapter checkpoint trained for 426 steps on 8 GPUs (2 epoch) total on the Scene30k dataset (corrected)" \
    --no-wandb-upload

./upload_model_checkpoint.sh \
    --id videor1sft-lora-sft-Scene30k_traineval_4epochs \
    --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/videor1sft/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
    --wandb-log /project/aip-wangcs/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260427_072733-1abe7g77 \
    --commit-message "VideoR1 (SFT-only, no RL) LoRA adapter checkpoint trained for 852 steps on 8 GPUs (4 epoch) total on the Scene30k dataset (corrected)" \
    --no-wandb-upload

./upload_model_checkpoint.sh \
    --id videor1sft-lora-sft-Scene30k_traineval_5epochs \
    --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/videor1sft/lora/sft/Scene30k_traineval_5epochs/checkpoint-1065/ \
    --wandb-log /project/aip-wangcs/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260427_072733-1abe7g77 \
    --commit-message "VideoR1 (SFT-only, no RL) LoRA adapter checkpoint trained for 1065 steps on 8 GPUs (5 epoch) total on the Scene30k dataset (corrected)" \

./upload_model_checkpoint.sh \
    --id videor1-lora-sft-Scene30k_traineval_2epochs \
    --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/videor1/lora/sft/Scene30k_traineval_5epochs/checkpoint-426/ \
    --wandb-log /project/aip-wangcs/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260425_075152-5bx1dfj1 \
    --commit-message "VideoR1 LoRA adapter checkpoint trained for 426 steps on 8 GPUs (2 epoch) total on the Scene30k dataset (corrected)" \
    --no-wandb-upload

./upload_model_checkpoint.sh \
    --id videor1-lora-sft-Scene30k_traineval_4epochs \
    --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/videor1/lora/sft/Scene30k_traineval_5epochs/checkpoint-852/ \
    --wandb-log /project/aip-wangcs/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260425_075152-5bx1dfj1 \
    --commit-message "VideoR1 LoRA adapter checkpoint trained for 852 steps on 8 GPUs (4 epoch) total on the Scene30k dataset (corrected)" \
    --no-wandb-upload

./upload_model_checkpoint.sh \
    --id videor1-lora-sft-Scene30k_traineval_5epochs \
    --checkpoint /project/aip-wangcs/indrisch/LLaMA-Factory/saves/videor1/lora/sft/Scene30k_traineval_5epochs/checkpoint-1065/ \
    --wandb-log /project/aip-wangcs/indrisch/LLaMA-Factory/wandb/wandb/offline-run-20260425_075152-5bx1dfj1 \
    --commit-message "VideoR1 LoRA adapter checkpoint trained for 1065 steps on 8 GPUs (5 epoch) total on the Scene30k dataset (corrected)" \
