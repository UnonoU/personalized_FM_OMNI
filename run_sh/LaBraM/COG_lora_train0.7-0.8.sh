#!/bin/bash

# ================= Global Variables =================
recon_path="/mnt/dataset4/yinuo/Rest_tuning/result/LaBraM/lora_mask_0.7/wo_label_channel_0418"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:4"

# 实验参数设置
train_sizes=(0.7 0.8)
val_size=0.1
cls_fitune_types=("full" "lora" "linear")

# 激活环境
source activate EEGPT
export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH


for ts in "${train_sizes[@]}"; do
    test_s=$(echo "0.9 - $ts" | bc)

    # 1. Multitask Classification
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Multitask] Type: $cls_type, TrainSize: $ts"
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_stf_cog.py \
            --dataset 'cogbci' \
            --abs_pos_emb \
            --seed 42 \
            --disable_qkv_bias \
            --disable_rel_pos_bias \
            --epochs 300 \
            --batch_size 8 \
            --lr 1e-4 \
            --train_size "$ts" \
            --test_size "$test_s" \
            --val_size "$val_size" \
            --device $cuda_device \
            --recon_results_dir "$recon_path" \
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5" \
            --num_classes 6 \
            --output_root "$recon_path/Multitask/train_$ts" \
            --lora_r "$lora_r" \
            --lora_alpha "$lora_alpha" \
            --lora_dropout "$lora_dropout" \
            --fitune_type 'lora' \
            --cls_fitune_type "$cls_type" \
            --cls_lora_alpha 32 \
            --cls_lora_r 8 \
            --cls_lora_dropout 0.1 \
            --multi_task \
            --multi_task_ids 1,2
    done

    # 2. Unitask MATB
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Unitask MATB] Type: $cls_type, TrainSize: $ts"
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_task_sft_cls.py \
            --model labram_base_patch200_200 \
            --dataset "cogbci" \
            --abs_pos_emb \
            --seed 42 \
            --disable_qkv_bias \
            --disable_rel_pos_bias \
            --epochs 200 \
            --batch_size 8 \
            --lr 1e-4 \
            --train_size "$ts" \
            --test_size "$test_s" \
            --val_size "$val_size" \
            --device $cuda_device \
            --recon_results_dir "$recon_path" \
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_session1_ChangeFilter/cogbci_matb.h5" \
            --num_classes 3 \
            --output_root "$recon_path/classification/MATB/train_${ts}_b8" \
            --lora_r "$lora_r" \
            --lora_alpha "$lora_alpha" \
            --lora_dropout "$lora_dropout" \
            --fitune_type 'lora' \
            --cls_fitune_type "$cls_type" \
            --cls_lora_alpha 32 \
            --cls_lora_r 8 \
            --cls_lora_dropout 0.1 \
            --dataset_suffle
    done

    # 3. Unitask nback
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Unitask nback] Type: $cls_type, TrainSize: $ts"
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_task_sft_cls.py \
            --model labram_base_patch200_200 \
            --dataset "cogbci" \
            --abs_pos_emb \
            --seed 42 \
            --disable_qkv_bias \
            --disable_rel_pos_bias \
            --epochs 200 \
            --batch_size 8 \
            --lr 1e-4 \
            --train_size "$ts" \
            --test_size "$test_s" \
            --val_size "$val_size" \
            --device $cuda_device \
            --recon_results_dir "$recon_path" \
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_session1_ChangeFilter/cogbci_nback.h5" \
            --num_classes 3 \
            --output_root "$recon_path/classification/nback/train_${ts}_b8" \
            --lora_r "$lora_r" \
            --lora_alpha "$lora_alpha" \
            --lora_dropout "$lora_dropout" \
            --fitune_type 'lora' \
            --cls_fitune_type "$cls_type" \
            --cls_lora_alpha 32 \
            --cls_lora_r 8 \
            --cls_lora_dropout 0.1 \
            --dataset_suffle
    done

done