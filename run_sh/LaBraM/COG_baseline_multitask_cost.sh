#!/bin/bash

# 全局变量定义
recon_path="/mnt/dataset4/yinuo/Rest_tuning/result/LaBraM/baseline"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:6"

# ####################### 第一个任务 ###########################################################
source activate EEGPT
export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH

train_sizes=(0.05 0.1 0.5 0.7)
val_size=0.1
cls_fitune_types=("lora" "full")

for ts in "${train_sizes[@]}"; do
    test_s=$(echo "0.9 - $ts" | bc)
    for cls_fitune_type in "${cls_fitune_types[@]}";
    do
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_baseline.py \
            --model labram_base_patch200_200 \
            --dataset 'cogbci' \
            --abs_pos_emb \
            --seed 42 \
            --disable_qkv_bias \
            --disable_rel_pos_bias \
            --epochs 300 \
            --batch_size 8 \
            --patience 10 \
            --lr 1e-4 \
            --train_size "$ts" \
            --test_size "$test_s" \
            --val_size "$val_size" \
            --device $cuda_device \
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5" \
            --pretrained_path /mnt/dataset0/UnonoU/personalized_FM/dataset_process/labram-base.pth \
            --num_classes 6 \
            --output_root "$recon_path/Multitask_time_cost_test/train_$ts" \
            --target_modules "$recon_target_modules" \
            --cls_fitune_type $cls_fitune_type \
            --cls_lora_alpha 32 \
            --cls_lora_r 8 \
            --cls_lora_dropout 0.1 \
            --multi_task \
            --multi_task_ids 1 2
    done
done