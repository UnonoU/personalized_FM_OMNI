#!/bin/bash

# 全局变量定义
recon_path="/mnt/dataset4/yinuo/Rest_tuning/result/BIOT/Baseline"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:5"

# ####################### 第一个任务 ###########################################################
source activate EEGPT
export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH

train_sizes=(0.7)
val_size=0.1
cls_fitune_types=("full")

for ts in "${train_sizes[@]}"; do
    test_s=$(echo "0.9 - $ts" | bc)
    for cls_fitune_type in "${cls_fitune_types[@]}";
    do
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_baseline.py \
            --model 'BIOT' \
            --dataset 'cogbci' \
            --abs_pos_emb \
            --seed 42 \
            --disable_qkv_bias \
            --disable_rel_pos_bias \
            --epochs 300 \
            --batch_size 32 \
            --patience 30 \
            --lr 1e-4 \
            --train_size "$ts" \
            --test_size "$test_s" \
            --val_size "$val_size" \
            --device $cuda_device \
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5" \
            --pretrained_path /mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/EEG-six-datasets-18-channels.ckpt \
            --num_classes 6 \
            --output_root "$recon_path/Multitask_time_cost_test_0507/train_$ts" \
            --target_modules "$recon_target_modules" \
            --cls_fitune_type $cls_fitune_type \
            --cls_lora_alpha 32 \
            --cls_lora_r 8 \
            --cls_lora_dropout 0.1 \
            --multi_task \
            --multi_task_ids 1 2
    done
done