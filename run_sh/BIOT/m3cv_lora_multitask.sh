#!/bin/bash

# 全局变量定义
recon_path="/mnt/dataset4/yinuo/Rest_tuning/result/BIOT/lora_mask_0.7/wo_label_channel"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:5"

# ####################### 第一个任务 ###########################################################
source activate EEGPT
export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH

# 实验参数设置

train_sizes=(0.7)
val_size=0.1
cls_fitune_types=("lora")

# 激活环境
source activate EEGPT
export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH


for ts in "${train_sizes[@]}"; do
    test_s=$(echo "0.9 - $ts" | bc)
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Multitask] Type: $cls_type, TrainSize: $ts"
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_sft_m3cv.py \
            --model BIOT \
            --dataset 'm3cv' \
            --patience 50 \
            --abs_pos_emb \
            --seed 91 \
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
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_M3CV_multitask_ChangeFilter/m3cv_multitask_session1_temporal.h5" \
            --num_classes 11 \
            --output_root "$recon_path/Multitask/train_${ts}_0502_2" \
            --lora_r "$lora_r" \
            --lora_alpha "$lora_alpha" \
            --lora_dropout "$lora_dropout" \
            --fitune_type 'lora' \
            --cls_fitune_type "$cls_type" \
            --cls_lora_alpha 32 \
            --cls_lora_r 8 \
            --cls_lora_dropout 0.1 \
            --multi_task \
            --multi_task_ids 1,2,3,4
    done
done