#!/bin/bash

# 全局变量定义
recon_path="/mnt/dataset4/yinuo/Rest_tuning/result/BIOT/full_mask_0.3/wo_label_channel"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:1"


train_sizes=(0.7 0.1 0.2 0.3 0.4 0.5 0.6)
val_size=0.1
cls_fitune_types=("linear")

# 激活环境
source activate EEGPT
export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH


for ts in "${train_sizes[@]}"; do
    test_s=$(echo "0.9 - $ts" | bc)
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Unitask TS] Type: $cls_type, TrainSize: $ts"
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_task_sft_cls.py \
            --model BIOT \
            --dataset "m3cv" \
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
            --h5_path "/mnt/dataset0/UnonoU/personalized_FM/dataset_process/m3cv_Transient_Sensory.h5" \
            --num_classes 3 \
            --output_root "$recon_path/classification/TS/train_${ts}_old_data" \
            --lora_r "$lora_r" \
            --lora_alpha "$lora_alpha" \
            --lora_dropout "$lora_dropout" \
            --fitune_type 'full' \
            --cls_fitune_type "$cls_type" \
            --cls_lora_alpha 32 \
            --cls_lora_r 8 \
            --cls_lora_dropout 0.1 \
            --dataset_suffle
    done

done