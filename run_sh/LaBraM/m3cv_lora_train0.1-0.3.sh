#!/bin/bash

# ================= Global Variables =================
recon_path="/mnt/dataset4/yinuo/Rest_tuning/result/LaBraM/lora_mask_0.7/wo_label_channel"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:3"

# 实验参数设置
train_sizes=(0.1 0.2 0.3)
val_size=0.1
cls_fitune_types=("full" "lora" "linear")

# 激活环境
source activate EEGPT
export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH


for ts in "${train_sizes[@]}"; do
    # 动态计算 test_size (确保三者之和为 1.0)
    # 使用 bc 进行浮点数运算。如果系统没装 bc，也可以手动写死映射。
    test_s=$(echo "0.9 - $ts" | bc)

    # 1. Multitask Classification
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Multitask] Type: $cls_type, TrainSize: $ts"
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_sft_m3cv.py \
            --model labram_base_patch200_200 \
            --dataset 'm3cv' \
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
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_M3CV_multitask_ChangeFilter/m3cv_multitask_session1_temporal.h5" \
            --num_classes 11 \
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
            --multi_task_ids 1,2,3,4
    done

    # 2. Unitask Motor
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Unitask Motor] Type: $cls_type, TrainSize: $ts"
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_task_sft_cls.py \
            --model labram_base_patch200_200 \
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
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_M3CV_session1_unified/m3cv_Motor_Session1_temporal.h5" \
            --num_classes 3 \
            --output_root "$recon_path/classification/Motor/train_${ts}_b8" \
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

    # 3. Unitask p300
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Unitask p300] Type: $cls_type, TrainSize: $ts"
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
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_M3CV_session1_unified/m3cv_P300_Session1_temporal.h5" \
            --num_classes 2 \
            --output_root "$recon_path/classification/P300/train_${ts}_b8" \
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

    # 3. Unitask SSS
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Unitask SSS] Type: $cls_type, TrainSize: $ts"
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
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_M3CV_unitask_globalZscore/m3cv_SSS_Sesssion1_temporal.h5" \
            --num_classes 3 \
            --output_root "$recon_path/classification/SSS/train_${ts}_b8" \
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

    # 3. Unitask TS
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Unitask TS] Type: $cls_type, TrainSize: $ts"
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
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_M3CV_session1_unified/m3cv_TS_Session1_temporal.h5" \
            --num_classes 3 \
            --output_root "$recon_path/classification/TS/train_${ts}_b8" \
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