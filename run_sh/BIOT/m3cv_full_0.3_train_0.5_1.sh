#!/bin/bash

# 全局变量定义
recon_path="/mnt/dataset4/yinuo/Rest_tuning/result/BIOT/full_mask_0.3/wo_label_channel"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:1"

# ####################### 第一个任务 ###########################################################
source activate EEGPT
export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH
train_size=0.5
test_size=0.4

# ##################### P300 train 0.1 ###########################################################
# cls_fitune_types=( "linear"  "full"  "lora")
# for cls_fitune_type in "${cls_fitune_types[@]}";
# do
#     python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_task_sft_cls.py \
#         --model BIOT \
#         --dataset "m3cv" \
#         --abs_pos_emb \
#         --seed 42 \
#         --disable_qkv_bias \
#         --disable_rel_pos_bias \
#         --epochs 200 \
#         --batch_size 8 \
#         --lr 1e-4 \
#         --train_size $train_size \
#         --test_size $test_size \
#         --val_size 0.1 \
#         --device $cuda_device \
#         --recon_results_dir "$recon_path" \
#         --h5_path "/mnt/dataset4/fuxy/m3cv/processed_M3CV_unitask_globalZscore/m3cv_P300_Session1_temporal.h5" \
#         --num_classes 2 \
#         --output_root "$recon_path/classification/P300/train_$train_size" \
#         --target_modules "$recon_target_modules" \
#         --lora_r "$lora_r" \
#         --lora_alpha "$lora_alpha" \
#         --lora_dropout "$lora_dropout" \
#         --fitune_type 'full' \
#         --cls_fitune_type $cls_fitune_type \
#         --cls_lora_alpha 32 \
#         --cls_lora_r 8 \
#         --cls_lora_dropout 0.1 \
#         --dataset_suffle
# done



# ###################### SSS train 0.1 ###########################################################
# cls_fitune_types=( "full" "linear"  "lora")
# for cls_fitune_type in "${cls_fitune_types[@]}";
# do
#     python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_task_sft_cls.py \
#         --model BIOT \
#         --dataset "m3cv" \
#         --abs_pos_emb \
#         --seed 42 \
#         --disable_qkv_bias \
#         --disable_rel_pos_bias \
#         --epochs 200 \
#         --batch_size 8 \
#         --lr 1e-4 \
#         --train_size $train_size \
#         --test_size $test_size \
#         --val_size 0.1 \
#         --device $cuda_device \
#         --recon_results_dir "$recon_path" \
#         --h5_path "/mnt/dataset4/fuxy/m3cv/processed_M3CV_unitask_globalZscore/m3cv_SSS_Sesssion1_temporal.h5" \
#         --num_classes 3 \
#         --output_root "$recon_path/classification/SSS/train_$train_size" \
#         --target_modules "$recon_target_modules" \
#         --lora_r "$lora_r" \
#         --lora_alpha "$lora_alpha" \
#         --lora_dropout "$lora_dropout" \
#         --fitune_type 'full' \
#         --cls_fitune_type $cls_fitune_type \
#         --cls_lora_alpha 32 \
#         --cls_lora_r 8 \
#         --cls_lora_dropout 0.1 \
#         --dataset_suffle
# done

###################### TS train 0.1 ###########################################################
cls_fitune_types=(  "linear"  "lora" "full")
for cls_fitune_type in "${cls_fitune_types[@]}";
do
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
        --train_size $train_size \
        --test_size $test_size \
        --val_size 0.1 \
        --device $cuda_device \
        --recon_results_dir "$recon_path" \
        --h5_path "/mnt/dataset4/fuxy/m3cv/processed_M3CV_unitask_globalZscore/m3cv_TransientState_Session1_temporal.h5" \
        --num_classes 3 \
        --output_root "$recon_path/classification/TS/train_$train_size" \
        --target_modules "$recon_target_modules" \
        --lora_r "$lora_r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout "$lora_dropout" \
        --fitune_type 'full' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --dataset_suffle
done
