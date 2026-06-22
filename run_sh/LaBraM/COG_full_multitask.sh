#!/bin/bash

# 全局变量定义
recon_path="/mnt/dataset4/yinuo/Rest_tuning/result/LaBraM/full_mask_0.3/wo_label_channel_0418"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:0"

source activate EEGPT
export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH

################## multitask ##########################################

cls_fitune_types=( "full" )
for cls_fitune_type in "${cls_fitune_types[@]}";
do
    python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_stf_cog.py \
        --dataset 'cogbci' \
        --abs_pos_emb \
        --seed 42 \
        --disable_qkv_bias \
        --disable_rel_pos_bias \
        --epochs 300 \
        --batch_size 8 \
        --lr 1e-4 \
        --train_size 0.5 \
        --test_size 0.4 \
        --val_size 0.1 \
        --device $cuda_device \
        --recon_results_dir "$recon_path" \
        --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5" \
        --num_classes 6 \
        --output_root "$recon_path/Multitask_time_cost_test/train_0.5" \
        --target_modules "$recon_target_modules" \
        --lora_r "$lora_r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout "$lora_dropout" \
        --fitune_type 'full' \
        --cls_fitune_type $cls_fitune_type \
        --cls_lora_alpha 32 \
        --cls_lora_r 8 \
        --cls_lora_dropout 0.1 \
        --multi_task \
        --multi_task_ids 1,2
done

# #################### train 0.1 ######################
# cls_fitune_types=( "full" )
# for cls_fitune_type in "${cls_fitune_types[@]}";
# do
#     python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_stf_cog.py \
#         --dataset 'cogbci' \
#         --abs_pos_emb \
#         --seed 42 \
#         --disable_qkv_bias \
#         --disable_rel_pos_bias \
#         --epochs 300 \
#         --batch_size 8 \
#         --lr 1e-4 \
#         --train_size 0.1 \
#         --test_size 0.8 \
#         --val_size 0.1 \
#         --device $cuda_device \
#         --recon_results_dir "$recon_path" \
#         --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5" \
#         --num_classes 6 \
#         --output_root "$recon_path/Multitask_time_cost_test/train_0.1" \
#         --target_modules "$recon_target_modules" \
#         --lora_r "$lora_r" \
#         --lora_alpha "$lora_alpha" \
#         --lora_dropout "$lora_dropout" \
#         --fitune_type 'full' \
#         --cls_fitune_type $cls_fitune_type \
#         --cls_lora_alpha 32 \
#         --cls_lora_r 8 \
#         --cls_lora_dropout 0.1 \
#         --multi_task \
#         --multi_task_ids 1,2
# done


# #################### train 0.7 ######################
# cls_fitune_types=( "full"  )
# for cls_fitune_type in "${cls_fitune_types[@]}";
# do
#     python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_stf_cog.py \
#         --dataset 'cogbci' \
#         --abs_pos_emb \
#         --seed 42 \
#         --disable_qkv_bias \
#         --disable_rel_pos_bias \
#         --epochs 300 \
#         --batch_size 8 \
#         --lr 1e-4 \
#         --train_size 0.7 \
#         --test_size 0.2 \
#         --val_size 0.1 \
#         --device $cuda_device \
#         --recon_results_dir "$recon_path" \
#         --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5" \
#         --num_classes 6 \
#         --output_root "$recon_path/Multitask_time_cost_test/train_0.7" \
#         --target_modules "$recon_target_modules" \
#         --lora_r "$lora_r" \
#         --lora_alpha "$lora_alpha" \
#         --lora_dropout "$lora_dropout" \
#         --fitune_type 'full' \
#         --cls_fitune_type $cls_fitune_type \
#         --cls_lora_alpha 32 \
#         --cls_lora_r 8 \
#         --cls_lora_dropout 0.1 \
#         --multi_task \
#         --multi_task_ids 1,2
# done