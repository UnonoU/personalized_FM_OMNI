#!/bin/bash

# 全局变量定义
recon_path="/mnt/dataset4/yinuo/Rest_tuning/result/LaBraM/full_mask_0.3/wo_label_channel_0418"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:1"

# ####################### 第一个任务 ###########################################################
source activate EEGPT
export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH
cls_fitune_types=( "lora" "linear" "full")
n_shot=(2 4 6 8 10)
for cls_fitune_type in "${cls_fitune_types[@]}";do
    for n in "${n_shot[@]}";do
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_stf_cog.py \
            --model "labram_base_patch200_200" \
            --dataset "cogbci" \
            --abs_pos_emb \
            --seed 42 \
            --disable_qkv_bias \
            --disable_rel_pos_bias \
            --epochs 300 \
            --batch_size 1 \
            --lr 1e-4 \
            --train_size 0.7 \
            --test_size 0.2 \
            --val_size 0.1 \
            --device $cuda_device \
            --recon_results_dir "$recon_path" \
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5" \
            --num_classes 6 \
            --output_root "$recon_path/Multitask/fewshot_cog/$n" \
            --lora_r "$lora_r" \
            --lora_alpha "$lora_alpha" \
            --lora_dropout "$lora_dropout" \
            --fitune_type 'full' \
            --cls_fitune_type $cls_fitune_type \
            --cls_lora_alpha 32 \
            --cls_lora_r 8 \
            --cls_lora_dropout 0.1 \
            --n_shot $n \
            --multitask_fewshot \
            --multi_task_ids 1,2
    done
done
