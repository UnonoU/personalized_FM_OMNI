#!/bin/bash

# ================= 全局变量定义 =================
recon_path="/mnt/dataset4/yinuo/Rest_tuning/result/BIOT/full_mask_0.3/wo_label_channel_0418"
lora_r=8
lora_alpha=32
lora_dropout=0.1
cuda_device="cuda:4"

# 实验梯度设置
train_sizes=(0.1)
val_size=0.1
cls_fitune_types=("full" "lora")

# 激活环境
source activate EEGPT
export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH

for ts in "${train_sizes[@]}"; do
    test_s=$(echo "0.9 - $ts" | bc)

#     # 2. Multitask Classification 
#     for cls_type in "${cls_fitune_types[@]}"; do
#         echo "[Multitask] Type: $cls_type, TS: $ts"
#         python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_stf_cog.py \
#             --model "BIOT" \
#             --dataset 'cogbci' \
#             --abs_pos_emb \
#             --seed 42 \
#             --disable_qkv_bias \
#             --disable_rel_pos_bias \
#             --epochs 300 \
#             --batch_size 8 \
#             --lr 1e-4 \
#             --train_size "$ts" \
#             --test_size "$test_s" \
#             --val_size "$val_size" \
#             --device $cuda_device \
#             --recon_results_dir "$recon_path" \
#             --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5" \
#             --num_classes 6 \
#             --output_root "$recon_path/Multitask/train_$ts" \
#             --target_modules "$recon_target_modules" \
#             --lora_r "$lora_r" \
#             --lora_alpha "$lora_alpha" \
#             --lora_dropout "$lora_dropout" \
#             --fitune_type 'full' \
#             --cls_fitune_type "$cls_type" \
#             --cls_lora_alpha 32 \
#             --cls_lora_r 8 \
#             --cls_lora_dropout 0.1 \
#             --multi_task \
#             --multi_task_ids 1,2
#     done

    # 3. Unitask MATB
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Unitask MATB] Type: $cls_type, TS: $ts"
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_task_sft_cls.py \
            --model "BIOT" \
            --dataset "cogbci" \
            --abs_pos_emb \
            --seed 42 \
            --disable_qkv_bias \
            --disable_rel_pos_bias \
            --epochs 200 \
            --batch_size 8 \
            --patience 30 \
            --lr 1e-4 \
            --train_size "$ts" \
            --test_size "$test_s" \
            --val_size "$val_size" \
            --device $cuda_device \
            --recon_results_dir "$recon_path" \
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_session1_ChangeFilter/cogbci_matb.h5" \
            --num_classes 3 \
            --output_root "$recon_path/classification/MATB/train_${ts}_cost_time_1" \
            --target_modules "$recon_target_modules" \
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

    # 4. Unitask nback
    for cls_type in "${cls_fitune_types[@]}"; do
        echo "[Unitask nback] Type: $cls_type, TS: $ts"
        python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_task_sft_cls.py \
            --model "BIOT" \
            --dataset "cogbci" \
            --abs_pos_emb \
            --seed 42 \
            --disable_qkv_bias \
            --disable_rel_pos_bias \
            --epochs 200 \
            --batch_size 8 \
            --patience 30 \
            --lr 1e-4 \
            --train_size "$ts" \
            --test_size "$test_s" \
            --val_size "$val_size" \
            --device $cuda_device \
            --recon_results_dir "$recon_path" \
            --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_session1_ChangeFilter/cogbci_nback.h5" \
            --num_classes 3 \
            --output_root "$recon_path/classification/nback/train_${ts}_cost_time_1" \
            --target_modules "$recon_target_modules" \
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
