export PYTHONPATH=/mnt/dataset4/yinuo/Rest_tuning:$PYTHONPATH

# 设置输出路径和分类相关的参数
lora_r=8 
lora_alpha=32 
lora_dropout=0.1 
cuda_device="cuda:1"

recon_path="/mnt/dataset4/yinuo/Rest_tuning/result/LaBraM/baseline_continue_pretrain_all"

source activate EEGPT
for cls_type in  "full" "lora" "linear"
do
    python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_baseline.py \
        --model labram_base_patch200_200 \
        --dataset "cogbci" \
        --abs_pos_emb \
        --seed 42 \
        --disable_qkv_bias \
        --disable_rel_pos_bias \
        --epochs 300 \
        --batch_size 8 \
        --lr 1e-4 \
        --train_size 0.1 \
        --test_size 0.8 \
        --val_size 0.1 \
        --device $cuda_device \
        --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5" \
        --pretrained_path "/mnt/dataset4/yinuo/personlized_FM/LaBraM/checkpoints/labram_base_resume/checkpoint.pth" \
        --num_classes 6 \
        --output_root "$recon_path/Multitask/train_0.1" \
        --cls_fitune_type "$cls_type" \
        --cls_lora_r $lora_r \
        --cls_lora_alpha $lora_alpha \
        --cls_lora_dropout $lora_dropout \
        --multi_task \
        --multi_task_ids 1 2
done

# for cls_type in "linear" “lora”
# do
#     python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_multitask_baseline.py \
#         --model labram_base_patch200_200 \
#         --dataset "cogbci" \
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
#         --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_multitask_session1_ChangeFilter/cogbci_multitask_session1_temporal.h5" \
#         --pretrained_path '/mnt/dataset4/yinuo/personlized_FM/LaBraM/checkpoints/labram-base.pth' \
#         --num_classes 6 \
#         --output_root "$recon_path/Multitask/train_0.7" \
#         --cls_fitune_type "$cls_type" \
#         --cls_lora_r $lora_r \
#         --cls_lora_alpha $lora_alpha \
#         --cls_lora_dropout $lora_dropout \
#         --multi_task \
#         --multi_task_ids 1 2

# done

# # ##################### unitask ##################################
# # ##################### MATB ##################################
# for cls_type in "lora" "full" "linear"
# do
#     python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_baseline_stf.py \
#         --model labram_base_patch200_200 \
#         --dataset "cogbci" \
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
#         --device "cuda:3" \
#         --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_session1_ChangeFilter/cogbci_matb.h5" \
#         --pretrained_path '/mnt/dataset4/yinuo/personlized_FM/LaBraM/checkpoints/labram-base.pth' \
#         --num_classes 3 \
#         --output_root "$recon_path/classification/MATB/train_0.1" \
#         --dataset_suffle \
#         --cls_fitune_type "$cls_type" \
#         --cls_lora_r $lora_r \
#         --cls_lora_alpha $lora_alpha \
#         --cls_lora_dropout $lora_dropout \
#         --dataset_suffle
# done

# for cls_type in "lora" "linear" "full" 
# do
#     python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_baseline_stf.py \
#         --model labram_base_patch200_200 \
#         --dataset "cogbci" \
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
#         --device "cuda:3" \
#         --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_session1_ChangeFilter/cogbci_matb.h5" \
#         --pretrained_path '/mnt/dataset4/yinuo/personlized_FM/LaBraM/checkpoints/labram-base.pth' \
#         --num_classes 3 \
#         --output_root "$recon_path/classification/MATB/train_0.7" \
#         --dataset_suffle \
#         --cls_fitune_type "$cls_type" \
#         --cls_lora_r $lora_r \
#         --cls_lora_alpha $lora_alpha \
#         --cls_lora_dropout $lora_dropout \
#         --dataset_suffle
# done

# ##################### Nback ##################################
# for cls_type in "lora" "full" "linear"
# do
#     python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_baseline_stf.py \
#         --model labram_base_patch200_200 \
#         --dataset "cogbci" \
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
#         --device "cuda:3" \
#         --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_session1_ChangeFilter/cogbci_nback.h5" \
#         --pretrained_path "/mnt/dataset4/yinuo/personlized_FM/LaBraM/checkpoints/labram-base.pth" \
#         --num_classes 3 \
#         --output_root "$recon_path/classification/Nback/train_0.1" \
#         --dataset_suffle \
#         --cls_fitune_type "$cls_type" \
#         --cls_lora_r $lora_r \
#         --cls_lora_alpha $lora_alpha \
#         --cls_lora_dropout $lora_dropout \
#         --dataset_suffle
# done

# for cls_type in "lora" "linear" "full" 
# do
#     python3 /mnt/dataset4/yinuo/Rest_tuning/train/run_baseline_stf.py \
#         --model labram_base_patch200_200 \
#         --dataset "cogbci" \
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
#         --device "cuda:3" \
#         --h5_path "/mnt/dataset4/fuxy/m3cv/processed_cogbci_session1_ChangeFilter/cogbci_nback.h5" \
#         --pretrained_path '/mnt/dataset4/yinuo/personlized_FM/LaBraM/checkpoints/labram-base.pth' \
#         --num_classes 3 \
#         --output_root "$recon_path/classification/Nback/train_0.7" \
#         --dataset_suffle \
#         --cls_fitune_type "$cls_type" \
#         --cls_lora_r $lora_r \
#         --cls_lora_alpha $lora_alpha \
#         --cls_lora_dropout $lora_dropout \
#         --dataset_suffle
# done