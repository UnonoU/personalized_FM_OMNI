import json
import glob
import statistics

base_path = "/mnt/dataset4/yinuo/Rest_tuning/result/LaBraM/lora_mask_0.7/wo_label_channel_0418/Multitask_time_cost_test/train_0.05/cogbci/lora"

# 获取所有subject的metrics.json文件
subject_dirs = sorted(glob.glob(f"{base_path}/subject_*/metrics.json"))

best_epochs = []
subject_epochs = []

for filepath in subject_dirs:
    with open(filepath, 'r') as f:
        data = json.load(f)
    subject_id = data.get('subject_id', 'unknown')
    best_epoch = data.get('best_epoch')
    if best_epoch is not None:
        best_epochs.append(best_epoch)
        subject_epochs.append((subject_id, best_epoch))

print("=" * 50)
print("各subject的best epoch:")
print("-" * 50)
for subject_id, epoch in subject_epochs:
    print(f"  subject_{subject_id:02d}: {epoch}")

n = len(best_epochs)
mean_val = sum(best_epochs) / n
var_val = sum((x - mean_val) ** 2 for x in best_epochs) / n

print("=" * 50)
print(f"总共统计了 {n} 个subject")
print(f"best epoch 列表: {best_epochs}")
print()
print(f"均值 (Mean):   {mean_val:.4f}")
print(f"方差 (Var):    {var_val:.4f}")
print(f"标准差 (Std):  {var_val ** 0.5:.4f}")
print(f"中位数 (Med):  {statistics.median(best_epochs):.4f}")
print(f"最小值 (Min):  {min(best_epochs)}")
print(f"最大值 (Max):  {max(best_epochs)}")
print("=" * 50)
