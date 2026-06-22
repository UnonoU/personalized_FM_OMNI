import json
import os
import argparse
import statistics

def calculate_avg_test_acc(base_dir, subjects=None):
    """
    统计指定被试的平均 test_acc
    
    Args:
        base_dir: 包含 subject_{id} 文件夹的目录
        subjects: 指定的 subject id 列表，如果为 None 则统计所有存在的 subject
    """
    if subjects is None:
        # 自动发现所有 subject 文件夹
        subjects = []
        for d in sorted(os.listdir(base_dir)):
            if d.startswith('subject_') and os.path.isdir(os.path.join(base_dir, d)):
                try:
                    sid = int(d.split('_')[1])
                    subjects.append(sid)
                except ValueError:
                    continue
        subjects.sort()
    
    test_accs = []
    results = []
    
    for sid in subjects:
        subject_dir = os.path.join(base_dir, f"subject_{sid:02d}")
        metrics_path = os.path.join(subject_dir, "metrics.json")
        
        if not os.path.exists(metrics_path):
            print(f"  ⚠️  Subject {sid:02d}: metrics.json not found, skipped")
            continue
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        test_acc = metrics.get('test_acc', None)
        if test_acc is not None:
            test_accs.append(test_acc)
            results.append({
                'subject_id': sid,
                'test_acc': test_acc,
                'test_f1': metrics.get('test_f1', None),
                'best_val_acc': metrics.get('best_val_acc', None),
                'best_epoch': metrics.get('best_epoch', None)
            })
            print(f"  ✓ Subject {sid:02d}: test_acc = {test_acc:.4f}")
        else:
            print(f"  ⚠️  Subject {sid:02d}: test_acc not found in metrics.json")
    
    print("\n" + "="*60)
    print(f"Total subjects processed: {len(results)}")
    
    if len(test_accs) > 0:
        mean_acc = statistics.mean(test_accs)
        std_acc = statistics.stdev(test_accs) if len(test_accs) > 1 else 0.0
        print(f"\n📊 Average Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"📊 Min Test Accuracy:     {min(test_accs):.4f}")
        print(f"📊 Max Test Accuracy:     {max(test_accs):.4f}")
        print(f"📊 Median Test Accuracy:  {statistics.median(test_accs):.4f}")
    else:
        print("\n✗ No valid test_acc found")
    
    return results, test_accs


def main():
    parser = argparse.ArgumentParser(description='统计指定被试的平均 test_acc')
    parser.add_argument('--base_dir',
                        default='/mnt/dataset4/yinuo/Rest_tuning/result/LaBraM/baseline_continue_pretrain_new/1_15/Multitask/train_0.1/cogbci/linear',
                        type=str,
                        help='包含 subject_{id} 文件夹的目录')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help='指定一个或多个 subject id (例如: --subjects 1 3 5)')
    
    args = parser.parse_args()
    
    print(f"📁 Base directory: {args.base_dir}")
    if args.subjects:
        print(f"🎯 Specified subjects: {args.subjects}")
    else:
        print("🎯 Mode: All available subjects")
    print("="*60)
    
    calculate_avg_test_acc(args.base_dir, args.subjects)


if __name__ == '__main__':
    main()
