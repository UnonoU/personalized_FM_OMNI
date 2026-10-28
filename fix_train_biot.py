import re

with open('src/train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line that contains 'elif args.model'"'"'==BIOT'"'"':' around line 8463 (0-indexed: 8462)
# and the next few lines that need replacement
target_idx = None
for i, line in enumerate(lines):
    if 'elif args.model' in line and "'BIOT'" in line and 'eeg_projector = BIOT_DataProjector()' in lines[i+1]:
        target_idx = i
        break

if target_idx is None:
    print("ERROR: Could not find the target BIOT block")
    exit(1)

print(f"Found target at line {target_idx + 1}")

# Find the end of the block we want to replace (the line after encoder_sub.load_state_dict(...))
end_idx = None
for j in range(target_idx + 4, min(target_idx + 20, len(lines))):
    if 'encoder_sub.load_state_dict(torch.load(args.pretrained_path' in lines[j]:
        end_idx = j
        break

if end_idx is None:
    print("ERROR: Could not find the end of the target block")
    exit(1)

print(f"End of block at line {end_idx + 1}")

# Build replacement
replacement = [
    "        elif args.model =='BIOT':\n",
    "            eeg_projector = BIOT_DataProjector()\n",
    "            from src.biot import BIOTEncoder\n",
    "            encoder_sub = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=18).to(device)\n",
    "            \n",
    "            # Load continue-pretrain checkpoint with proper unpacking\n",
    '            print(f"Loading BIOT continue-pretrain checkpoint from: {args.pretrained_path}")\n',
    "            checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)\n",
    "            \n",
    "            # Handle checkpoint dict format (model/optimizer/epoch/args)\n",
    "            if isinstance(checkpoint, dict) and 'model' in checkpoint:\n",
    "                state_dict = checkpoint['model']\n",
    '                print(f"  Extracted state_dict from checkpoint[\'model\']")\n',
    "            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:\n",
    "                state_dict = checkpoint['state_dict']\n",
    "            else:\n",
    "                state_dict = checkpoint\n",
    "            \n",
    "            # Remove 'biot.' prefix (continue-pretrain uses UnsupervisedPretrainSafe wrapper)\n",
    "            new_state_dict = {}\n",
    "            for k, v in state_dict.items():\n",
    "                if k.startswith('biot.'):\n",
    "                    new_state_dict[k[5:]] = v  # Remove 'biot.' prefix\n",
    "                elif k.startswith('prediction.'):\n",
    "                    # Skip prediction MLP (only used during pretraining)\n",
    "                    continue\n",
    "                else:\n",
    "                    new_state_dict[k] = v\n",
    "            \n",
    "            # Load with strict=False to handle any remaining mismatches\n",
    "            missing, unexpected = encoder_sub.load_state_dict(new_state_dict, strict=False)\n",
    '            if missing:\n',
    '                print(f"  Missing keys (random init): {missing[:3]}{\'...\' if len(missing)>3 else \'\'}"\n',
    '            if unexpected:\n',
    '                print(f"  Unexpected keys (skipped): {unexpected[:3]}{\'...\' if len(unexpected)>3 else \'\'}"\n',
    '            print(f"  BIOT continue-pretrain weights loaded successfully!")\n',
]

# Replace lines target_idx to end_idx with replacement
new_lines = lines[:target_idx] + replacement + lines[end_idx+1:]

with open('src/train.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fixed!")
