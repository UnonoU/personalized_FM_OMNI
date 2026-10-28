with open('src/train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

target_line = '            encoder_sub.load_state_dict(torch.load(args.pretrained_path, map_location=device, weights_only=False))\n'

fixed_count = 0
for i in range(len(lines) - 4):
    if lines[i].strip() == "elif args.model =='BIOT':" and lines[i+1].strip() == 'eeg_projector = BIOT_DataProjector()':
        # Find the encoder_sub.load_state_dict line in the next few lines
        for j in range(i+2, min(i+10, len(lines))):
            if lines[j] == target_line:
                replacement = [
                    '            \n',
                    '            # Load continue-pretrain checkpoint with proper unpacking\n',
                    '            print(f"Loading BIOT continue-pretrain checkpoint from: {args.pretrained_path}")\n',
                    '            checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)\n',
                    '            \n',
                    "            # Handle checkpoint dict format (model/optimizer/epoch/args)\n",
                    "            if isinstance(checkpoint, dict) and 'model' in checkpoint:\n",
                    "                state_dict = checkpoint['model']\n",
                    '                print(f"  Extracted state_dict from checkpoint[\'model\']")\n',
                    "            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:\n",
                    "                state_dict = checkpoint['state_dict']\n",
                    "            else:\n",
                    "                state_dict = checkpoint\n",
                    '            \n',
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
                    '            \n',
                    "            # Load with strict=False to handle any remaining mismatches\n",
                    "            missing, unexpected = encoder_sub.load_state_dict(new_state_dict, strict=False)\n",
                    '            if missing:\n',
                    '                print(f"  Missing keys (random init): {missing[:3]}{\'...\' if len(missing)>3 else \'\'}"\n',
                    '            if unexpected:\n',
                    '                print(f"  Unexpected keys (skipped): {unexpected[:3]}{\'...\' if len(unexpected)>3 else \'\'}"\n',
                    '            print(f"  BIOT continue-pretrain weights loaded successfully!")\n',
                ]
                lines = lines[:j] + replacement + lines[j+1:]
                fixed_count += 1
                print(f"Fixed occurrence at line {j+1}")
                break

with open('src/train.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Fixed {fixed_count} occurrences")
