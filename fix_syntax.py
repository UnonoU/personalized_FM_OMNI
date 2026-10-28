with open('src/train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix lines 6976 and 6978 (0-indexed: 6975 and 6977)
lines[6975] = '                print(f"  Missing keys (random init): {missing[:3]}{\'...\' if len(missing)>3 else \'\'}")\n'
lines[6977] = '                print(f"  Unexpected keys (skipped): {unexpected[:3]}{\'...\' if len(unexpected)>3 else \'\'}")\n'

with open('src/train.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed!")
