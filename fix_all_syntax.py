with open('src/train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix all lines that have the missing closing paren pattern
for i in range(len(lines)):
    if "print(f\"  Missing keys (random init): {missing[:3]}{'...' if len(missing)>3 else ''}\"" in lines[i] and not lines[i].strip().endswith(')'):
        lines[i] = lines[i].rstrip() + ')\n'
        print(f"Fixed missing at line {i+1}")
    if "print(f\"  Unexpected keys (skipped): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}\"" in lines[i] and not lines[i].strip().endswith(')'):
        lines[i] = lines[i].rstrip() + ')\n'
        print(f"Fixed unexpected at line {i+1}")

with open('src/train.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Done!")
