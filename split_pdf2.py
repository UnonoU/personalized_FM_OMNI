with open(r"W:\yinuo\Rest_tuning\pdf2_text.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Write chunks
chunk_size = 300
for i in range(0, len(lines), chunk_size):
    chunk = lines[i:i+chunk_size]
    out_path = f"W:\\yinuo\\Rest_tuning\\pdf2_part_{i//chunk_size + 1}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(chunk)
    print(f"Wrote {len(chunk)} lines to {out_path}")
