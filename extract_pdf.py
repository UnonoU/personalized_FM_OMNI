import fitz  # PyMuPDF

pdf_path = r"C:\Users\nono\Documents\WPSDrive\403841822\WPS云盘\nonoZhang\project\Rest-tuning\NeurIPS2026_Rest_Tuning.pdf"

doc = fitz.open(pdf_path)
print(f"Total pages: {doc.page_count}")

# Extract key sections
key_pages = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
for p in key_pages:
    if p < doc.page_count:
        page = doc[p]
        text = page.get_text()
        print(f"\n=== Page {p} ===\n")
        print(text[:3000])

doc.close()
