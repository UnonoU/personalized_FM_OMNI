import pdfplumber
import sys

pdf1_path = r"W:\yinuo\personlized_FM\1_rebuttal\2405.19346v2.pdf"
pdf2_path = r"W:\yinuo\personlized_FM\1_rebuttal\Kwak et al. - 2023 - Subject-Invariant Deep Neural Networks Based on Baseline Correction for EEG Motor Imagery BCI.pdf"

def extract_text(path, out_path, max_pages=None):
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        text = ""
        for i, page in enumerate(pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {i+1} ---\n" + page_text
            except Exception as e:
                text += f"\n--- Page {i+1} ERROR: {e} ---\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted {len(text)} chars to {out_path}")

extract_text(pdf1_path, r"W:\yinuo\Rest_tuning\pdf1_text.txt")
extract_text(pdf2_path, r"W:\yinuo\Rest_tuning\pdf2_text.txt")
