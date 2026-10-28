import sys
import subprocess

try:
    import PyPDF2
    print("PyPDF2 available")
except Exception as e:
    print(f"PyPDF2 not available: {e}")

try:
    import pdfplumber
    print("pdfplumber available")
except Exception as e:
    print(f"pdfplumber not available: {e}")

try:
    import fitz
    print("PyMuPDF available")
except Exception as e:
    print(f"PyMuPDF not available: {e}")
