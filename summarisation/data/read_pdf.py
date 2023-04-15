import re
from pypdf import PdfReader

def read_pdf(file_path):
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        page_raw = page.extract_text()
        page_processed = re.sub('[^a-zA-ZæøåÆØÅ.]+', ' ', page_raw)
        page_processed = re.sub(r'\s+\.', '.', page_processed)
        pages.append(page_processed)
    return pages