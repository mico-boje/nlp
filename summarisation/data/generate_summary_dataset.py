## Generate summary datasets from data/pdf folder
import os
import json

from summarisation.data.read_pdf import read_pdf
from summarisation.data.get_summary_openai import get_summary
from summarisation.utils.utility import get_root_path


def main():
    pdf_folder = os.path.join(get_root_path(), 'data', 'pdf')
    for pdf in os.listdir(pdf_folder):
        processed_pdf = process_pdf(os.path.join(pdf_folder, pdf))
        # Save to json
        with open(os.path.join(get_root_path(), 'data', 'json', f'{processed_pdf["file_name"]}.json'.strip()), 'w') as f:
            json.dump(processed_pdf, f, indent=4)
        print(f'Processed {pdf} and saved to json file')
        # move PDF to data/processed_pdf
        os.rename(os.path.join(pdf_folder, pdf), os.path.join(get_root_path(), 'data', 'processed_pdf', pdf))
        print(f'Moved {pdf} to processed_pdf folder')

def process_pdf(pdf_path):
    file_name = os.path.basename(pdf_path).strip(".pdf")
    pages = read_pdf(pdf_path)
    summary = get_summary(pages)
    return {"text": ''.join(pages), "summary": summary, "file_name": file_name}
    
if __name__ == '__main__':
    main()