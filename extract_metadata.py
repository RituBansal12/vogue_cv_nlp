import os
import re
import pdfplumber
import pandas as pd
from tqdm import tqdm

RAW_DATA_DIR = 'raw_data'
TABULAR_DATA_DIR = 'tabular_data'
os.makedirs(TABULAR_DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(TABULAR_DATA_DIR, 'covers_metadata.csv')

# Regex for date extraction (e.g., Jun 1, 2025)
DATE_REGEX = re.compile(r'(\b\w{3,9} \d{1,2}, \d{4}\b)')

def extract_date(text):
    match = DATE_REGEX.search(text)
    if match:
        from datetime import datetime
        try:
            dt = datetime.strptime(match.group(1), '%b %d, %Y')
            return dt.strftime('%Y_%m_%d')
        except Exception:
            pass
    return None

def extract_metadata(text):
    meta = {}
    patterns = {
        'Volume': r'^Volume:?\s*([\w\d]+)',
        'Issue': r'^Issue:?\s*(\d+)',  # Only match at start of line and only digits
        'Publication Year': r'Publication (?:year|info)[:\s]*([\d]{4})',
        'Editor': r'Editor:?\s*([^\(\;\n]+)\s*\(\d',
        'Caption': r'Caption:?\s*([\s\S]*?)(?:Retail information:|Credits:|Photographer|Editor:|Section editor:|Volume:|Issue:|Pages:|Publication year:|$)',
        'Retail information': r'Retail information:?\s*([\s\S]*?)(?:Credits:|Photographer|Editor:|Section editor:|Volume:|Issue:|Pages:|Publication year:|$)',
        'Photographer': r'(?:Photographer/illustrator|Photographer):?\s*([^\.;\n]+)',
    }
    for field, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if field == 'Issue':
            meta[field] = m.group(1).strip() if m and m.group(1).isdigit() else ''
        else:
            meta[field] = m.group(1).replace('\n', ' ').strip() if m else ''
    return meta

def process_pdf(pdf_path, metadata_rows, max_pages=None, debug_pages=0):
    with pdfplumber.open(pdf_path) as plumber_doc:
        all_text = ''
        num_pages = len(plumber_doc.pages) if max_pages is None else min(len(plumber_doc.pages), max_pages)
        for i in range(num_pages):
            page = plumber_doc.pages[i]
            text = page.extract_text() or ''
            all_text += '\n' + text
        # Split into blocks by 'document x of x'
        blocks = re.split(r'document \d+ of \d+', all_text, flags=re.IGNORECASE)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            date_str = extract_date(block)
            meta = extract_metadata(block)
            meta['date_column'] = date_str if date_str else ''
            metadata_rows.append(meta)

def main():
    metadata_rows = []
    pdf_files = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith('.pdf') and f != 'sample.pdf']
    for fname in tqdm(pdf_files, desc='PDF Files'):
        pdf_path = os.path.join(RAW_DATA_DIR, fname)
        process_pdf(pdf_path, metadata_rows)
    df = pd.DataFrame(metadata_rows)
    df.to_csv(CSV_FILE, index=False)
    print(f'Metadata saved to {CSV_FILE}')

if __name__ == '__main__':
    main() 