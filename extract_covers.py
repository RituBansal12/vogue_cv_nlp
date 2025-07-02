import os
import re
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm

RAW_DATA_DIR = 'raw_data'
COVERS_DIR = 'covers'

os.makedirs(COVERS_DIR, exist_ok=True)

# Regex for date extraction (e.g., Jan 1, 2020)
DATE_REGEX = re.compile(r'(\b\w{3,9} \d{1,2}, \d{4}\b)')

def extract_date(text):
    match = DATE_REGEX.search(text)
    if match:
        try:
            from datetime import datetime
            dt = datetime.strptime(match.group(1), '%b %d, %Y')
            return dt.strftime('%Y_%m_%d')
        except Exception:
            pass
    return None

def save_image(img_bytes, out_path):
    img = Image.open(BytesIO(img_bytes))
    img = img.convert('RGB')
    img.save(out_path, 'JPEG', quality=90)

def process_pdf(pdf_path, metadata_rows, max_pages=None):
    with fitz.open(pdf_path) as doc:
        with pdfplumber.open(pdf_path) as plumber_doc:
            num_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
            all_text = ''
            # Track all dates and their page numbers
            date_by_page = {}
            for i in range(num_pages):
                plumber_page = plumber_doc.pages[i] if i < len(plumber_doc.pages) else None
                text = plumber_page.extract_text() if plumber_page else ''
                all_text += '\n' + text
                date_str = extract_date(text)
                if date_str:
                    date_by_page[i] = date_str
            # For each page, extract images and associate with closest previous date
            img_found_pages = set()
            for i in tqdm(range(num_pages), desc=f'Pages ({os.path.basename(pdf_path)})'):
                page = doc[i]
                # Find the closest previous date (or look ahead if none found yet)
                date_str = None
                for j in range(i, -1, -1):
                    if j in date_by_page:
                        date_str = date_by_page[j]
                        break
                if not date_str:
                    for j in range(i+1, num_pages):
                        if j in date_by_page:
                            date_str = date_by_page[j]
                            break
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image['image']
                    if date_str:
                        out_path = os.path.join(COVERS_DIR, f'cover_{date_str}.jpg')
                    else:
                        out_path = os.path.join(COVERS_DIR, f'cover_unknown_page{i+1}_img{img_index+1}.jpg')
                        tqdm.write(f'  Warning: No date found for image on page {i+1} in {os.path.basename(pdf_path)}.')
                    if not os.path.exists(out_path):
                        save_image(img_bytes, out_path)
                        img_found_pages.add(i)
            if not img_found_pages:
                tqdm.write(f'  No images found in {os.path.basename(pdf_path)}.')

def main():
    metadata_rows = []
    pdf_files = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith('.pdf')]
    for fname in tqdm(pdf_files, desc='PDF Files'):
        pdf_path = os.path.join(RAW_DATA_DIR, fname)
        process_pdf(pdf_path, metadata_rows)

if __name__ == '__main__':
    main() 