import pandas as pd
import requests
import os
import PyPDF2
from io import BytesIO

def download_pdf(url, save_path):
    """Downloads a PDF from a given URL and saves it locally."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def process_csv(csv_path):
    """Reads the CSV file and processes each PDF link."""
    df = pd.read_csv(csv_path)
    if 'pdf_link' not in df.columns:
        print("CSV does not contain a 'pdf_link' column.")
        return
    
    for index, row in df.iterrows():
        pdf_url = row['pdf_link']
        pdf_filename = f"paper_{index}.pdf"
        
        if download_pdf(pdf_url, pdf_filename):
            text = extract_text_from_pdf(pdf_filename)
            print(f"Extracted {len(text)} characters from {pdf_filename}")
            os.remove(pdf_filename)  # Delete after processing
        
        # Limit processing to a few PDFs for testing
        if index >= 2:
            break

# Run the script with your CSV file
csv_file_path = "/mnt/data/third_arxiv_dump.csv"
process_csv(csv_file_path)
