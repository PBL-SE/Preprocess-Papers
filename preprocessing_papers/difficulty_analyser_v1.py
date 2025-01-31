import pandas as pd
import torch
import fitz  # PyMuPDF for better text extraction
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import requests
from io import BytesIO
import concurrent.futures
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def download_and_extract_text(pdf_url):
    """Downloads a PDF and extracts text using PyMuPDF (fitz)."""
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()

        with BytesIO(response.content) as pdf_buffer:
            doc = fitz.open(stream=pdf_buffer, filetype="pdf")
            text = "\n".join([page.get_text("text") for page in doc if page.get_text("text")])
        return text.strip()
    except Exception as e:
        print(f"[ERROR] PDF Extraction Failed: {pdf_url} - {e}")
        return ""

def classify_paper(text, model, tokenizer):
    """Classifies a paper's readability level using SciBERT."""
    if not text.strip():
        return "Intermediate"  # Default level if no text is available

    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).item()

    labels = {0: "Beginner", 1: "Intermediate", 2: "Pro", 3: "Expert"}
    return labels[prediction]

def process_batch(batch, model_name, output_path):
    """Processes a batch of papers, classifies readability, and saves results."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)

    difficulty_levels = []

    for index, row in batch.iterrows():
        pdf_url = row['PDF Link']
        title = row['Title']
        summary = str(row.get('Summary', "")).strip()
        comment = str(row.get('Comment', "")).strip()

        print(f"[INFO] Processing: {title}")

        # Extract text from PDF or use fallback
        text = download_and_extract_text(pdf_url)
        if not text:
            print(f"[WARNING] Using Summary/Comment for {title}")
            text = summary if summary else comment

        # Classify based on extracted/fallback text
        category = classify_paper(text, model, tokenizer)
        difficulty_levels.append(category)

    # Save results
    batch['difficulty_level'] = difficulty_levels
    batch.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    print(f"[INFO] Batch processed and saved to {output_path}")

def process_csv_in_batches(csv_path, output_path, batch_size=50):
    """Processes the CSV in batches using parallel execution."""
    df = pd.read_csv(csv_path)
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

    model_name = "allenai/scibert_scivocab_uncased"

    # Using ThreadPoolExecutor to avoid multiprocessing model-sharing issues
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_batch, batch, model_name, output_path) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensures exceptions are raised

    print(f"✅ All batches processed. Results saved to {output_path}")

# Example usage
csv_file_path = "output/third_arxiv_dump.csv"
output_csv_path = "output/updated_arxiv_dump.csv"
process_csv_in_batches(csv_file_path, output_csv_path, batch_size=50)
