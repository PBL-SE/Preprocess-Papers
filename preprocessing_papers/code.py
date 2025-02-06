import pandas as pd
import requests
import os
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from huggingface_hub import snapshot_download

# Replace with your Hugging Face repo ID
HF_REPO_ID = "shravani-10/fine_tuned_scibert"

# Download full model snapshot (handles Git LFS files)
model_dir = snapshot_download(repo_id=HF_REPO_ID)

# Load tokenizer and model from downloaded directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Define classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define label mapping (ensure these match your trained model)
labels = ["Beginner", "Intermediate", "Pro", "Expert"]

# Directory to store PDFs
os.makedirs("pdfs", exist_ok=True)

def download_pdf(pdf_url, save_path):
    """Downloads a PDF from the given URL and saves it."""
    try:
        response = requests.get(pdf_url, timeout=15)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    except Exception as e:
        print(f"❌ Failed to download {pdf_url}: {e}")
        return None

def extract_text_from_pdf_pymupdf(pdf_path):
    """Extracts text from a given PDF using PyMuPDF (fitz)."""
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text("text") for page in doc)
        if not text.strip():
            raise ValueError("Extracted text is empty.")
        return text
    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path} using PyMuPDF: {e}")
        return None

def chunk_text(text, tokenizer, max_length=512):
    """Splits long text into 512-token chunks."""
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    return [tokenizer.decode(tokens[i: i + max_length], skip_special_tokens=True) for i in range(0, len(tokens), max_length)]

def classify_paper(pdf_url):
    """Processes a research paper: downloads, extracts, classifies, and assigns difficulty."""
    pdf_filename = pdf_url.split("/")[-1]
    pdf_path = f"pdfs/{pdf_filename}.pdf"

    if not download_pdf(pdf_url, pdf_path):
        return "Unknown"

    text = extract_text_from_pdf_pymupdf(pdf_path)
    if not text:
        return "Unknown"

    chunks = chunk_text(text, tokenizer)
    difficulties = []
    for chunk in chunks:
        prediction = classifier(chunk, truncation=True, padding=True, max_length=512)[0]
        label = prediction["label"]
        difficulties.append(labels.index(label) if label in labels else 0)

    if not difficulties:
        return "Unknown"

    avg_difficulty = np.mean(difficulties)
    final_label = labels[int(round(avg_difficulty))]
    print("Processing paper...")
    return final_label

# Load dataset
df = pd.read_csv("third_arxiv_dump.csv")
if "PDF Link" not in df.columns:
    raise KeyError("❌ Column 'PDF Link' not found in dataset.")

# Process in batches of 100 papers each
batch_size = 100
output_file = "classified_papers.csv"

for batch_num, start_idx in enumerate(range(0, len(df), batch_size)):
    end_idx = min(start_idx + batch_size, len(df))
    df_batch = df.iloc[start_idx:end_idx].copy()
    print(f"Processing batch {batch_num + 1} ({start_idx} to {end_idx})...")
    
    df_batch["predicted_difficulty"] = df_batch["PDF Link"].apply(lambda url: classify_paper(url) if isinstance(url, str) else "Unknown")
    
    df_batch.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    print(f"✅ Batch {batch_num + 1} appended to '{output_file}'")

print("✅ Classification complete for all batches!")
