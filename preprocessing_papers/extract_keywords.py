import pandas as pd
import torch
import fitz  # PyMuPDF for better text extraction
from transformers import AutoModel, AutoTokenizer
import requests
from io import BytesIO
import logging
import os
import yake
from pinecone import Pinecone
import numpy as np

# 🔹 Logging Setup
logging.basicConfig(filename="processing.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# 🔹 Pinecone API Setup
PINECONE_API_KEY = "pcsk_5FL492_g2vJnmKKbX52zVcv6yvK7UoeWEbiW2V7FwusT7D6iRB8mVxPCg4itupD4epKmk"
PINECONE_INDEX_NAME = "paper-search"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
    
# List available indexes to check if the index exists
available_indexes = pc.list_indexes()
print('list indexes: ', pc.list_indexes())

index = pc.Index(PINECONE_INDEX_NAME)
print('index: ', index)
print(type(index))

if(index):
    print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")


# 🔹 Device Setup (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Load BERT Tokenizer & Model for Embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# 🔹 YAKE Keyword Extractor Setup
kw_extractor = yake.KeywordExtractor(n=3, top=10)

def extract_text_from_pdf(pdf_url):
    """Downloads and extracts text from a PDF."""
    try:
        print(f"📥 Downloading PDF: {pdf_url}")
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        with BytesIO(response.content) as pdf_buffer:
            doc = fitz.open(stream=pdf_buffer, filetype="pdf")
            text = "\n".join([page.get_text("text") for page in doc if page.get_text("text")])
        return text.strip()
    except Exception as e:
        logging.error(f"❌ Error extracting text from {pdf_url}: {e}")
        return ""

def get_keywords(text):
    """Extracts top keywords from text using YAKE."""
    return [kw[0] for kw in kw_extractor.extract_keywords(text)] if text else []

def get_bert_embedding(text):
    """Generates BERT embeddings for text."""
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Use CLS token ([0, :]) as the sentence embedding
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)

    return embedding.tolist()  # Convert NumPy array to Python list

def process_batch(batch):
    """Processes a batch of papers and stores embeddings in Pinecone."""
    pinecone_vectors = []
    
    for i, row in batch.iterrows():
        title = row['Title']
        pdf_url = row['PDF Link']
        summary = row.get('Summary', "").strip()
        comment = str(row.get('Comment', "")).strip()

        print(f"🔍 Processing: {title}")
        logging.info(f"Processing: {title}")

        # Extract text from PDF (or fallback to summary/comment)
        text = extract_text_from_pdf(pdf_url)
        if not text:
            text = summary if summary else comment
        
        # Extract keywords
        keywords = get_keywords(text)
        keyword_str = ", ".join(keywords)

        # Generate BERT embeddings
        embedding = get_bert_embedding(text)

        # Store in Pinecone
        pinecone_vectors.append({
            "id": str(row["ArXiv ID"]),
            "values": embedding,  # Ensure it's a list of floats
            "metadata": {
                "title": title, 
                "keywords": keyword_str, 
                "pdf_url": pdf_url
            }
        })

    print(f"📝 Storing {len(pinecone_vectors)} papers in Pinecone...")
    # Upsert batch into Pinecone
    if pinecone_vectors:
        index.upsert(vectors=pinecone_vectors, namespace="ns1")
        print(index.describe_index_stats())
        logging.info(f"✅ Batch of {len(pinecone_vectors)} papers stored in Pinecone.")

def process_csv(csv_path, batch_size=20):
    """Reads CSV, processes in batches, and stores embeddings in Pinecone."""
    df = pd.read_csv(csv_path)
    
    print(f"📂 Loaded CSV with {len(df)} papers. Processing in batches of {batch_size}...")
    
    for i in range(100, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        process_batch(batch)

    print("🎉 All papers processed and stored in Pinecone.")

# 🔹 Run Pipeline
csv_file_path = "output/third_arxiv_dump.csv"
process_csv(csv_file_path, batch_size=20)
