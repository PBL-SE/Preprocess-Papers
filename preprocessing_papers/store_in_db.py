import psycopg2
import requests
import fitz  # PyMuPDF
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration
import re
import time
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL Connection URL
DATABASE_URL = os.getenv("DATABASE_URL")

MAX_INPUT_LENGTH = 16000  # Slightly below 16384 to avoid errors

# Load LED Model
model_name = "allenai/led-base-16384"
tokenizer = LEDTokenizer.from_pretrained(model_name)
model = LEDForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

def extract_text_from_pdf(pdf_url):
    """Downloads and extracts text from a PDF."""
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        doc = fitz.open(stream=response.content, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in doc)
        return text.strip() if text.strip() else None  # Return None if text is empty
    except Exception as e:
        print(f"Error fetching PDF: {e}")
        return None

def clean_text(text):
    """Removes non-printable characters and excess whitespace."""
    text = re.sub(r"[^\x20-\x7E]", "", text)  # Keep only printable ASCII
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

def summarize_text(text):
    """Summarizes scientific text using LED-Base-16384."""
    trimmed_text = text[:MAX_INPUT_LENGTH]  # Truncate to safe length
    inputs = tokenizer(trimmed_text, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True)
    inputs = {key: value.to("cuda" if torch.cuda.is_available() else "cpu") for key, value in inputs.items()}

    summary_ids = model.generate(inputs["input_ids"], max_length=500, min_length=100, length_penalty=1.0)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def update_paper_summaries():
    """Fetches papers without summaries, generates summaries and full text, and updates the database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        print('Starting job...')
        i = 0

        # Fetch papers where summary or full_text is NULL
        cursor.execute("SELECT arxiv_id, pdf_link FROM Papers WHERE summary IS NULL OR full_text IS NULL")
        papers = cursor.fetchall()

        for arxiv_id, pdf_link in papers:
            if not pdf_link:
                print(f"Skipping {arxiv_id}: No PDF link")
                continue

            print(f"Processing {arxiv_id}...")

            # Extract and clean text
            paper_text = extract_text_from_pdf(pdf_link)
            if paper_text is None:  # Check if no text was extracted
                print(f"Skipping {arxiv_id}: Could not extract text")
                continue

            paper_text = clean_text(paper_text)

            # Generate summary using Gemini (or your LLM)
            summary = summarize_text(paper_text)

            # Update the database
            cursor.execute("""
                UPDATE Papers
                SET summary = %s, full_text = %s
                WHERE arxiv_id = %s
            """, (summary, paper_text, arxiv_id))
            conn.commit()
            print(f"Updated {arxiv_id} Number {i}")
            i += 1

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

def handle_unavailable_summaries():
    """Checks and handles papers with unavailable summaries."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Count papers with 'Summary unavailable.'
        cursor.execute("SELECT COUNT(arxiv_id) FROM Papers WHERE summary='Summary unavailable.';")
        count = cursor.fetchone()[0]

        if count > 0:
            print(f"{count} papers have 'Summary unavailable.'")
            # Update papers with 'Summary unavailable.' to set summary to NULL
            cursor.execute("""
                UPDATE Papers
                SET summary = NULL
                WHERE summary = 'Summary unavailable.'
            """)
            conn.commit()
            print("Updated papers with 'Summary unavailable.'")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Database error during summary handling: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run(iterations=10, cooldown=120):
    """Runs the summarization process for a given number of iterations, then handles summaries."""
    try:
        for i in range(iterations):
            logging.info(f"Iteration {i + 1}...")
            update_paper_summaries()

        # Handle the unavailable summaries after defined iterations
        handle_unavailable_summaries()

        # Cooldown period before restarting the process
        logging.info(f"Cooldown for {cooldown // 60} minutes...")
        time.sleep(cooldown)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    while True:
        run()
