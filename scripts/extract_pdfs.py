"""
PDF Text Extraction Script for BiodiversityASSET Project

This script extracts text from PDF files and splits it into paragraphs.
Input: PDF files in data/raw/pdfs/
Output: CSV files with extracted paragraphs in data/processed/extracted_paragraphs_from_pdfs/

The script processes PDFs in chunks of 20 to manage memory usage and creates
separate CSV files for each chunk.
"""

import os
import pandas as pd
import re
from pathlib import Path
from pdfminer.high_level import extract_text

# Define project directories using relative paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from scripts/ to project root
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_paragraphs_from_pdfs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_text(text):
    """Remove control characters from text"""
    return re.sub(r'[\000-\010]|[\013-\014]|[\016-\037]', '', text)

def merge_paragraphs(text_series):
    """Merge text segments into complete paragraphs based on sentence endings"""
    merged_paragraphs = []
    temp_paragraph = ""
    for text in text_series:
        if temp_paragraph:
            temp_paragraph += " " + text
        else:
            temp_paragraph = text
        if text.endswith(".") or text.endswith("!") or text.endswith("?"):
            merged_paragraphs.append(temp_paragraph.strip())
            temp_paragraph = ""
    if temp_paragraph:
        merged_paragraphs.append(temp_paragraph.strip())
    return merged_paragraphs

def process_with_pdfminer(file_path):
    """Extract text from PDF using pdfminer"""
    try:
        text = extract_text(file_path)
        text = sanitize_text(text)
        return text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""

def split_into_paragraphs(text):
    """Split text into paragraphs"""
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def main():
    """Main function to run the PDF extraction process"""
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    if not INPUT_DIR.exists():
        print(f"Error: Input directory {INPUT_DIR} does not exist!")
        print("Please place PDF files in the data/raw/pdfs/ directory.")
        return
    
    # Get all PDF files recursively from the input directory and its subfolders
    all_files = []
    for pdf_file in INPUT_DIR.rglob("*.pdf"):
        # Get relative path from INPUT_DIR to maintain folder structure info
        relative_path = pdf_file.relative_to(INPUT_DIR)
        all_files.append({
            'file_path': pdf_file,
            'relative_path': str(relative_path),
            'filename': pdf_file.name
        })
    
    if not all_files:
        print(f"No PDF files found in {INPUT_DIR} or its subfolders")
        return
    
    print(f"Found {len(all_files)} PDF files to process (including subfolders)")
    
    # Process in chunks of 20
    chunk_size = 20
    for chunk_idx, chunk_start in enumerate(range(0, len(all_files), chunk_size), start=1):
        chunk_files = all_files[chunk_start:chunk_start+chunk_size]
        print(f"Processing chunk {chunk_idx} with {len(chunk_files)} files")

        # Create a list to store all paragraph data for this chunk
        chunk_data = []

        for file_info in chunk_files:
            file_path = file_info['file_path']
            relative_path = file_info['relative_path']
            filename = file_info['filename']
            
            print(f"Processing file: {relative_path}")

            text_pdfminer = process_with_pdfminer(file_path)
            split_text = split_into_paragraphs(text_pdfminer)
            merged_text = merge_paragraphs(split_text)

            for i, paragraph in enumerate(merged_text, start=1):
                chunk_data.append({
                    'pdf_file_name': filename,
                    'paragraph_id': i,
                    'paragraph_text': paragraph
                })

        # Create DataFrame and save as CSV
        df = pd.DataFrame(chunk_data)
        output_csv = OUTPUT_DIR / f"chunk_{chunk_idx}.csv"
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Finished processing chunk {chunk_idx}, saved to {output_csv}")
    
    print(f"\nProcessing complete! Processed {len(all_files)} PDF files in {chunk_idx} chunks.")
    print(f"Output files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()