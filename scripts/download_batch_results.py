"""
Download and Process Batch Results for BiodiversityASSET Project

This script handles the final step of the batch processing workflow:
1. Download results from a completed batch job
2. Process and merge results with original data
3. Apply task-specific filtering and formatting
4. Save final results to appropriate output directories

Usage:
    python scripts/download_batch_results.py --batch-id <batch_id>
    python scripts/download_batch_results.py --batch-id <batch_id> --output-dir custom/path
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path
from dotenv import load_dotenv
from batch_api_utils import BatchAPIProcessor

# Define project directories
PROJECT_ROOT = Path(__file__).parent.parent
BATCH_METADATA_DIR = PROJECT_ROOT / "results" / "batch_jobs"

# Load environment variables
load_dotenv()

def load_batch_metadata(batch_id: str) -> dict:
    """
    Load batch metadata from saved file.
    
    Args:
        batch_id: Batch ID to load metadata for
        
    Returns:
        Metadata dictionary or None if not found
    """
    metadata_file = BATCH_METADATA_DIR / f"{batch_id}.json"
    
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata for batch {batch_id}: {e}")
        return None

def update_batch_metadata(batch_id: str, updates: dict):
    """
    Update batch metadata file.
    
    Args:
        batch_id: Batch ID to update
        updates: Dictionary of updates to apply
    """
    metadata_file = BATCH_METADATA_DIR / f"{batch_id}.json"
    
    if not metadata_file.exists():
        print(f"Warning: Metadata file not found for batch {batch_id}")
        return
    
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        metadata.update(updates)
        metadata["last_updated"] = pd.Timestamp.now().isoformat()
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        print(f"Error updating metadata for batch {batch_id}: {e}")

def reconstruct_original_data(metadata: dict) -> pd.DataFrame:
    """
    Reconstruct the original data that was submitted for batch processing.
    
    Args:
        metadata: Batch metadata containing input file information
        
    Returns:
        DataFrame with the original data
    """
    input_files = metadata.get("input_files", [])
    task = metadata.get("task", "unknown")
    
    if not input_files:
        print("Warning: No input files found in metadata")
        return pd.DataFrame()
    
    combined_data = []
    
    print(f"Reconstructing original data from {len(input_files)} files...")
    
    for file_path_str in input_files:
        file_path = Path(file_path_str)
        
        if not file_path.exists():
            print(f"Warning: Input file not found: {file_path}")
            continue
        
        try:
            data = pd.read_csv(file_path)
            
            # Validate required columns
            if "paragraph_text" not in data.columns:
                print(f"Warning: 'paragraph_text' column not found in {file_path.name}")
                continue
            
            # Filter out empty paragraphs (same as done during submission)
            data = data.dropna(subset=["paragraph_text"]).copy()
            
            # Apply task-specific filtering (same as done during submission)
            if task == "assetization_features_scoring":
                # Only process paragraphs that were classified as investment activity (score == 1)
                if "score" in data.columns:
                    initial_count = len(data)
                    data = data[data["score"] == 1].copy()
                    filtered_count = len(data)
                    print(f"  Filtered to {filtered_count} paragraphs with investment activity (score=1) from {initial_count} total")
                else:
                    print(f"Warning: 'score' column not found in {file_path.name} - cannot filter by investment activity")
            
            if len(data) == 0:
                print(f"No valid paragraphs found in {file_path.name}")
                continue
            
            # Add source file information
            data["source_csv_file"] = file_path.name
            
            combined_data.append(data)
            print(f"✓ Reconstructed {len(data)} paragraphs from {file_path.name}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not combined_data:
        print("Could not reconstruct any original data")
        return pd.DataFrame()
    
    # Combine all DataFrames (same order as during submission)
    combined_df = pd.concat(combined_data, ignore_index=True)
    print(f"✓ Reconstructed {len(combined_df)} total paragraphs")
    
    return combined_df

def process_investment_activity_classification_results(results_df: pd.DataFrame, output_dir: Path, batch_id: str) -> Path:
    """
    Process results for investment activity classification task.
    
    Args:
        results_df: DataFrame with processed results
        output_dir: Directory to save results
        
    Returns:
        Path to the final output file
    """
    print("Processing investment activity results...")
    
    # Map the OpenAI response field names to our expected column names
    # The response schema uses "Score" and "Explanation"
    if "Score" in results_df.columns:
        results_df["score"] = results_df["Score"]
    if "Explanation" in results_df.columns:
        results_df["explanation"] = results_df["Explanation"]
    
    # Save all results (no filtering by score)
    if "score" in results_df.columns:
        all_results = results_df.copy()
        
        print(f"Processing {len(all_results)} total paragraphs")
        
        # Show score distribution
        score_counts = all_results['score'].value_counts().sort_index()
        print(f"Score distribution:")
        for score, count in score_counts.items():
            print(f"  Score {score}: {count} paragraphs")
        
        # Create final output with required columns
        final_results = []
        for _, row in all_results.iterrows():
            result_entry = {
                "pdf_file_name": row.get("pdf_file_name", "unknown"),
                "csv_file_name": row.get("source_csv_file", "unknown"),
                "paragraph_text": row["paragraph_text"],
                "score": row.get("score", None),
                "explanation": row.get("explanation", "")
            }
            final_results.append(result_entry)
        
        final_df = pd.DataFrame(final_results)
        
        # Save only individual chunk CSV files, not the combined one
        output_file = None
        if "csv_file_name" in final_df.columns:
            print(f"\nBreakdown by source file:")
            source_counts = final_df['csv_file_name'].value_counts()
            for source_file, count in source_counts.items():
                print(f"  {source_file}: {count} paragraphs")
                
                # Save individual file results
                source_results = final_df[final_df['csv_file_name'] == source_file].copy()
                # Use original filename without batch ID
                source_output = output_dir / source_file
                source_results.to_csv(source_output, index=False, encoding='utf-8')
                print(f"    ✓ Individual results saved to: {source_output}")
                
                # Return the first file path for metadata purposes
                if output_file is None:
                    output_file = source_output
        
        if output_file is None:
            # Fallback if no individual files were saved
            output_file = output_dir / "investment_activity_classification.csv"
            final_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"✅ Results saved to: {output_file}")
        
        return output_file
    else:
        print("Warning: 'score' column not found in results")
        # Save raw results as fallback
        output_file = output_dir / "investment_activity_classification_raw.csv"
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        return output_file

def process_assetization_features_scoring_results(results_df: pd.DataFrame, output_dir: Path, batch_id: str) -> Path:
    """
    Process results for assetization features scoring task.
    
    Args:
        results_df: DataFrame with processed results
        output_dir: Directory to save results
        
    Returns:
        Path to the final output file
    """
    print("Processing assetization features results...")
    
    # The assetization response includes multiple scores
    # Map response field names if needed
    score_columns = [
        "IntrinsicValueScore", "CashFlowScore", "OwnershipControlScore"
    ]
    
    explanation_columns = [
        "Explanation"
    ]
    
    # Create final output with all relevant columns
    output_columns = [
        "pdf_file_name", "csv_file_name", "paragraph_text"
    ] + score_columns + explanation_columns
    
    final_results = []
    for _, row in results_df.iterrows():
        result_entry = {
            "pdf_file_name": row.get("pdf_file_name", "unknown"),
            "csv_file_name": row.get("source_csv_file", "unknown"),
            "paragraph_text": row["paragraph_text"],
        }
        
        # Add score columns
        for col in score_columns + explanation_columns:
            result_entry[col] = row.get(col, None)
        
        final_results.append(result_entry)
    
    # Save results
    output_file = output_dir / "assetization_features_scored.csv"
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ Results saved to: {output_file}")
    
    # Show some statistics
    if "OverallAssetizationScore" in final_df.columns:
        valid_scores = final_df["OverallAssetizationScore"].dropna()
        if len(valid_scores) > 0:
            print(f"Overall Assetization Score statistics:")
            print(f"  Mean: {valid_scores.mean():.2f}")
            print(f"  Median: {valid_scores.median():.2f}")
            print(f"  Min: {valid_scores.min():.2f}")
            print(f"  Max: {valid_scores.max():.2f}")
    
    return output_file

def download_and_process_results(batch_id: str, output_dir: Path = None) -> dict:
    """
    Download and process batch results.
    
    Args:
        batch_id: Batch ID to download results for
        output_dir: Optional custom output directory
        
    Returns:
        Dictionary with processing results
    """
    # Load metadata
    metadata = load_batch_metadata(batch_id)
    
    if not metadata:
        print(f"❌ No metadata found for batch ID: {batch_id}")
        return {"status": "error", "error": "Metadata not found"}
    
    # Check if batch is completed
    if metadata.get("status") != "completed":
        print(f"❌ Batch is not completed. Current status: {metadata.get('status', 'unknown')}")
        print("Use check_batch_status.py to monitor the batch until completion")
        return {"status": "error", "error": "Batch not completed"}
    
    # Get output file ID
    output_file_id = metadata.get("output_file_id")
    if not output_file_id:
        print("❌ No output file ID found in metadata")
        return {"status": "error", "error": "No output file ID"}
    
    # Initialize batch processor
    try:
        processor = BatchAPIProcessor()
    except Exception as e:
        print(f"Error initializing batch processor: {e}")
        return {"status": "error", "error": str(e)}
    
    # Determine output directory
    if output_dir is None:
        task = metadata.get("task", "unknown")
        if task in ["investment_activity_classification", "assetization_features_scoring"]:
            base_output_dir = PROJECT_ROOT / "data" / "processed" / task
        else:
            base_output_dir = PROJECT_ROOT / "results" / "batch_results"
        
        # Create a subfolder with the batch ID
        output_dir = base_output_dir / batch_id
    else:
        # If custom output dir is provided, still create batch ID subfolder
        output_dir = Path(output_dir) / batch_id
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Downloading Batch Results ===")
    print(f"Batch ID: {batch_id}")
    print(f"Task: {metadata.get('task', 'unknown')}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Download results
        results_file = output_dir / "batch_results.jsonl"
        processor.download_results(output_file_id, results_file)
        
        # Reconstruct original data
        original_data = reconstruct_original_data(metadata)
        
        if original_data.empty:
            print("❌ Could not reconstruct original data")
            return {"status": "error", "error": "Could not reconstruct original data"}
        
        # Process results
        results_df = processor.process_batch_results(results_file, original_data)
        
        if results_df.empty:
            print("❌ No results found in batch output")
            return {"status": "error", "error": "No results found"}
        
        print(f"✓ Processed {len(results_df)} results")
        
        # Apply task-specific processing
        task = metadata.get("task", "unknown")
        
        if task == "investment_activity_classification":
            output_file = process_investment_activity_classification_results(results_df, output_dir, batch_id)
        elif task == "assetization_features_scoring":
            output_file = process_assetization_features_scoring_results(results_df, output_dir, batch_id)
        else:
            # Generic processing - just save the results
            output_file = output_dir / f"{task}_results.csv"
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"✅ Generic results saved to: {output_file}")
        
        # Update metadata
        updates = {
            "status": "processed",
            "processed_at": pd.Timestamp.now().isoformat(),
            "output_file": str(output_file),
            "results_file": str(results_file),
            "total_processed": len(results_df)
        }
        
        update_batch_metadata(batch_id, updates)
        
        print(f"\n🎉 Batch processing completed successfully!")
        print(f"📁 Final results: {output_file}")
        
        return {
            "status": "success",
            "output_file": output_file,
            "results_file": results_file,
            "total_processed": len(results_df)
        }
        
    except Exception as e:
        error_msg = f"Error downloading/processing results: {e}"
        print(f"❌ {error_msg}")
        
        # Update metadata with error
        update_batch_metadata(batch_id, {
            "status": "error",
            "error_info": str(e),
            "error_at": pd.Timestamp.now().isoformat()
        })
        
        return {"status": "error", "error": str(e)}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and process batch results for BiodiversityASSET")
    parser.add_argument(
        "--batch-id", 
        type=str, 
        required=True,
        help="Batch ID to download results for"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        help="Custom output directory (auto-detected from task if not provided)"
    )
    
    return parser.parse_args()

def main():
    """Main function to download and process batch results"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return
    
    # Download and process results
    result = download_and_process_results(args.batch_id, args.output_dir)
    
    if result.get("status") == "error":
        exit(1)

if __name__ == "__main__":
    main()
