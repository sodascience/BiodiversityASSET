"""
Submit Batch Job for BiodiversityASSET Project

This script handles the first step of the batch processing workflow:
1. Combines input CSV files
2. Prepares the batch JSONL file
3. Submits the job to OpenAI Batch API
4. Returns the batch ID for later status checking

Usage:
    python scripts/submit_batch_job.py --task investment_activity_classification [options]
    python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id <prerequisite_batch_id> [options]

For assetization_features_scoring, you must specify the batch ID of a completed investment_activity_classification job.

The script outputs the batch ID which can be used with check_batch_status.py and download_batch_results.py
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path
from dotenv import load_dotenv
from batch_api_utils import (
    BatchAPIProcessor, 
    InvestmentActivityResponse,
    AssetizationFeaturesResponse, 
    get_response_schema
)

# Define project directories using relative paths
PROJECT_ROOT = Path(__file__).parent.parent
BATCH_METADATA_DIR = PROJECT_ROOT / "results" / "batch_jobs"

# Create metadata directory
BATCH_METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv()

def load_prompt_from_file(prompt_file: Path) -> str:
    """
    Load a prompt from a text file.
    
    Args:
        prompt_file: Path to the prompt file
        
    Returns:
        Prompt content as string
    """
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            raise ValueError(f"Prompt file is empty: {prompt_file}")
        
        return content
    except Exception as e:
        raise RuntimeError(f"Error reading prompt file {prompt_file}: {e}")

def get_default_prompt_files(task: str) -> tuple[Path, Path]:
    """
    Get default prompt file paths for a given task.
    
    Args:
        task: Task type (investment_activity_classification or assetization_features_scoring)

    Returns:
        Tuple of (system_prompt_file, user_prompt_file)
    """
    prompts_dir = PROJECT_ROOT / "prompts"
    
    if task == "investment_activity_classification":
        system_file = prompts_dir / f"{task}_system_prompt.txt"
    elif task == "assetization_features_scoring":
        system_file = prompts_dir / f"{task}_system_prompt.txt"
    else:
        raise ValueError(f"Unknown task: {task}")
    
    user_file = prompts_dir / "user_prompt_template.txt"
    
    return system_file, user_file

def get_prompts_from_files(system_prompt_file: Path, user_prompt_file: Path) -> tuple[str, str]:
    """
    Load prompts from specified files.
    
    Args:
        system_prompt_file: Path to system prompt file
        user_prompt_file: Path to user prompt template file
        
    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    system_prompt = load_prompt_from_file(system_prompt_file)
    user_prompt_template = load_prompt_from_file(user_prompt_file)
    
    return system_prompt, user_prompt_template

def combine_csv_files(csv_files: list[Path], task: str = None) -> pd.DataFrame:
    """
    Combine multiple CSV files into a single DataFrame.
    
    Args:
        csv_files: List of CSV file paths
        task: Task type for applying task-specific filtering
        
    Returns:
        Combined DataFrame with source file information
    """
    combined_data = []
    
    for csv_file in csv_files:
        try:
            print(f"Reading {csv_file.name}...")
            data = pd.read_csv(csv_file)
            
            # Validate required columns
            if "paragraph_text" not in data.columns:
                print(f"Warning: 'paragraph_text' column not found in {csv_file.name}")
                continue
            
            # Filter out empty paragraphs
            data = data.dropna(subset=["paragraph_text"]).copy()
            
            # Apply task-specific filtering
            if task == "assetization_features_scoring":
                # Only process paragraphs that were classified as investment activity (score == 1)
                if "score" in data.columns:
                    initial_count = len(data)
                    data = data[data["score"] == 1].copy()
                    filtered_count = len(data)
                    print(f"  Filtered to {filtered_count} paragraphs with investment activity (score=1) from {initial_count} total")
                else:
                    print(f"Warning: 'score' column not found in {csv_file.name} - cannot filter by investment activity")
            
            if len(data) == 0:
                print(f"No valid paragraphs found in {csv_file.name}")
                continue
            
            # Add source file information
            data["source_csv_file"] = csv_file.name
            
            combined_data.append(data)
            print(f"✓ Added {len(data)} paragraphs from {csv_file.name}")
            
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue
    
    if not combined_data:
        print("No valid data found in any CSV files")
        return pd.DataFrame()
    
    # Combine all DataFrames
    combined_df = pd.concat(combined_data, ignore_index=True)
    print(f"\n✓ Combined {len(combined_df)} total paragraphs from {len(combined_data)} files")
    
    return combined_df

def save_batch_metadata(batch_id: str, task: str, args: argparse.Namespace, input_files: list[Path], data_info: dict) -> Path:
    """
    Save batch job metadata for later use.
    
    Args:
        batch_id: The batch job ID
        task: Task type (investment_activity or assetization_features)
        args: Command line arguments
        input_files: List of input CSV files processed
        data_info: Information about the processed data
        
    Returns:
        Path to the metadata file
    """
    metadata = {
        "batch_id": batch_id,
        "task": task,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "input_files": [str(f) for f in input_files],
        "total_paragraphs": data_info.get("total_paragraphs", 0),
        "job_name": f"{task}_{batch_id[6:]}",
        "submitted_at": pd.Timestamp.now().isoformat(),
        "status": "submitted",
        "system_prompt_file": str(args.system_prompt) if hasattr(args, 'system_prompt') and args.system_prompt else None,
        "user_prompt_file": str(args.user_prompt) if hasattr(args, 'user_prompt') and args.user_prompt else None
    }
    
    # Add prerequisite batch_id for assetization_features_scoring
    if task == "assetization_features_scoring" and hasattr(args, 'batch_id') and args.batch_id:
        metadata["prerequisite_batch_id"] = args.batch_id
    
    metadata_file = BATCH_METADATA_DIR / f"{batch_id}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Batch metadata saved to: {metadata_file}")
    return metadata_file

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

def validate_prerequisite_batch(batch_id: str, required_task: str) -> bool:
    """
    Validate that a prerequisite batch exists and is completed.
    
    Args:
        batch_id: Batch ID to validate
        required_task: Required task type for the batch
        
    Returns:
        True if batch is valid and completed, False otherwise
    """
    metadata = load_batch_metadata(batch_id)
    
    if not metadata:
        print(f"❌ No metadata found for batch ID: {batch_id}")
        return False
    
    # Check if it's the correct task type
    if metadata.get("task") != required_task:
        print(f"❌ Batch {batch_id} is for task '{metadata.get('task')}', but '{required_task}' is required")
        return False
    
    # Check if batch is completed
    if metadata.get("status") not in ["completed", "processed"]:
        print(f"❌ Batch {batch_id} is not completed. Current status: {metadata.get('status', 'unknown')}")
        print("Please wait for the batch to complete before using it as input for assetization features scoring")
        return False
    
    print(f"✓ Prerequisite batch {batch_id} is valid and completed")
    return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Submit batch job for BiodiversityASSET processing")
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        choices=["investment_activity_classification", "assetization_features_scoring"],
        help="Task type to process"
    )
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        help="Input directory containing CSV files (auto-detected if not provided)"
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        help="Batch ID of completed investment_activity_classification job (required for assetization_features_scoring)"
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        help="Path to system prompt file (default: auto-selected based on task)"
    )
    parser.add_argument(
        "--user-prompt",
        type=Path,
        help="Path to user prompt template file (default: prompts/user_prompt_template.txt)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model to use (default: gpt-4o-mini or from OPENAI_MODEL env var)"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=500,
        help="Maximum tokens for response (default: 500)"
    )
    return parser.parse_args()

def main():
    """Main function to submit the batch job"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate batch_id requirement for assetization_features_scoring
    if args.task == "assetization_features_scoring":
        if not args.batch_id:
            print("❌ Error: --batch-id is required for assetization_features_scoring task")
            print("Please provide the batch ID of a completed investment_activity_classification job")
            print("Example: python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id batch_123456789")
            return
        
        # Validate the prerequisite batch
        if not validate_prerequisite_batch(args.batch_id, "investment_activity_classification"):
            return
    
    # Determine input directory based on task
    if args.input_dir:
        input_dir = args.input_dir
    else:
        if args.task == "investment_activity_classification":
            input_dir = PROJECT_ROOT / "data" / "processed" / "biodiversity_related_paragraphs"
        elif args.task == "assetization_features_scoring":
            # Use the results from the specified batch
            input_dir = PROJECT_ROOT / "data" / "processed" / "investment_activity_classification" / args.batch_id
        else:
            print(f"Unknown task: {args.task}")
            return
    
    # Determine prompt files
    if args.system_prompt or args.user_prompt:
        # Use custom prompt files if provided
        if args.system_prompt:
            system_prompt_file = args.system_prompt
        else:
            # Use default system prompt for the task
            system_prompt_file, _ = get_default_prompt_files(args.task)
        
        if args.user_prompt:
            user_prompt_file = args.user_prompt
        else:
            # Use default user prompt
            _, user_prompt_file = get_default_prompt_files(args.task)
    else:
        # Use default prompt files for the task
        system_prompt_file, user_prompt_file = get_default_prompt_files(args.task)
    
    print(f"=== BiodiversityASSET Batch Job Submission ===")
    print(f"Task: {args.task}")
    if args.task == "assetization_features_scoring" and args.batch_id:
        print(f"Prerequisite batch ID: {args.batch_id}")
    print(f"Input directory: {input_dir}")
    print(f"System prompt file: {system_prompt_file}")
    print(f"User prompt file: {user_prompt_file}")
    print(f"OpenAI Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return
    
    # Validate prompt files exist
    try:
        system_prompt, user_prompt_template = get_prompts_from_files(system_prompt_file, user_prompt_file)
        print(f"✓ Successfully loaded prompts from files")
    except Exception as e:
        print(f"Error loading prompt files: {e}")
        return
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return
    
    # Initialize batch processor
    try:
        processor = BatchAPIProcessor()
        print("✓ OpenAI Batch API processor initialized")
    except Exception as e:
        print(f"Error initializing batch processor: {e}")
        return
    
    # Get all CSV files from the input directory
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to combine and process")
    
    # Combine all CSV files into a single DataFrame
    combined_data = combine_csv_files(csv_files, args.task)
    
    if combined_data.empty:
        print("No valid data to process")
        return
    
    print(f"\nPreparing batch job for {len(combined_data)} paragraphs...")
    
    try:
        # Get response schema based on task
        if args.task == "investment_activity_classification":
            response_schema = get_response_schema(InvestmentActivityResponse)
        elif args.task == "assetization_features_scoring":
            response_schema = get_response_schema(AssetizationFeaturesResponse)
        else:
            print(f"Unknown task: {args.task}")
            return
        
        # Create output directory for this specific job
        job_output_dir = PROJECT_ROOT / "results" / "batch_jobs" / f"{args.task}_processing"
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare batch file
        job_name = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        input_file = job_output_dir / f"{args.task}_{job_name}_input.jsonl"
        
        processor.prepare_batch_file(
            data=combined_data,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            text_column="paragraph_text",
            response_schema=response_schema,
            model=args.model,
            output_path=input_file,
            max_tokens=args.max_tokens,
            custom_id_prefix=job_name
        )
        
        # Submit batch job
        batch_id = processor.submit_batch(input_file, f"BiodiversityASSET {args.task}")
        
        # Save metadata for later use
        data_info = {
            "total_paragraphs": len(combined_data),
            "input_files_count": len(csv_files)
        }
        
        metadata_file = save_batch_metadata(batch_id, args.task, args, csv_files, data_info)
        
        print(f"\n🚀 Batch job submitted successfully!")
        print(f"📋 Batch ID: {batch_id}")
        print(f"📁 Job files saved in: {job_output_dir}")
        print(f"📄 Metadata saved to: {metadata_file}")
        print(f"\nNext steps:")
        print(f"1. Check job status: python scripts/check_batch_status.py --batch-id {batch_id}")
        print(f"2. Download results when complete: python scripts/download_batch_results.py --batch-id {batch_id}")
        
    except Exception as e:
        print(f"Error during batch submission: {e}")
        return

if __name__ == "__main__":
    main()
