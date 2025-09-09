"""
Check Batch Job Status for BiodiversityASSET Project

This script handles the second step of the batch processing workflow:
1. Check the status of a submitted batch job
2. Display progress information
3. Optionally wait for completion
4. Cancel batch jobs if needed

Usage:
    python scripts/check_batch_status.py --batch-id <batch_id>
    python scripts/check_batch_status.py --batch-id <batch_id> --wait
    python scripts/check_batch_status.py --batch-id <batch_id> --cancel
    python scripts/check_batch_status.py --list-jobs
"""

import os
import json
import time
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
        metadata["last_checked"] = pd.Timestamp.now().isoformat()
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        print(f"Error updating metadata for batch {batch_id}: {e}")

def list_batch_jobs():
    """List all batch jobs with their status."""
    if not BATCH_METADATA_DIR.exists():
        print("No batch jobs directory found.")
        return
    
    metadata_files = list(BATCH_METADATA_DIR.glob("*.json"))
    
    if not metadata_files:
        print("No batch jobs found.")
        return
    
    print(f"=== Batch Jobs ({len(metadata_files)} found) ===")
    print(f"{'Batch ID':<45} {'Task':<10} {'Status':<15} {'Last_Checked':<17} {'Submitted':<17} {'Paragraphs':<12}")
    print("-" * 131)
    
    for metadata_file in sorted(metadata_files):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            batch_id = metadata.get("batch_id", "unknown")  
            task = metadata.get("task", "unknown")[:10] 
            status = metadata.get("status", "unknown")[:15]
            
            # Format last checked time (to minutes)
            last_checked = metadata.get("last_status_check") or metadata.get("last_checked")
            if last_checked:
                try:
                    # Parse timestamp and format to show date and time to minutes
                    last_checked_dt = pd.Timestamp(last_checked)
                    last_checked_str = last_checked_dt.strftime("%m-%d %H:%M")
                except:
                    last_checked_str = "unknown"
            else:
                last_checked_str = "never"
            
            submitted = metadata.get("submitted_at", "unknown")[:16]  # Reduced from 19 to 16
            paragraphs = str(metadata.get("total_paragraphs", 0))
            
            print(f"{batch_id:<45} {task:<10} {status:<15} {last_checked_str:<17} {submitted:<17} {paragraphs:<12}")
            
        except Exception as e:
            print(f"Error reading {metadata_file.name}: {e}")
    
    print(f"\nNote: Status may not be current. Use --batch-id to get live status.")

def format_duration(start_time: str, end_time: str = None) -> str:
    """
    Format duration between two timestamps.
    
    Args:
        start_time: Start timestamp string
        end_time: End timestamp string (current time if None)
        
    Returns:
        Formatted duration string
    """
    try:
        start = pd.Timestamp(start_time)
        end = pd.Timestamp(end_time) if end_time else pd.Timestamp.now()
        
        duration = end - start
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    except:
        return "unknown"

def check_batch_status(batch_id: str, wait: bool = False, poll_interval: int = 30) -> dict:
    """
    Check the status of a batch job.
    
    Args:
        batch_id: Batch ID to check
        wait: Whether to wait for completion
        poll_interval: Seconds between status checks when waiting
        
    Returns:
        Dictionary with status information
    """
    # Load metadata
    metadata = load_batch_metadata(batch_id)
    
    if not metadata:
        print(f"❌ No metadata found for batch ID: {batch_id}")
        print("This might be a batch job not created with submit_batch_job.py")
        metadata = {"task": "unknown", "submitted_at": "unknown", "total_paragraphs": 0}
    
    # Initialize batch processor
    try:
        processor = BatchAPIProcessor()
    except Exception as e:
        print(f"Error initializing batch processor: {e}")
        return {"status": "error", "error": str(e)}
    
    print(f"=== Batch Job Status ===")
    print(f"Batch ID: {batch_id}")
    print(f"Task: {metadata.get('task', 'unknown')}")
    print(f"Submitted: {metadata.get('submitted_at', 'unknown')}")
    print(f"Total paragraphs: {metadata.get('total_paragraphs', 0)}")
    
    if wait:
        print(f"Waiting for completion (checking every {poll_interval}s)...")
        print("Press Ctrl+C to stop waiting")
    
    try:
        start_time = time.time()
        
        while True:
            try:
                # Get batch status from OpenAI
                batch = processor.client.batches.retrieve(batch_id)
                status = batch.status
                
                # Calculate elapsed time
                submitted_time = metadata.get('submitted_at')
                if submitted_time:
                    elapsed = format_duration(submitted_time)
                else:
                    elapsed = format_duration(pd.Timestamp.now().isoformat(), pd.Timestamp.now().isoformat())
                
                print(f"\n⌛ Status: {status.upper()}")
                print(f"⏱️  Elapsed time: {elapsed}")
                
                # Show additional info if available
                if hasattr(batch, 'request_counts') and batch.request_counts:
                    counts = batch.request_counts
                    total = counts.total if hasattr(counts, 'total') else 0
                    completed = counts.completed if hasattr(counts, 'completed') else 0
                    failed = counts.failed if hasattr(counts, 'failed') else 0
                    
                    if total > 0:
                        progress = (completed / total) * 100
                        print(f"📊 Progress: {completed}/{total} ({progress:.1f}%)")
                        if failed > 0:
                            print(f"❌ Failed requests: {failed}")
                
                # Update metadata with current status
                updates = {
                    "status": status,
                    "last_status_check": pd.Timestamp.now().isoformat()
                }
                
                if status == "completed" and hasattr(batch, 'output_file_id') and batch.output_file_id:
                    updates["output_file_id"] = batch.output_file_id
                    updates["completed_at"] = pd.Timestamp.now().isoformat()
                    print(f"✅ Batch completed! Output file ID: {batch.output_file_id}")
                    
                elif status in ["failed", "cancelled"]:
                    updates["error_info"] = f"Batch {status}"
                    if hasattr(batch, 'error_file_id') and batch.error_file_id:
                        updates["error_file_id"] = batch.error_file_id
                        print(f"❌ Batch {status}. Error file ID: {batch.error_file_id}")
                    else:
                        print(f"❌ Batch {status}. Check batch details for error information.")
                
                update_batch_metadata(batch_id, updates)
                
                # Return result if not waiting or if job is done
                if not wait or status in ["completed", "failed", "cancelled"]:
                    result = {
                        "batch_id": batch_id,
                        "status": status,
                        "elapsed_time": elapsed
                    }
                    
                    if status == "completed" and hasattr(batch, 'output_file_id'):
                        result["output_file_id"] = batch.output_file_id
                        print(f"\nNext step: python scripts/download_batch_results.py --batch-id {batch_id}")
                    
                    return result
                
                # Wait before next check
                if wait:
                    time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                print(f"\n⏹️  Stopped waiting. Current status: {status}")
                print(f"You can check again later with: python scripts/check_batch_status.py --batch-id {batch_id}")
                return {"status": status, "interrupted": True}
                
    except Exception as e:
        error_msg = f"Error checking batch status: {e}"
        print(f"❌ {error_msg}")
        
        # Update metadata with error
        update_batch_metadata(batch_id, {
            "status": "error",
            "error_info": str(e),
            "last_error_check": pd.Timestamp.now().isoformat()
        })
        
        return {"status": "error", "error": str(e)}

def cancel_batch_job(batch_id: str) -> dict:
    """
    Cancel a batch job.
    
    Args:
        batch_id: Batch ID to cancel
        
    Returns:
        Dictionary with cancellation result
    """
    # Load metadata
    metadata = load_batch_metadata(batch_id)
    
    if not metadata:
        print(f"❌ No metadata found for batch ID: {batch_id}")
        return {"status": "error", "error": "Metadata not found"}
    
    # Initialize batch processor
    try:
        processor = BatchAPIProcessor()
    except Exception as e:
        print(f"Error initializing batch processor: {e}")
        return {"status": "error", "error": str(e)}
    
    print(f"=== Cancelling Batch Job ===")
    print(f"Batch ID: {batch_id}")
    print(f"Task: {metadata.get('task', 'unknown')}")
    
    try:
        # First check current status
        batch = processor.client.batches.retrieve(batch_id)
        current_status = batch.status
        
        print(f"Current status: {current_status}")
        
        if current_status in ["completed", "failed", "cancelled"]:
            print(f"❌ Cannot cancel batch - already {current_status}")
            return {"status": "error", "error": f"Batch already {current_status}"}
        
        # Confirm cancellation
        confirm = input(f"Are you sure you want to cancel this batch job? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Cancellation aborted")
            return {"status": "aborted"}
        
        # Cancel the batch
        print("🔄 Cancelling batch job...")
        cancelled_batch = processor.client.batches.cancel(batch_id)
        
        # Update metadata
        updates = {
            "status": "cancelled",
            "cancelled_at": pd.Timestamp.now().isoformat(),
            "last_status_check": pd.Timestamp.now().isoformat()
        }
        update_batch_metadata(batch_id, updates)
        
        print(f"✅ Batch job {batch_id} has been cancelled")
        
        return {
            "status": "cancelled",
            "batch_id": batch_id,
            "cancelled_at": updates["cancelled_at"]
        }
        
    except Exception as e:
        error_msg = f"Error cancelling batch: {e}"
        print(f"❌ {error_msg}")
        
        # Update metadata with error
        update_batch_metadata(batch_id, {
            "status": "error",
            "error_info": str(e),
            "last_error_check": pd.Timestamp.now().isoformat()
        })
        
        return {"status": "error", "error": str(e)}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Check batch job status for BiodiversityASSET processing")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--batch-id", 
        type=str, 
        help="Batch ID to check status for"
    )
    group.add_argument(
        "--list-jobs",
        action="store_true",
        help="List all batch jobs and their status"
    )
    
    parser.add_argument(
        "--wait", 
        action="store_true",
        help="Wait for batch completion (only with --batch-id)"
    )
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Cancel the specified batch job (only with --batch-id)"
    )
    parser.add_argument(
        "--poll-interval", 
        type=int, 
        default=30,
        help="Polling interval in seconds when waiting (default: 30)"
    )
    
    return parser.parse_args()

def main():
    """Main function to check batch status"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return
    
    if args.list_jobs:
        list_batch_jobs()
        return
    
    if (args.wait or args.cancel) and not args.batch_id:
        print("Error: --wait and --cancel can only be used with --batch-id")
        return
    
    if args.wait and args.cancel:
        print("Error: --wait and --cancel cannot be used together")
        return
    
    # Handle cancellation
    if args.cancel:
        result = cancel_batch_job(args.batch_id)
        if result.get("status") == "error":
            exit(1)
        return
    
    # Check specific batch job
    result = check_batch_status(args.batch_id, args.wait, args.poll_interval)
    
    if result.get("status") == "error":
        exit(1)

if __name__ == "__main__":
    main()