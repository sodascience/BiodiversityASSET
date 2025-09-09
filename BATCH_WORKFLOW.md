# BiodiversityASSET Batch Processing Workflow

This document describes the new modular batch processing workflow for the BiodiversityASSET project. The workflow has been split into three distinct steps to provide better control and flexibility.

## Overview

The batch processing workflow now consists of three separate scripts:

1. **`submit_batch_job.py`** - Submit batch jobs to OpenAI
2. **`check_batch_status.py`** - Check job status and monitor progress
3. **`download_batch_results.py`** - Download and process completed results

This modular approach allows you to:

- Submit jobs and come back later to check results
- Monitor multiple jobs independently
- Resume the workflow from any step using batch IDs
- Handle batch processing failures more gracefully
- **Enforce sequential processing** where assetization features scoring depends on completed investment activity classification

## Workflow Steps

### Step 1: Submit Batch Job

Submit a new batch processing job to OpenAI Batch API.

```bash
# Submit investment activity classification job
python scripts/submit_batch_job.py --task investment_activity_classification

# Submit assetization features scoring job (requires batch ID from completed investment activity job)
python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id <investment_batch_id>

# Use custom model and settings
python scripts/submit_batch_job.py --task investment_activity_classification --model gpt-4o --max-tokens 1024
```

**Options:**

- `--task` (required): Task type (`investment_activity_classification` or `assetization_features_scoring`)
- `--batch-id`: Batch ID of completed investment activity classification job (required for `assetization_features_scoring`)
- `--input-dir`: Custom input directory (auto-detected based on task if not provided)
- `--model`: OpenAI model to use (default: `gpt-4o-mini`)
- `--max-tokens`: Maximum response tokens (default: 500)

**Output:**
The script will output a **Batch ID** that you'll need for the next steps. Example:

```
🚀 Batch job submitted successfully!
📋 Batch ID: batch_67891234abcd5678
```

> **Important**: For `assetization_features_scoring`, you must provide the batch ID of a **completed** `investment_activity_classification` job. The script will validate that the prerequisite batch exists and has finished successfully.

### Step 2: Check Batch Status

Monitor the progress of your submitted batch job.

```bash
# Check status once
python scripts/check_batch_status.py --batch-id batch_67891234abcd5678

# Wait for completion (checks every 30 seconds)
python scripts/check_batch_status.py --batch-id batch_67891234abcd5678 --wait

# Cancel a running batch job
python scripts/check_batch_status.py --batch-id batch_67891234abcd5678 --cancel

# List all batch jobs with their last checked status
python scripts/check_batch_status.py --list-jobs
```

**Options:**

- `--batch-id`: Batch ID to check (from Step 1)
- `--wait`: Wait for completion instead of checking once
- `--cancel`: Cancel the specified batch job (requires confirmation)
- `--poll-interval`: Seconds between checks when waiting (default: 30)
- `--list-jobs`: List all batch jobs and their status

**Output:**

```
⌛ Status: IN_PROGRESS
📊 Progress: 150/200 (75.0%)
⏱️  Elapsed time: 5m 30s
```

**List Jobs Output:**

```
=== Batch Jobs (3 found) ===
Batch ID                              Task                      Status          Last Checked      Submitted         Paragraphs  
-------------------------------------------------------------------------------------------------------------------------------
batch_686fc36b2da08190903bc237510c52f5 investment_activity_class completed       07-10 16:55       2025-07-10T15:43  120   
batch_686fd9e4f814819088b69150a57753d6 assetization_features_sc  submitted       never             2025-07-10T17:19  3   
batch_686fdd5143248190aae3f8185f24a415 investment_activity_class in_progress     07-10 14:30       2025-07-10T14:15  274   

Note: Status may not be current. Use --batch-id to get live status.
```

### Step 3: Download and Process Results

Download and process results from a completed batch job.

```bash
# Download and process results
python scripts/download_batch_results.py --batch-id batch_67891234abcd5678

# Use custom output directory
python scripts/download_batch_results.py --batch-id batch_67891234abcd5678 --output-dir custom/path
```

**Options:**

- `--batch-id`: Batch ID of completed job
- `--output-dir`: Custom output directory (auto-detected based on task if not provided)

**Output:**
Processed results will be saved to batch-specific subfolders:

- **Investment Activity Classification**: `data/processed/investment_activity_classification/<batch_id>/`
- **Assetization Features Scoring**: `data/processed/assetization_features_scoring/<batch_id>/`

> **Note**: All output files are organized in subfolders named with the full batch ID. Filenames are clean and do not include batch ID suffixes for better organization.

## Environment Setup

Ensure you have the required environment variables:

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Optional  
export OPENAI_MODEL="gpt-4o-mini"  # Default model to use
```

Or create a `.env` file:

```
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
```

## Input Data Requirements

### Investment Activity Task

- **Input Directory**: `data/processed/biodiversity_related_paragraphs/`
- **Required Files**: CSV files with `paragraph_text` column
- **Output**: Filtered paragraphs with investment activity scores

### Assetization Features Task

- **Input Directory**: `data/processed/investment_activity_classification/<batch_id>/`
- **Required Files**: CSV files with `paragraph_text` column and `score` column (only processes score=1)
- **Prerequisite**: Completed investment activity classification batch job
- **Output**: Paragraphs scored on assetization dimensions

## Batch Job Management

### Job Metadata

Each batch job creates a metadata file in `results/batch_jobs/` containing:

- Batch ID and task type
- Input files and processing parameters
- Submission and completion timestamps
- Status and progress information

### Resume from Any Step

You can resume the workflow from any step using the batch ID:

```bash
# If you lost track of your job, list all jobs
python scripts/check_batch_status.py --list-jobs

# Continue checking a job
python scripts/check_batch_status.py --batch-id <your-batch-id>

# Download results when ready
python scripts/download_batch_results.py --batch-id <your-batch-id>
```

## Error Handling

### Common Issues and Solutions

**Batch Status "Failed":**

```bash
# Check detailed error information
python scripts/check_batch_status.py --batch-id <batch-id>
```

**Missing Input Files:**

- Ensure input CSV files exist in the expected directory
- Run earlier pipeline steps to generate input data

**API Key Issues:**

- Verify `OPENAI_API_KEY` is set correctly
- Check API key permissions and quota

**Incomplete Results:**

- Check if all input files were processed correctly
- Review batch job metadata for processing details

## Migration from Old Workflow

The old `filter_investment_activity_paragraphs.py` script has been deprecated in favor of this modular approach. To migrate:

1. **Old way:**

   ```bash
   python scripts/filter_investment_activity_paragraphs.py
   ```
2. **New way:**

   ```bash
   # Step 1: Submit job
   python scripts/submit_batch_job.py --task investment_activity_classification

   # Step 2: Check status
   python scripts/check_batch_status.py --batch-id <batch-id> --wait

   # Step 3: Download results  
   python scripts/download_batch_results.py --batch-id <batch-id>
   ```

## Advanced Usage

### Processing Multiple Tasks

You need to process tasks sequentially since assetization features scoring depends on investment activity classification:

```bash
# Step 1: Submit investment activity classification
python scripts/submit_batch_job.py --task investment_activity_classification
# Output: batch_abc123...

# Step 2: Wait for completion
python scripts/check_batch_status.py --batch-id batch_abc123... --wait

# Step 3: Download investment activity results
python scripts/download_batch_results.py --batch-id batch_abc123...

# Step 4: Submit assetization features scoring using the completed batch ID
python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id batch_abc123...
python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id batch_abc123...
# Output: batch_def456...

# Step 5: Monitor and download assetization results
python scripts/check_batch_status.py --batch-id batch_def456... --wait
python scripts/download_batch_results.py --batch-id batch_def456...
```

### Custom Model Selection

Use different models for different tasks:

```bash
# Use GPT-4 for investment activity (higher accuracy)
python scripts/submit_batch_job.py --task investment_activity_classification --model gpt-4o

# Use GPT-4o-mini for assetization features (faster, cheaper)
# Note: Must provide the batch ID from completed investment activity job
python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id <investment_batch_id> --model gpt-4o-mini
```

### Monitoring Long-Running Jobs

For jobs that take several hours:

```bash
# Submit and forget
python scripts/submit_batch_job.py --task investment_activity_classification

# Check later (no waiting)
python scripts/check_batch_status.py --batch-id <batch-id>

# Or wait with longer polling interval  
python scripts/check_batch_status.py --batch-id <batch-id> --wait --poll-interval 300


#For more than one batch - directly in powershell
$batch_ids = @(
    "batch_688a1c2e4a4c8190ac6e08f8945bf18d",
    "batch_688a1c305488819092a72358ff968045",
    "batch_688a1c35425481908b20a166bd737db5",
    "batch_688a1c36ba6c8190bd908810fe629317",
    "batch_688a1c3838788190aa6b2c41e06ff865",
    "batch_688a1c3a192c8190918a5e7ff21502ad",
    "batch_688a1c3cbc2481909929eb2677b8d2f7",
    "batch_688a1c404c608190a7ff41942e8fe69f",
    "batch_688a1c45484c8190a05218fb308160ba",
    "batch_688a1c4b701c8190896ba444d823c26a",
    "batch_688a1c4dac848190ae68ce5ea1be6bdf",
    "batch_688a1c51f9c0819080cf10afd718f8cd",
    "batch_688a1c55af24819085bbd1cb36864c73",
    "..."
)

foreach ($id in $batch_ids) {
    Write-Host "`n🔎 Checking status for $id" -ForegroundColor Cyan
    python scripts/check_batch_status.py --batch-id $id
}

#To delete a batch directly in powershell
$batchIds = @(
    "batch_6889fce513dc81909b5a6004a9c5f9d9",
    "..."
)

foreach ($batchId in $batchIds) {
    $file = "results/batch_jobs/$batchId.json"
    if (Test-Path $file) {
        Remove-Item $file
        Write-Host "🗑️ Deleted: $file"
    } else {
        Write-Host "❌ File not found: $file"
    }


#submit all batches for assetization after listing all of them (using list jobs)
# All investment batch IDs
$batches = @(
    "batch_6887dc97f2e08190a0e7272c4085f0ee",
    "..."
)

foreach ($batchId in $batches) {
    Write-Host "`n🚀 Submitting assetization scoring for $batchId" -ForegroundColor Cyan
    python scripts/submit_batch_job.py `
        --task assetization_features_scoring `
        --batch-id $batchId `
        --model gpt-4o-mini `
        --system-prompt prompts/assetization_features_scoring_system_prompt.txt `
        --user-prompt prompts/user_prompt_template.txt
    Start-Sleep -Seconds 2
}

```

## Output File Structure

### Investment Activity Classification Results

```
data/processed/investment_activity_classification/
└── batch_686ec512cd648190b4af479e60ad47fe/
    ├── batch_results.jsonl                           # Raw batch results
    ├── chunk_1.csv                                   # Individual file results
    └── chunk_2.csv                                   # Individual file results
```

### Assetization Features Scoring Results

```
data/processed/assetization_features_scoring/
└── batch_def456789abcd1234efgh567890ijklmn/
    ├── batch_results.jsonl                           # Raw batch results
    └── assetization_features_scored.csv              # Scored paragraphs
```

**Key Features:**

- **Batch ID Subfolders**: All results are organized in subfolders named with the full batch ID
- **Clean Filenames**: Filenames no longer include batch ID suffixes (organized by folder instead)
- **Individual Chunk Files**: Investment activity results are saved as separate files per input chunk
- **No Combined Files**: Only individual chunk files are saved (no combined CSV for investment activity)

Each output file includes:

- Original paragraph text and metadata
- AI-generated scores and explanations
- Source file traceability
- Processing timestamps
- **Organized by batch ID subfolder** for easy identification and organization

## Troubleshooting

### Debug Mode

Add verbose output by checking metadata files:

```bash
# View batch job details
cat results/batch_jobs/<batch-id>.json

# Check raw results
head results/batch_jobs/*/*_results.jsonl
```

### Restart Failed Jobs

If a job fails, you can resubmit with the same parameters:

```bash
# Check what failed
python scripts/check_batch_status.py --batch-id <failed-batch-id>

# Resubmit with same settings
python scripts/submit_batch_job.py --task <same-task> --model <same-model>
```

For more help, check the individual script help:

```bash
python scripts/submit_batch_job.py --help
python scripts/check_batch_status.py --help  
python scripts/download_batch_results.py --help
```
