# BiodiversityASSET

This project aims to extract, classify, and analyze biodiversity-related investor report paragraphs in terms of:

1. **Biodiversity relevance**
2. **Investment activity**
3. **Assetization characteristics** (intrinsic value, cash flow, ownership/control)

## Key Features

- **Modular batch processing**: Submit jobs, monitor progress, and download results independently
- **External prompt system**: Easily customize AI prompts using external text files
- **OpenAI Batch API integration**: Cost-effective processing of large datasets
- **Resume capability**: Pick up workflows from any step using batch IDs
- **Flexible configuration**: Support for different models, custom prompts, and processing parameters
- **Batch job management**: List, monitor, and cancel batch jobs as needed
- **Organized output**: Results saved in batch-specific subfolders for clean organization

## Processing Pipeline
| Step | Purpose | Script | Input Folder(s) | Output Folder(s) |
|------|---------|--------|----------------|------------------|
| 1 | Extract paragraphs from financial PDFs | `scripts/extract_pdfs.py` | `data/raw/pdfs` | `data/processed/extracted_paragraphs_from_pdfs` |
| 2 | Filter biodiversity-related paragraphs with hard-coded rules and BERT | `scripts/filter_biodiversity_paragraphs.py` | `data/processed/extracted_paragraphs_from_pdfs` | `data/processed/biodiversity_related_paragraphs` |
| 3a | **Submit** investment activity classification batch job | `scripts/submit_batch_job.py --task investment_activity_classification` | `biodiversity_related_paragraphs` + `prompts/` | Returns batch ID |
| 3b | **Check** batch job status | `scripts/check_batch_status.py` | Batch ID | Status updates |
| 3c | **Download** investment activity results | `scripts/download_batch_results.py` | Batch ID | `data/processed/investment_activity_classification` |
| 4a | **Submit** assetization features scoring batch job | `scripts/submit_batch_job.py --task assetization_features_scoring --batch-id <investment_batch_id>` | `investment_activity_classification/<batch_id>/` + `prompts/` | Returns batch ID |
| 4b | **Check** batch job status | `scripts/check_batch_status.py` | Batch ID | Status updates |
| 4c | **Download** assetization features results | `scripts/download_batch_results.py` | Batch ID | `data/processed/assetization_features_scoring` |
| 5 | Evaluate with human annotations | `scripts/evaluate.py` | `data/labels/human_annotations` | `results/evaluation` |

> **Note**: Steps 3-4 use OpenAI's Batch API for cost-effective processing. Each step can be run independently, allowing you to submit jobs and check results later. **Step 4 (assetization features scoring) requires the batch ID from a completed investment activity classification job.**

**Quick Start:**

# 0. Install dependencies
```bash
uv sync
```

> Make sure you have [`uv`](https://github.com/astral-sh/uv) installed beforehand, installation process is as follows:
    
Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux/MacOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
---

# 1. Extract paragraphs from PDFs
```bash
python scripts/extract_pdfs.py
```

# 2. Filter biodiversity-related paragraphs
```bash
python scripts/filter_biodiversity_paragraphs.py
```

# 3. Complete pipeline for investment activity classification
```bash
python scripts/submit_batch_job.py --task investment_activity_classification
python scripts/check_batch_status.py --batch-id <batch-id>
python scripts/download_batch_results.py --batch-id <batch-id>
```

# 4. Complete pipeline for assetization features scoring
```bash
python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id <investment_batch_id>
python scripts/check_batch_status.py --batch-id <assetization_batch_id> 
python scripts/download_batch_results.py --batch-id <assetization_batch_id>
```

# Monitoring and managing batch jobs
```bash
# List all batch jobs with status
python scripts/check_batch_status.py --list-jobs

# Wait for a job to complete
python scripts/check_batch_status.py --batch-id <batch-id> --wait

# Cancel a running job
python scripts/check_batch_status.py --batch-id <batch-id> --cancel
```

# Using custom prompt files (optional)
```bash
python scripts/submit_batch_job.py --task investment_activity_classification --system-prompt prompts/custom_system.txt --user-prompt prompts/custom_user.txt
```

For detailed workflow documentation, see [BATCH_WORKFLOW.md](BATCH_WORKFLOW.md).

## Environment Setup

Set up your OpenAI API key:
```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-openai-api-key"

# Option 2: Create .env file in project root
echo "OPENAI_API_KEY=your-openai-api-key" > .env
```

## Prompt File System

The project uses external text files for prompts, making them easy to customize:

- **`prompts/investment_activity_classification_system_prompt.txt`** - System prompt for investment activity classification
- **`prompts/assetization_features_scoring_system_prompt.txt`** - System prompt for assetization features scoring  
- **`prompts/user_prompt_template.txt`** - User prompt template applied to each paragraph

You can specify custom prompt files using the `--system-prompt` and `--user-prompt` options:
```bash
python scripts/submit_batch_job.py --task investment_activity_classification --system-prompt my_custom_system.txt
```

## Evaluation

Evaluation with human annotations is planned for future implementation.

## Project Structure

```text
BiodiversityASSET/
├── archive/                             # Backup/old files
├── data/
│   ├── raw/
│   │   └── pdfs/                        # PDF investor reports (input)
│   ├── processed/
│   │   ├── extracted_paragraphs_from_pdfs/      # Step 1: Extracted paragraphs
│   │   ├── biodiversity_related_paragraphs/     # Step 2: Filtered biodiversity paragraphs
│   │   ├── investment_activity_classification/  # Step 3c: Investment activity results
│   │   │   └── <batch_id>/
│   │   │       ├── batch_results.jsonl
│   │   │       ├── chunk_1.csv
│   │   │       └── chunk_2.csv
│   │   └── assetization_features_scoring/       # Step 4c: Assetization scoring results
│   │       └── <batch_id>/
│   │           ├── batch_results.jsonl
│   │           └── assetization_features_scored.csv
│   └── human_annotations/               # Manual annotations for evaluation
├── prompts/
│   ├── investment_activity_classification_system_prompt.txt
│   ├── assetization_features_scoring_system_prompt.txt
│   └── user_prompt_template.txt
├── results/
│   ├── batch_jobs/
│   │   ├── <batch_id>.json
│   │   ├── investment_activity_classification_processing/
│   │   └── assetization_features_scoring_processing/
│   └── evaluation/                      # Evaluation results (future)
├── scripts/                             # Python processing scripts
├── pyproject.toml                       # Project dependencies
├── uv.lock                              # Lock file for dependencies
├── README.md                            # Project documentation
├── BATCH_WORKFLOW.md                    # Detailed batch processing workflow
└── REFACTORING_SUMMARY.md               # Summary of refactoring changes
```


## Output File Organization

Results are organized in batch-specific subfolders for clean separation:

- **Investment Activity Classification**: Results saved in `data/processed/investment_activity_classification/<batch_id>/`
  - Individual chunk files (e.g., `chunk_1.csv`, `chunk_2.csv`)
  - Raw batch results (`batch_results.jsonl`)

- **Assetization Features Scoring**: Results saved in `data/processed/assetization_features_scoring/<batch_id>/`
  - Scored paragraphs (`assetization_features_scored.csv`)
  - Raw batch results (`batch_results.jsonl`)

This organization ensures that results from different batch jobs never conflict with each other.