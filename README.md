# BiodiversityASSET

> **LLM-powered analysis of biodiversity-related investment activities in financial reports**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-Batch%20API-green.svg)](https://platform.openai.com/docs/guides/batch)
[![uv](https://img.shields.io/badge/uv-package%20manager-purple.svg)](https://github.com/astral-sh/uv)

BiodiversityASSET is a comprehensive pipeline for extracting, classifying, and analyzing biodiversity-related content from investor reports. The system uses LLMs to evaluate paragraphs across three key dimensions:

1. **🌿 Biodiversity relevance** - Identifies content related to biodiversity and environmental impact
2. **💰 Investment activity** - Classifies paragraphs containing concrete investment activities  
3. **📊 Assetization characteristics** - Scores content on intrinsic value, cash flow, and ownership/control

## Table of Contents

- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Processing Pipeline](#processing-pipeline)
- [Batch Job Management](#batch-job-management)
- [Prompt Customization](#prompt-customization)
- [Project Structure](#project-structure)
- [Output Organization](#output-organization)
- [Documentation](#documentation)

## Key Features

✨ **Modular Architecture**
- Submit batch jobs and monitor progress independently
- Resume workflows from any step using batch IDs
- Cancel running jobs with safety confirmations

🤖 **LLM-Powered Processing**
- OpenAI Batch API integration for cost-effective analysis (~50% cost reduction)
- External prompt system for easy customization
- Support for multiple models and configurations

📁 **Organized Output**
- Results saved in batch-specific subfolders
- Clean filenames without ID conflicts
- Individual chunk processing for large datasets

🔧 **Developer-Friendly**
- Comprehensive CLI tools with intuitive options
- Detailed progress monitoring and error handling
- Flexible configuration and custom prompt support

## Processing Pipeline

The processing pipeline consists of sequential steps, with LLM-powered batch processing for steps 3-4:

| Step | Purpose | Script | Input | Output |
|------|---------|--------|-------|--------|
| **1** | Extract paragraphs from PDFs | `extract_pdfs.py` | `data/raw/pdfs/` | `extracted_paragraphs_from_pdfs/` |
| **2** | Filter biodiversity content | `filter_biodiversity_paragraphs.py` | `extracted_paragraphs_from_pdfs/` | `biodiversity_related_paragraphs/` |
| **3a** | **Submit** investment classification | `submit_batch_job.py` | `biodiversity_related_paragraphs/` | Returns batch ID |
| **3b** | **Monitor** batch progress | `check_batch_status.py` | Batch ID | Status updates |
| **3c** | **Download** investment results | `download_batch_results.py` | Batch ID | `investment_activity_classification/` |
| **4a** | **Submit** assetization scoring | `submit_batch_job.py` | `investment_activity_classification/` | Returns batch ID |
| **4b** | **Monitor** batch progress | `check_batch_status.py` | Batch ID | Status updates |
| **4c** | **Download** assetization results | `download_batch_results.py` | Batch ID | `assetization_features_scoring/` |

> **💡 Key Points:**
> - Steps 3-4 use OpenAI's Batch API for cost-effective processing
> - Each batch step can be run independently 
> - **Step 4 requires a completed investment activity classification batch ID**
> - All results are organized in batch-specific subfolders

## Quick Start

### Prerequisites

Ensure you have [`uv`](https://github.com/astral-sh/uv) installed:

<details>
<summary>📦 Install uv (click to expand)</summary>

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/MacOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
</details>

### 🚀 Installation

```bash
# Clone the repository
git clone <repository-url>
cd BiodiversityASSET

# Install dependencies
uv sync
```

### ⚙️ Environment Setup

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key"

# Or create a .env file
echo "OPENAI_API_KEY=your-openai-api-key" > .env
```

### 📝 Basic Usage

#### 1. Extract paragraphs from PDFs
```bash
python scripts/extract_pdfs.py
```

#### 2. Filter biodiversity-related content
```bash
python scripts/filter_biodiversity_paragraphs.py
```

#### 3. Classify investment activities
```bash
# Submit the batch job
python scripts/submit_batch_job.py --task investment_activity_classification

# Monitor progress (replace <batch-id> with actual ID)
python scripts/check_batch_status.py --batch-id <batch-id> --wait

# Download results
python scripts/download_batch_results.py --batch-id <batch-id>
```

#### 4. Score assetization features
```bash
# Submit dependent job (requires investment batch ID)
python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id <investment_batch_id>

# Monitor and download
python scripts/check_batch_status.py --batch-id <assetization_batch_id> --wait
python scripts/download_batch_results.py --batch-id <assetization_batch_id>
```

## Batch Job Management

### 📊 Monitoring Jobs

```bash
# List all batch jobs with LAST-CHECKED status and timestamps
python scripts/check_batch_status.py --list-jobs

# Check CURRENT status of a specific job
python scripts/check_batch_status.py --batch-id <batch-id>

# Wait for job completion (polls every 30 seconds)
python scripts/check_batch_status.py --batch-id <batch-id> --wait

# Custom polling interval
python scripts/check_batch_status.py --batch-id <batch-id> --wait --poll-interval 60
```

### ❌ Canceling Jobs

```bash
# Cancel a running batch job (requires confirmation)
python scripts/check_batch_status.py --batch-id <batch-id> --cancel
```

### 📋 Example Job Listing Output

```
=== Batch Jobs (3 found) ===
Batch ID                              Task                      Status          Last Checked      Submitted         Paragraphs  
-------------------------------------------------------------------------------------------------------------------------------
batch_686fc36b2da08190903bc237510c52f5 investment_activity_class completed       07-10 16:55       2025-07-10T15:43  120         
batch_686fd9e4f814819088b69150a57753d6 assetization_features_sc  submitted       never             2025-07-10T17:19  3           
batch_686fdd5143248190aae3f8185f24a415 investment_activity_class in_progress     07-10 14:30       2025-07-10T14:15  274         
```

## Prompt Customization

BiodiversityASSET uses external text files for prompts, making them easy to customize without code changes:

### 📁 Default Prompt Files

- **`prompts/investment_activity_classification_system_prompt.txt`** - System prompt for investment activity classification
- **`prompts/assetization_features_scoring_system_prompt.txt`** - System prompt for assetization features scoring  
- **`prompts/user_prompt_template.txt`** - User prompt template applied to each paragraph

### 🛠️ Using Custom Prompts

```bash
# Use custom system prompt
python scripts/submit_batch_job.py --task investment_activity_classification \
    --system-prompt prompts/my_custom_system.txt

# Use both custom system and user prompts
python scripts/submit_batch_job.py --task investment_activity_classification \
    --system-prompt prompts/my_custom_system.txt \
    --user-prompt prompts/my_custom_user.txt

# Use different model with custom prompts
python scripts/submit_batch_job.py --task assetization_features_scoring \
    --batch-id <investment_batch_id> \
    --model gpt-4o \
    --max-tokens 750 \
    --system-prompt prompts/my_custom_system.txt
```

## Project Structure

```
BiodiversityASSET/
├── 📁 data/
│   ├── 📁 raw/
│   │   └── 📁 pdfs/                     # 📄 Input: PDF investor reports
│   ├── 📁 processed/
│   │   ├── 📁 extracted_paragraphs_from_pdfs/      # Step 1: Extracted paragraphs
│   │   ├── 📁 biodiversity_related_paragraphs/     # Step 2: Filtered biodiversity content
│   │   ├── 📁 investment_activity_classification/  # Step 3: Investment classification results
│   │   │   └── 📁 <batch_id>/
│   │   │       ├── 📊 batch_results.jsonl
│   │   │       ├── 📊 chunk_1.csv
│   │   │       └── 📊 chunk_2.csv
│   │   └── 📁 assetization_features_scoring/       # Step 4: Assetization scoring results
│   │       └── 📁 <batch_id>/
│   │           ├── 📊 batch_results.jsonl
│   │           └── 📊 assetization_features_scored.csv
│   └── 📁 human_annotations/            # 👥 Manual annotations for evaluation
├── 📁 prompts/                          # 🤖 LLM prompt templates
│   ├── 📝 investment_activity_classification_system_prompt.txt
│   ├── 📝 assetization_features_scoring_system_prompt.txt
│   └── 📝 user_prompt_template.txt
├── 📁 results/
│   ├── 📁 batch_jobs/                   # 📋 Batch job metadata and raw results
│   │   ├── 📄 <batch_id>.json
│   │   ├── 📁 investment_activity_classification_processing/
│   │   └── 📁 assetization_features_scoring_processing/
│   └── 📁 evaluation/                   # 📈 Evaluation results (future)
├── 📁 scripts/                          # 🐍 Python processing scripts
├── ⚙️ pyproject.toml                    # 📦 Project dependencies
├── 🔒 uv.lock                           # 🔐 Lock file for dependencies
├── 📖 README.md                         # 📚 Project documentation
├── 📖 BATCH_WORKFLOW.md                 # 🔄 Detailed batch processing workflow
└── 📖 REFACTORING_SUMMARY.md            # 📝 Summary of refactoring changes
```

## Output Organization

Results are organized in batch-specific subfolders to prevent conflicts and enable easy tracking:

### 💼 Investment Activity Classification

```
data/processed/investment_activity_classification/<batch_id>/
├── 📊 batch_results.jsonl              # Raw API responses
├── 📊 chunk_1.csv                      # Processed results for chunk 1
└── 📊 chunk_2.csv                      # Processed results for chunk 2
```

**Contains:** Investment activity scores, explanations, and original paragraph metadata

### 📈 Assetization Features Scoring

```
data/processed/assetization_features_scoring/<batch_id>/
├── 📊 batch_results.jsonl              # Raw API responses
└── 📊 assetization_features_scored.csv # Scored paragraphs with all dimensions
```

**Contains:** Intrinsic value, cash flow, and ownership/control scores with detailed reasoning

### 🔑 Key Benefits

- **🔒 Conflict-free:** Each batch job gets its own subfolder
- **🏷️ Clean naming:** Filenames without batch ID suffixes  
- **📝 Traceable:** Easy to identify which batch produced which results
- **🔄 Resumable:** Can re-run or reference specific batch outputs

## Documentation

📖 **[BATCH_WORKFLOW.md](BATCH_WORKFLOW.md)** - Detailed step-by-step workflow guide with examples

📝 **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Complete summary of system architecture and changes

---

## Contributing

We welcome contributions! Please see our contribution guidelines for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use BiodiversityASSET in your research, please cite:

```bibtex
@software{biodiversityasset,
  title={BiodiversityASSET: LLM-powered analysis of biodiversity-related investment activities},
  author={SoDa},
  year={2025},
  url={https://github.com/yourusername/BiodiversityASSET}
}
```