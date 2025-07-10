# BiodiversityASSET: Complete Refactoring Summary

## ✅ PHASE 1 COMPLETED: Modular Architecture (Previous)

The BiodiversityASSET project was initially refactored to use a modern, modular architecture with:

- **Modular batch API utilities** following the ODISSEI-SODA tutorial structure
- **CSV-based data pipeline** with proper folder structure
- **Progress bars and robust error handling**
- **Reusable components** across different processing steps
- **OpenAI Batch API integration** for efficient LLM processing

## ✅ PHASE 3 COMPLETED: Batch ID Dependencies & Folder Organization (Latest)

Building on the external prompt system, the pipeline has been further enhanced with:

### New Features

1. **Batch ID Dependencies**:
   - Assetization features scoring now requires a completed investment activity classification batch ID
   - Added validation to ensure prerequisite batches exist and are completed
   - Enhanced metadata tracking with prerequisite batch relationships

2. **Improved File Organization**:
   - Results are now organized in batch ID subfolders for better isolation
   - Simplified filenames without batch ID suffixes (organized by folder instead)
   - Individual chunk files only (no combined CSV files for investment activity)

3. **Enhanced Workflow Enforcement**:
   - Sequential processing enforced through command-line validation
   - Clear error messages when prerequisites are missing
   - Automatic input directory detection based on prerequisite batch results

4. **Enhanced Batch Job Management**:
   - Added `--cancel` option to cancel running batch jobs with confirmation
   - Enhanced `--list-jobs` to show last checked timestamp (to the minute)
   - Improved error handling and detailed status information
   - Better job monitoring with comprehensive metadata tracking

## ✅ PHASE 2 COMPLETED: External Prompt System & Enhanced Modularity

The pipeline has been completely refactored to implement a file-based prompt system and enhanced modular batch processing workflow with the following key improvements:

### Major Changes

1. **External Prompt File System**:
   - Removed all hardcoded prompts from scripts
   - Created dedicated prompt files in `prompts/` directory
   - Implemented flexible prompt loading utilities
   - Added command-line options for custom prompt files

2. **Enhanced Batch Job Submission**:
   - Refactored `submit_batch_job.py` to be completely modular
   - Added support for custom system and user prompts
   - Implemented proper task-to-prompt file mapping
   - Enhanced CLI with intuitive options

3. **Updated Documentation**:
   - Completely revised README.md with new workflow
   - Updated BATCH_WORKFLOW.md with correct task names
   - Added comprehensive environment setup instructions
   - Documented the external prompt system

4. **Improved .gitignore**:
   - Added comprehensive patterns for Python projects
   - Included security-critical .env file exclusion
   - Added IDE, OS, and development tool patterns

### Core Components

1. **`scripts/submit_batch_job.py`** - Completely refactored batch job submission script:
   - File-based prompt loading with `load_prompt_from_file()`
   - Automatic prompt file detection with `get_default_prompt_files()`
   - Flexible CLI supporting `--system-prompt` and `--user-prompt` options
   - Support for both standard task names and custom configurations

2. **External Prompt Files**:
   - `prompts/investment_activity_classification_system_prompt.txt` - Investment activity classification prompt
   - `prompts/assetization_features_scoring_system_prompt.txt` - Assetization features scoring prompt  
   - `prompts/user_prompt_template.txt` - User prompt template for paragraphs
   - Easy to customize and version control

3. **Enhanced Utility Scripts**:
   - `scripts/check_batch_status.py` - Monitor batch job progress, cancel jobs, list all jobs with timestamps
   - `scripts/download_batch_results.py` - Download and process completed results with batch ID organization
   - `scripts/batch_api_utils.py` - Core batch API processing utilities

4. **Updated Documentation**:
   - Comprehensive README.md with new workflow examples
   - Detailed BATCH_WORKFLOW.md with step-by-step instructions
   - Environment setup and configuration guidance

## Key Features

### 1. External Prompt File System

All prompts are now stored in external text files for easy customization:

```python
from scripts.submit_batch_job import load_prompt_from_file, get_default_prompt_files

# Load prompts from files
system_prompt = load_prompt_from_file("prompts/investment_activity_classification_system_prompt.txt")
user_prompt = load_prompt_from_file("prompts/user_prompt_template.txt")

# Get default prompt files for a task
system_file, user_file = get_default_prompt_files("investment_activity_classification")
```

### 2. Flexible Command-Line Interface

The refactored `submit_batch_job.py` provides intuitive options:

```bash
# Use default prompts for a task
python scripts/submit_batch_job.py --task investment_activity_classification

# Submit assetization features scoring with prerequisite batch ID
python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id batch_abc123...

# Use custom prompt files
python scripts/submit_batch_job.py --task investment_activity_classification \
    --system-prompt custom_system.txt --user-prompt custom_user.txt

# Different models and parameters
python scripts/submit_batch_job.py --task assetization_features_scoring \
    --batch-id batch_abc123... --model gpt-4o --max-tokens 750
```

### 3. Batch Job Management and Monitoring

```bash
# List all batch jobs with status and last checked timestamp
python scripts/check_batch_status.py --list-jobs

# Check specific job status
python scripts/check_batch_status.py --batch-id <batch-id>

# Wait for completion with custom polling interval
python scripts/check_batch_status.py --batch-id <batch-id> --wait --poll-interval 60

# Cancel a running batch job (requires confirmation)
python scripts/check_batch_status.py --batch-id <batch-id> --cancel

# Example list output:
# === Batch Jobs (3 found) ===
# Batch ID                              Task                      Status          Last Checked      Submitted         Paragraphs  
# -------------------------------------------------------------------------------------------------------------------------------
# batch_686fc36b2da08190903bc237510c52f5 investment_activity_class completed       07-10 16:55       2025-07-10T15:43  120         
# batch_686fd9e4f814819088b69150a57753d6 assetization_features_sc  submitted       never             2025-07-10T17:19  3           
# batch_686fdd5143248190aae3f8185f24a415 investment_activity_class in_progress     07-10 14:30       2025-07-10T14:15  274         
```

### 4. Sequential Processing Workflow

The workflow now enforces proper dependencies between tasks:

1. **Submit Investment Activity**: `python scripts/submit_batch_job.py --task investment_activity_classification`
2. **Wait for Completion**: `python scripts/check_batch_status.py --batch-id <id> --wait`
3. **Download Results**: `python scripts/download_batch_results.py --batch-id <id>`
4. **Submit Assetization**: `python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id <investment_id>`
5. **Monitor & Download**: Continue with steps 2-3 for the new batch

### 5. Structured Output with Pydantic

Response schemas ensure consistent, validated outputs:

```python
# Investment activity filtering output
{
    "action_score": 1,
    "selected": true,
    "explanation": "Describes concrete investment in renewable energy projects"
}

# Assetization features scoring output  
{
    "intrinsic_value_score": 3,
    "cash_flow_score": 2,
    "ownership_control_score": 2,
    "overall_assetization_score": 2.33,
    "reasoning": "High intrinsic value due to tangible assets..."
}
```

### 6. Enhanced File Organization

**Batch ID Subfolders**: All output files are organized in batch-specific subfolders:
- **Clean separation**: Each batch gets its own subfolder named with the full batch ID
- **No filename conflicts**: Multiple batches won't overwrite each other's files
- **Simplified names**: Filenames are clean without batch ID suffixes
- **Easy identification**: Clear organization makes it easy to find specific batch results

```
data/processed/investment_activity_classification/
└── batch_686fc36b2da08190903bc237510c52f5/
    ├── batch_results.jsonl
    ├── chunk_1.csv
    └── chunk_2.csv

data/processed/assetization_features_scoring/
└── batch_def456789abcd1234efgh567890ijklmn/
    ├── batch_results.jsonl
    └── assetization_features_scored.csv
```

All scripts follow the documented folder structure:

```
BiodiversityASSET/
├── prompts/                    # External prompt files
│   ├── investment_activity_classification_system_prompt.txt
│   ├── assetization_features_scoring_system_prompt.txt
│   └── user_prompt_template.txt
├── data/
│   ├── raw/pdfs/                               # Step 1 input
│   ├── processed/
│   │   ├── extracted_paragraphs_from_pdfs/     # Step 1 output
│   │   ├── biodiversity_related_paragraphs/    # Step 2 output
│   │   ├── investment_activity_classification/ # Step 3c output
│   │   │   └── batch_686fc36b2da08190903bc237510c52f5/
│   │   │       ├── batch_results.jsonl
│   │   │       ├── chunk_1.csv
│   │   │       └── chunk_2.csv
│   │   └── assetization_features_scoring/      # Step 4c output
│   │       └── batch_def456789abcd1234efgh567890ijklmn/
│   │           ├── batch_results.jsonl
│   │           └── assetization_features_scored.csv
│   └── human_annotations/                      # Manual annotations
├── results/
│   ├── batch_jobs/                             # Batch job metadata
│   │   ├── batch_686fc36b2da08190903bc237510c52f5.json
│   │   ├── batch_def456789abcd1234efgh567890ijklmn.json
│   │   ├── investment_activity_classification_processing/
│   │   └── assetization_features_scoring_processing/
│   └── evaluation/                             # Future evaluation results
└── scripts/                                    # Processing scripts
```

## Usage Examples

### 1. Complete Modular Workflow

```bash
# Install dependencies
uv sync

# Set up environment
export OPENAI_API_KEY="your-api-key"

# Run complete pipeline
# Step 1: Extract PDFs
python scripts/extract_pdfs.py

# Step 2: Filter biodiversity paragraphs  
python scripts/filter_biodiversity_paragraphs.py

# Step 3: Investment activity classification (3-step process)
python scripts/submit_batch_job.py --task investment_activity_classification
python scripts/check_batch_status.py --batch-id <batch-id> --wait
python scripts/download_batch_results.py --batch-id <batch-id>

# Step 4: Assetization features scoring (3-step process with prerequisite)
python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id <investment_batch_id>
python scripts/check_batch_status.py --batch-id <assetization_batch_id> --wait  
python scripts/download_batch_results.py --batch-id <assetization_batch_id>
```

### 2. Using Custom Prompt Files

```bash
# Create custom prompt file
echo "You are an expert financial analyst..." > my_custom_system.txt

# Use custom prompts for investment activity
python scripts/submit_batch_job.py --task investment_activity_classification \
    --system-prompt my_custom_system.txt --user-prompt my_custom_user.txt

# Use custom prompts for assetization (requires prerequisite batch ID)
python scripts/submit_batch_job.py --task assetization_features_scoring \
    --batch-id <investment_batch_id> --system-prompt my_custom_system.txt

# Use different models
python scripts/submit_batch_job.py --task assetization_features_scoring \
    --batch-id <investment_batch_id> --model gpt-4o --max-tokens 750
```

### 4. Sequential Processing Workflow

```bash
# Complete sequential workflow
# Step 1: Submit investment activity classification
batch_id_1=$(python scripts/submit_batch_job.py --task investment_activity_classification | grep "Batch ID:" | cut -d' ' -f3)

# Step 2: Wait for completion
python scripts/check_batch_status.py --batch-id $batch_id_1 --wait

# Step 3: Download investment results
python scripts/download_batch_results.py --batch-id $batch_id_1

# Step 4: Submit assetization scoring using the completed batch ID
batch_id_2=$(python scripts/submit_batch_job.py --task assetization_features_scoring --batch-id $batch_id_1 | grep "Batch ID:" | cut -d' ' -f3)

# Step 5: Wait and download assetization results
python scripts/check_batch_status.py --batch-id $batch_id_2 --wait
python scripts/download_batch_results.py --batch-id $batch_id_2

# List all jobs to see the relationship
python scripts/check_batch_status.py --list-jobs
```

## Environment Setup

### API Key Configuration

```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-openai-api-key"

# Option 2: Create .env file (recommended)
echo "OPENAI_API_KEY=your-openai-api-key" > .env
```

### Dependencies

Current `pyproject.toml` includes all required packages:

```toml
dependencies = [
    "pdfminer.six>=20221105",
    "pandas>=2.0.0", 
    "transformers>=4.21.0",
    "torch>=1.12.0",
    "tokenizers>=0.13.0",
    "tqdm>=4.64.0",
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]
```

## Task Names and File Mapping

The refactored system uses consistent task naming:

| Task Name | System Prompt File | Purpose | Prerequisites |
|-----------|-------------------|---------|---------------|
| `investment_activity_classification` | `investment_activity_classification_system_prompt.txt` | Classify paragraphs for investment activities | Biodiversity-related paragraphs |
| `assetization_features_scoring` | `assetization_features_scoring_system_prompt.txt` | Score assetization characteristics | Completed investment activity classification batch |

All tasks use the same user prompt template: `user_prompt_template.txt`

## Benefits of the Refactored Architecture

### 1. **Maintainability & Customization**
- **External prompts**: Easy to modify AI behavior without code changes
- **Version control**: Prompt files can be tracked and versioned separately
- **A/B testing**: Simple to test different prompt variations
- **Team collaboration**: Non-programmers can modify prompts

### 2. **Modularity & Flexibility**
- **Independent steps**: Submit jobs and check results later
- **Custom configurations**: Support for different models, prompts, and parameters
- **Resume capability**: Pick up workflows from any step using batch IDs
- **Multiple job management**: Submit and monitor multiple jobs simultaneously

### 3. **Cost Efficiency & Performance**
- **Batch API integration**: ~50% cost reduction compared to real-time API
- **Efficient processing**: Handle large datasets without timeout issues
- **Background operation**: Submit jobs and check results when convenient
- **Resource optimization**: No need to keep terminals/processes running

### 4. **User Experience & Robustness**
- **Clear documentation**: Comprehensive README and BATCH_WORKFLOW guides
- **Intuitive CLI**: Easy-to-use command-line interface with helpful options
- **Progress monitoring**: Real-time status updates and job listing with timestamps
- **Error handling**: Robust error recovery and detailed error messages
- **Job cancellation**: Cancel running batch jobs with confirmation prompts

### 5. **Enhanced Batch Job Management**
- **Comprehensive listing**: View all batch jobs with status, last checked time, and metadata
- **Job cancellation**: Cancel running or queued batch jobs with safety confirmations
- **Status tracking**: Detailed progress monitoring with elapsed time and completion rates
- **Relationship tracking**: Maintain metadata links between dependent batch jobs

### 6. **Dependency Management & Workflow Enforcement**
- **Sequential processing**: Assetization features scoring requires completed investment activity classification
- **Automatic validation**: Scripts validate that prerequisite batches exist and are completed
- **Input directory detection**: Automatically finds results from prerequisite batches
- **Metadata tracking**: Tracks relationships between dependent batch jobs
- **Clear error messages**: Helpful guidance when prerequisites are missing

### 7. **Enhanced File Organization**
- **Batch ID subfolders**: All results organized in batch-specific subfolders
- **Clean filenames**: No batch ID suffixes in filenames (organized by folder instead)
- **Individual chunk files**: Investment activity results saved as separate chunk files
- **No file conflicts**: Multiple batches completely isolated from each other

## Migration from Previous Version

### What Changed:
- ❌ **Removed**: Hardcoded prompts in Python files
- ❌ **Removed**: Automatic job submission and waiting
- ❌ **Removed**: `run_pipeline.py` (replaced with individual scripts)
- ❌ **Removed**: Batch ID suffixes in filenames
- ❌ **Removed**: Combined CSV files for investment activity results
- ✅ **Added**: External prompt file system
- ✅ **Added**: Modular 3-step batch workflow
- ✅ **Added**: Enhanced CLI with custom prompt support
- ✅ **Added**: Batch ID dependency validation
- ✅ **Added**: Batch ID subfolder organization
- ✅ **Added**: Batch job cancellation with confirmation
- ✅ **Added**: Enhanced job listing with last checked timestamps
- ✅ **Updated**: All documentation and usage examples

### Migration Steps:
1. Update any custom prompt modifications to external files
2. Replace `run_pipeline.py` calls with new sequential workflow
3. Update task names to use new convention (`investment_activity_classification`, `assetization_features_scoring`)
4. Set up environment variables or .env file for API key
5. **Important**: For assetization features scoring, ensure you have the batch ID from a completed investment activity classification job

## Next Steps

The refactored pipeline is now ready for:
- **Production deployment** with robust error handling and dependency validation
- **Custom prompt development** for different use cases  
- **Integration with CI/CD pipelines** for automated sequential processing
- **Scaling to larger datasets** with isolated batch processing
- **Extension to other LLM providers** through modular design
- **Team collaboration** with non-technical stakeholders on prompt design
- **Complex workflow management** with proper dependency tracking

The system maintains full backward compatibility while providing significantly enhanced functionality, maintainability, and proper workflow enforcement.
