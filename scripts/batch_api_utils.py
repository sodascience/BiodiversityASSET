"""
Batch API Utilities for BiodiversityASSET Project

This module provides reusable utilities for OpenAI Batch API processing,
following the ODISSEI-SODA tutorial structure for LLM batch structured output.

Tutorial reference: https://odissei-soda.nl/tutorials/llm_batch_structured_output/
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from openai import OpenAI
from torch import res
from tqdm import tqdm


class BatchAPIProcessor:
    """
    A modular class for processing data with OpenAI's Batch API.
    Supports structured output and progress tracking.
    """
    
    # Supported OpenAI models for batch processing
    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Batch API processor.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment.
        """
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = OpenAI(api_key=api_key)
    
    def validate_model(self, model: str) -> str:
        """
        Validate and potentially suggest corrections for model names.
        
        Args:
            model: Model name to validate
            
        Returns:
            Validated model name
        """
        if model in self.SUPPORTED_MODELS:
            return model
        
        # Check for common variations
        model_lower = model.lower()
        if "gpt-4o-mini" in model_lower:
            return "gpt-4o-mini"
        elif "gpt-4o" in model_lower:
            return "gpt-4o"
        elif "gpt-4-turbo" in model_lower:
            return "gpt-4-turbo"
        elif "gpt-4" in model_lower:
            return "gpt-4"
        elif "gpt-3.5" in model_lower:
            return "gpt-3.5-turbo"
        
        print(f"Warning: Model '{model}' not in known supported models: {self.SUPPORTED_MODELS}")
        print(f"Proceeding with '{model}' - ensure it supports batch processing and structured output")
        return model
    
    def prepare_batch_file(
        self,
        data: pd.DataFrame,
        system_prompt: str,
        user_prompt_template: str,
        text_column: str,
        response_schema: Dict[str, Any],
        model: str = "gpt-4o-mini",
        output_path: Path = None,
        max_tokens: int = 1000,
        custom_id_prefix: str = "request"
    ) -> Path:
        """
        Prepare a JSONL file for OpenAI Batch API processing.
        
        Args:
            data: DataFrame containing the data to process
            system_prompt: System prompt for the model
            user_prompt_template: Template for user prompts (use {text} placeholder)
            text_column: Name of the column containing text to process
            response_schema: JSON schema for structured response
            model: OpenAI model to use
            output_path: Path to save the JSONL file
            max_tokens: Maximum tokens for response
            custom_id_prefix: Prefix for custom IDs
            
        Returns:
            Path to the created JSONL file
        """
        if output_path is None:
            output_path = Path(f"batch_input_{custom_id_prefix}.jsonl")
        
        # Validate model
        model = self.validate_model(model)
        
        jsonl_lines = []
        
        print(f"Preparing batch file with {len(data)} requests using model: {model}...")
        
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Creating batch requests"):
            text = self._clean_text(str(row[text_column]))
            user_prompt = user_prompt_template.format(text=text)
            
            request = {
                "custom_id": f"{custom_id_prefix}_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": max_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_response",
                            "schema": response_schema,
                            "strict": True
                        }
                    }
                }
            }
            
            jsonl_lines.append(json.dumps(request, ensure_ascii=False))
        
        # Save JSONL file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(jsonl_lines))
        
        print(f"✅ Batch file created: {output_path}")
        return output_path
    
    def submit_batch(
        self,
        input_file_path: Path,
        description: str = "Batch processing job"
    ) -> str:
        """
        Submit a batch job to OpenAI.
        
        Args:
            input_file_path: Path to the JSONL input file
            description: Description for the batch job
            
        Returns:
            Batch ID
        """
        print(f"Uploading batch file: {input_file_path.name}")
        
        with open(input_file_path, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")
        
        print(f"File uploaded with ID: {batch_input_file.id}")
        
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description}
        )
        
        print(f"🚀 Batch submitted with ID: {batch.id}")
        return batch.id
    
    def get_batch_status(self, batch_id: str) -> dict:
        """
        Get the current status of a batch job.
        
        Args:
            batch_id: Batch ID to check
            
        Returns:
            Dictionary with batch status information
        """
        try:
            batch = self.client.batches.retrieve(batch_id)
            
            result = {
                "batch_id": batch_id,
                "status": batch.status,
                "created_at": batch.created_at,
                "in_progress_at": getattr(batch, 'in_progress_at', None),
                "expires_at": getattr(batch, 'expires_at', None),
                "finalizing_at": getattr(batch, 'finalizing_at', None),
                "completed_at": getattr(batch, 'completed_at', None),
                "failed_at": getattr(batch, 'failed_at', None),
                "cancelled_at": getattr(batch, 'cancelled_at', None),
                "request_counts": getattr(batch, 'request_counts', None),
                "metadata": getattr(batch, 'metadata', {}),
                "input_file_id": getattr(batch, 'input_file_id', None),
                "output_file_id": getattr(batch, 'output_file_id', None),
                "error_file_id": getattr(batch, 'error_file_id', None)
            }
            
            return result
            
        except Exception as e:
            return {
                "batch_id": batch_id,
                "status": "error",
                "error": str(e)
            }

    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = 30,
        max_wait_time: int = 86400  # 24 hours
    ) -> str:
        """
        Wait for batch completion and return output file ID.
        
        Args:
            batch_id: Batch ID to monitor
            poll_interval: Seconds between status checks
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            Output file ID
        """
        start_time = time.time()
        
        print(f"⌛ Waiting for batch completion: {batch_id}")
        
        while True:
            if time.time() - start_time > max_wait_time:
                raise TimeoutError(f"Batch processing exceeded maximum wait time of {max_wait_time} seconds")
            
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            
            print(f"⌛ Batch status: {status}")
            
            if status == "completed" and batch.output_file_id:
                print("✅ Batch completed. Output file ready.")
                return batch.output_file_id
            elif status in ["failed", "cancelled"]:
                raise RuntimeError(f"❌ Batch {status}. Check batch details for error information.")
            
            time.sleep(poll_interval)
    
    def download_results(
        self,
        output_file_id: str,
        output_path: Path
    ) -> Path:
        """
        Download batch results to a file.
        
        Args:
            output_file_id: Output file ID from completed batch
            output_path: Path to save the results
            
        Returns:
            Path to the downloaded file
        """
        file_response = self.client.files.content(output_file_id)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(file_response.read())
        
        print(f"📄 Results downloaded to: {output_path}")
        return output_path
    
    def process_batch_results(
        self,
        results_file_path: Path,
        original_data: pd.DataFrame,
        output_csv_path: Path = None
    ) -> pd.DataFrame:
        """
        Process batch results and merge with original data.
        
        Args:
            results_file_path: Path to the JSONL results file
            original_data: Original DataFrame used for batch processing
            output_csv_path: Optional path to save processed results as CSV
            
        Returns:
            DataFrame with processed results
        """
        print(f"Processing batch results from: {results_file_path}")
        #print(original_data)
        #print(original_data.iloc[0])
        results = []
        
        print(f"Processing batch results from: {results_file_path}")
        results = []
 
        # Optional: map original (possibly non-contiguous) index -> 0..N-1 position
        index_to_pos = {idx: pos for pos, idx in enumerate(original_data.index)}
        use_seq_fallback = False
        seq_pos = 0  # sequential fallback cursor
 
        with open(results_file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Processing results"):
                line = line.strip()
                if not line:
                    continue
 
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
 
                custom_id = result.get("custom_id", "")
                # Try to parse JSON content from structured output
                response_content = (
                    result.get("response", {})
                        .get("body", {})
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "{}")
                )
                try:
                    parsed_response = json.loads(response_content)
                except json.JSONDecodeError:
                    continue
 
                # 1) First try to align by custom_id suffix (original index value)
                pos = None
                try:
                    key = int(custom_id.split("_")[-1])
                    pos = index_to_pos.get(key, None)
                except Exception:
                    pos = None
 
                # 2) If that fails or is out of bounds, fall back to sequential order
                if pos is None or pos < 0 or pos >= len(original_data):
                    use_seq_fallback = True
                    pos = seq_pos
                    seq_pos += 1
 
                if pos >= len(original_data):
                    # nothing left to align—stop attempting further merges
                    break
 
                row_data = original_data.iloc[pos].to_dict()
                row_data.update(parsed_response)
                row_data["batch_custom_id"] = custom_id
                results.append(row_data)
 
        results_df = pd.DataFrame(results)
        if results_df.empty:
            print("⚠️ Merged results are empty. "
                f"{'Sequential fallback was used.' if use_seq_fallback else 'No fallback used.'}")
        if output_csv_path:
            results_df.to_csv(output_csv_path, index=False, encoding="utf-8")
            print(f"✅ Results saved to: {output_csv_path}")
        return results_df
    
    def run_complete_batch_pipeline(
        self,
        data: pd.DataFrame,
        system_prompt: str,
        user_prompt_template: str,
        text_column: str,
        response_schema: Dict[str, Any],
        output_dir: Path,
        job_name: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1000,
        poll_interval: int = 30
    ) -> pd.DataFrame:
        """
        Run the complete batch processing pipeline.
        
        Args:
            data: DataFrame to process
            system_prompt: System prompt for the model
            user_prompt_template: User prompt template
            text_column: Column containing text to process
            response_schema: JSON schema for structured response
            output_dir: Directory to save files
            job_name: Name for this batch job
            model: OpenAI model to use
            max_tokens: Maximum tokens for response
            poll_interval: Seconds between status checks
            
        Returns:
            DataFrame with processed results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Prepare batch file
        input_file = output_dir / f"{job_name}_input.jsonl"
        self.prepare_batch_file(
            data=data,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            text_column=text_column,
            response_schema=response_schema,
            model=model,
            output_path=input_file,
            max_tokens=max_tokens,
            custom_id_prefix=job_name
        )
        
        # Step 2: Submit batch
        batch_id = self.submit_batch(input_file, f"BiodiversityASSET {job_name}")
        
        # Step 3: Wait for completion
        output_file_id = self.wait_for_completion(batch_id, poll_interval)
        
        # Step 4: Download results
        results_file = output_dir / f"{job_name}_results.jsonl"
        self.download_results(output_file_id, results_file)
        
        # Step 5: Process results
        output_csv = output_dir / f"{job_name}_processed.csv"
        results_df = self.process_batch_results(results_file, data, output_csv)
        
        return results_df
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for processing."""
        return str(text).replace("\n", " ").replace("\r", " ").strip()


# Pydantic schemas for different processing steps
class InvestmentActivityResponse(BaseModel):
    """Response schema for investment activity classification."""
    Score: int = Field(..., description="1 if describes concrete biodiversity investment activities, 0 if generic/policy references only")
    Explanation: str = Field(..., description="Brief explanation of the score")

class AssetizationFeaturesResponse(BaseModel):
    """Response schema for assetization features scoring."""
    Explanation: str = Field(..., description="Explanation the three scores")
    IntrinsicValueScore: int = Field(..., description="Score for intrinsic value dimension (0-3)")
    CashFlowScore: int = Field(..., description="Score for cash flow dimension (0-3)")
    OwnershipControlScore: int = Field(..., description="Score for ownership/control dimension (0-3)")


def get_response_schema(response_class: BaseModel) -> Dict[str, Any]:
    """Get JSON schema from Pydantic model."""
    try:
        from openai.lib._pydantic import to_strict_json_schema
        return to_strict_json_schema(response_class)
    except ImportError:
        # Fallback for different OpenAI library versions
        return response_class.model_json_schema()