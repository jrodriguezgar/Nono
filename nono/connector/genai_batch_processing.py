# -*- coding: utf-8 -*-
"""
Batch Processing Module - Gemini and OpenAI Batch APIs

Provides asynchronous large-scale processing capabilities using the Gemini and OpenAI
Batch APIs. Process large volumes of requests at 50% of the standard cost with a 
target response time of 24 hours (usually much faster).

Supported Providers:
    - Google Gemini: GeminiBatchService
    - OpenAI: OpenAIBatchService

Features:
    - Inline batch processing for small sets (< 20 MB for Gemini)
    - File-based batch processing for large sets (< 2 GB for Gemini, < 200 MB for OpenAI)
    - Job monitoring with polling and callbacks
    - Helper methods for building requests and JSONL files

Example (Gemini):
    >>> from genai_batch_processing import GeminiBatchService
    >>> 
    >>> batch_service = GeminiBatchService(api_key="your-api-key")
    >>> requests = GeminiBatchService.build_requests_from_prompts(
    ...     prompts=["Question 1?", "Question 2?"],
    ...     system_instruction="Answer briefly."
    ... )
    >>> job = batch_service.create_inline_batch(requests, display_name="my-batch")
    >>> completed = batch_service.wait_for_completion(job.name)
    >>> results = batch_service.get_inline_results(job.name)

Example (OpenAI):
    >>> from genai_batch_processing import OpenAIBatchService
    >>> 
    >>> batch_service = OpenAIBatchService(api_key="your-api-key")
    >>> requests = OpenAIBatchService.build_requests_from_prompts(
    ...     prompts=["Question 1?", "Question 2?"],
    ...     system_prompt="Answer briefly."
    ... )
    >>> batch_service.create_jsonl_file(requests, "batch_input.jsonl")
    >>> job = batch_service.create_batch("batch_input.jsonl")
    >>> completed = batch_service.wait_for_completion(job.id)
    >>> results = batch_service.get_results(job.id)

Author: DatamanEdge
License: MIT
Date: 2026-02-05
Version: 2.0.0
"""

import json
import time
from datetime import datetime
from typing import Any, Callable

# Import install_library from connector_genai for dependency management
try:
    from .connector_genai import install_library
except ImportError:
    from connector_genai import install_library


class GeminiBatchService:
    """
    Google Gemini Batch API service for asynchronous large-scale processing.
    
    The Batch API is designed to process large volumes of requests asynchronously
    at 50% of the standard cost. Target response time is 24 hours, but usually
    much faster (jobs expire after 48 hours).
    
    Features:
        - Inline batch processing for small sets (< 20 MB)
        - File-based batch processing for large sets (< 2 GB)
        - Support for all Gemini configurations (temperature, system instructions, etc.)
        - Google Search tool integration
        - Structured JSON output with response schemas
        - Image generation batch processing
    
    Attributes:
        STATE_PENDING: Job is queued, waiting for processing
        STATE_RUNNING: Job is currently being processed
        STATE_SUCCEEDED: Job completed successfully
        STATE_FAILED: Job failed
        STATE_CANCELLED: Job was cancelled
        STATE_EXPIRED: Job expired (ran or was pending for more than 48 hours)
    
    Example (basic):
        >>> batch_service = GeminiBatchService(api_key="your-api-key")
        >>> requests = GeminiBatchService.build_requests_from_prompts(
        ...     prompts=["Question 1?", "Question 2?"],
        ...     system_instruction="Answer briefly."
        ... )
        >>> job = batch_service.create_inline_batch(requests)
        >>> completed = batch_service.wait_for_completion(job.name)
        >>> results = batch_service.get_inline_results(job.name)
    
    Example (with Google Search):
        >>> request = GeminiBatchService.build_request_with_search(
        ...     prompt="Who won Euro 2024?"
        ... )
    
    Example (structured output):
        >>> request = GeminiBatchService.build_structured_request(
        ...     prompt="List 3 recipes",
        ...     response_schema=list[Recipe]
        ... )
    
    Example (image generation):
        >>> request = GeminiBatchService.build_image_request(
        ...     prompt="A cat playing piano"
        ... )
        >>> job = batch_service.create_inline_batch(
        ...     [request], model_name="gemini-3-pro-image-preview"
        ... )
    """
    
    # Batch job states
    STATE_PENDING = "JOB_STATE_PENDING"
    STATE_RUNNING = "JOB_STATE_RUNNING"
    STATE_SUCCEEDED = "JOB_STATE_SUCCEEDED"
    STATE_FAILED = "JOB_STATE_FAILED"
    STATE_CANCELLED = "JOB_STATE_CANCELLED"
    STATE_EXPIRED = "JOB_STATE_EXPIRED"
    
    def __init__(self, api_key: str | None = None):
        """
        Initialize the Gemini Batch Service.
        
        Args:
            api_key: Google API key for authentication. If None, will attempt
                    to use environment variable or default credentials.
        
        Raises:
            ImportError: If google-genai library cannot be installed
        """
        if not install_library("google.genai", package_name="google-genai"):
            raise ImportError(
                "The 'google-genai' library is required for GeminiBatchService "
                "and could not be installed."
            )
        
        from google import genai
        from google.genai import types
        
        self.genai = genai
        self.types = types
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
    
    def create_inline_batch(
        self,
        requests: list[dict],
        model_name: str = "gemini-3-flash-preview",
        display_name: str | None = None,
        config: dict | None = None
    ) -> Any:
        """
        Create a batch job with inline requests.
        
        Suitable for smaller batches that keep total request size under 20 MB.
        The output returned by the model is a list of inlineResponse objects.
        
        Args:
            requests: List of GenerateContentRequest dictionaries. Each dict should
                     contain 'contents' with the conversation structure.
            model_name: The model to use for processing (default: gemini-3-flash-preview)
            display_name: Optional human-readable name for the job
            config: Optional additional configuration parameters
        
        Returns:
            BatchJob: The created batch job object with job name and status
        
        Example:
            >>> requests = [
            ...     {"contents": [{"parts": [{"text": "Hello!"}], "role": "user"}]},
            ...     {"contents": [{"parts": [{"text": "What is 2+2?"}], "role": "user"}]}
            ... ]
            >>> job = batch_service.create_inline_batch(requests, display_name="my-job")
            >>> print(f"Created job: {job.name}")
        """
        batch_config = {"display_name": display_name or f"inline-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"}
        if config:
            batch_config.update(config)
        
        batch_job = self.client.batches.create(
            model=f"models/{model_name}" if not model_name.startswith("models/") else model_name,
            src=requests,
            config=batch_config
        )
        
        return batch_job
    
    def create_file_batch(
        self,
        file_path: str | None = None,
        uploaded_file_name: str | None = None,
        model_name: str = "gemini-3-flash-preview",
        display_name: str | None = None,
        config: dict | None = None
    ) -> Any:
        """
        Create a batch job from a JSONL file.
        
        Recommended for larger request sets. Each line in the JSONL file should be
        a JSON object with 'key' and 'request' fields. Maximum file size is 2 GB.
        
        Args:
            file_path: Local path to the JSONL file to upload. Mutually exclusive
                      with uploaded_file_name.
            uploaded_file_name: Name of an already-uploaded file via the File API.
                               Mutually exclusive with file_path.
            model_name: The model to use for processing (default: gemini-3-flash-preview)
            display_name: Optional human-readable name for the job
            config: Optional additional configuration parameters
        
        Returns:
            BatchJob: The created batch job object with job name and status
        
        JSONL File Format:
            Each line should be: {"key": "unique-id", "request": {GenerateContentRequest}}
            
            Example line:
            {"key": "req-1", "request": {"contents": [{"parts": [{"text": "Hello"}]}]}}
        
        Example:
            >>> # Using local file
            >>> job = batch_service.create_file_batch(
            ...     file_path="requests.jsonl",
            ...     display_name="large-batch-job"
            ... )
            >>> 
            >>> # Using already uploaded file
            >>> job = batch_service.create_file_batch(
            ...     uploaded_file_name="files/abc123",
            ...     display_name="from-uploaded-file"
            ... )
        """
        if file_path and uploaded_file_name:
            raise ValueError("Provide either file_path or uploaded_file_name, not both.")
        if not file_path and not uploaded_file_name:
            raise ValueError("Either file_path or uploaded_file_name must be provided.")
        
        # Upload file if local path provided
        if file_path:
            uploaded_file = self.client.files.upload(
                file=file_path,
                config=self.types.UploadFileConfig(
                    display_name=display_name or file_path.split("/")[-1].split("\\")[-1],
                    mime_type="application/jsonl"
                )
            )
            file_name = uploaded_file.name
        else:
            file_name = uploaded_file_name
        
        batch_config = {"display_name": display_name or f"file-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"}
        if config:
            batch_config.update(config)
        
        batch_job = self.client.batches.create(
            model=f"models/{model_name}" if not model_name.startswith("models/") else model_name,
            src=file_name,
            config=batch_config
        )
        
        return batch_job
    
    def create_jsonl_file(
        self,
        requests: list[dict],
        output_path: str,
        keys: list[str] | None = None
    ) -> str:
        """
        Helper method to create a JSONL file from a list of requests.
        
        Args:
            requests: List of GenerateContentRequest dictionaries
            output_path: Path where the JSONL file will be saved
            keys: Optional list of unique keys for each request. If None,
                 auto-generates keys as "request-1", "request-2", etc.
        
        Returns:
            str: The path to the created JSONL file
        
        Example:
            >>> requests = [
            ...     {"contents": [{"parts": [{"text": "Question 1"}]}]},
            ...     {"contents": [{"parts": [{"text": "Question 2"}]}]}
            ... ]
            >>> path = batch_service.create_jsonl_file(
            ...     requests, 
            ...     "my-requests.jsonl",
            ...     keys=["q1", "q2"]
            ... )
        """
        if keys and len(keys) != len(requests):
            raise ValueError(f"Number of keys ({len(keys)}) must match number of requests ({len(requests)})")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for i, request in enumerate(requests):
                key = keys[i] if keys else f"request-{i+1}"
                line = {"key": key, "request": request}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        
        return output_path
    
    def get_job(self, job_name: str) -> Any:
        """
        Get the current state of a batch job.
        
        Args:
            job_name: The name/ID of the batch job (returned when creating the job)
        
        Returns:
            BatchJob: The batch job object with current status and metadata
        """
        return self.client.batches.get(name=job_name)
    
    def get_job_status(self, job_name: str) -> str:
        """
        Get the current status of a batch job.
        
        Args:
            job_name: The name/ID of the batch job
        
        Returns:
            str: Status string (JOB_STATE_PENDING, JOB_STATE_RUNNING, 
                 JOB_STATE_SUCCEEDED, JOB_STATE_FAILED, JOB_STATE_CANCELLED,
                 JOB_STATE_EXPIRED)
        """
        job = self.get_job(job_name)
        return job.state
    
    def wait_for_completion(
        self,
        job_name: str,
        poll_interval: float = 30.0,
        timeout: float | None = None,
        callback: Callable[[Any, float], None] | None = None
    ) -> Any:
        """
        Wait for a batch job to complete, polling periodically.
        
        Args:
            job_name: The name/ID of the batch job
            poll_interval: Seconds between status checks (default: 30)
            timeout: Maximum seconds to wait. None for no timeout (default: None)
            callback: Optional function called on each poll with (job, elapsed_seconds)
        
        Returns:
            BatchJob: The completed job object
        
        Raises:
            TimeoutError: If timeout is reached before job completes
        
        Example:
            >>> def progress(job, elapsed):
            ...     print(f"Status: {job.state} after {elapsed:.0f}s")
            >>> 
            >>> completed_job = batch_service.wait_for_completion(
            ...     job.name,
            ...     poll_interval=60,
            ...     timeout=3600,
            ...     callback=progress
            ... )
        """
        start_time = time.time()
        terminal_states = {
            self.STATE_SUCCEEDED, 
            self.STATE_FAILED, 
            self.STATE_CANCELLED,
            self.STATE_EXPIRED
        }
        
        while True:
            job = self.get_job(job_name)
            elapsed = time.time() - start_time
            
            if callback:
                callback(job, elapsed)
            
            if job.state in terminal_states:
                return job
            
            if timeout and elapsed >= timeout:
                raise TimeoutError(
                    f"Batch job '{job_name}' did not complete within {timeout} seconds. "
                    f"Current state: {job.state}"
                )
            
            time.sleep(poll_interval)
    
    def get_inline_results(self, job_name: str) -> list[dict]:
        """
        Get results from an inline batch job.
        
        Args:
            job_name: The name/ID of the completed batch job
        
        Returns:
            list[dict]: List of response objects from the batch
        
        Raises:
            ValueError: If the job has not completed successfully
        """
        job = self.get_job(job_name)
        
        if job.state != self.STATE_SUCCEEDED:
            raise ValueError(
                f"Cannot get results from job with state '{job.state}'. "
                f"Job must be in '{self.STATE_SUCCEEDED}' state."
            )
        
        # For inline jobs, results are in the response attribute
        if hasattr(job, 'response') and job.response:
            return job.response.inline_responses if hasattr(job.response, 'inline_responses') else []
        
        return []
    
    def get_file_results(self, job_name: str, output_path: str | None = None) -> list[dict] | str:
        """
        Get results from a file-based batch job.
        
        For file-based jobs, results are returned as a JSONL file. This method
        can either download and parse the results or save them to a file.
        
        Args:
            job_name: The name/ID of the completed batch job
            output_path: Optional path to save the results file. If None,
                        results are parsed and returned as a list.
        
        Returns:
            list[dict]: Parsed results if output_path is None
            str: Path to saved file if output_path is provided
        
        Raises:
            ValueError: If the job has not completed successfully
        """
        job = self.get_job(job_name)
        
        if job.state != self.STATE_SUCCEEDED:
            raise ValueError(
                f"Cannot get results from job with state '{job.state}'. "
                f"Job must be in '{self.STATE_SUCCEEDED}' state."
            )
        
        # For file jobs, results are in a destination file
        if hasattr(job, 'dest') and job.dest:
            result_file_name = job.dest
            
            # Download the result file using the File API
            result_content = self.client.files.download(file=result_file_name)
            
            if output_path:
                # Save to file
                with open(output_path, "wb") as f:
                    f.write(result_content)
                return output_path
            else:
                # Parse JSONL and return
                results = []
                for line in result_content.decode('utf-8').strip().split('\n'):
                    if line:
                        results.append(json.loads(line))
                return results
        
        return []
    
    def cancel_job(self, job_name: str) -> Any:
        """
        Cancel a running or pending batch job.
        
        Args:
            job_name: The name/ID of the batch job to cancel
        
        Returns:
            BatchJob: The cancelled job object
        """
        return self.client.batches.cancel(name=job_name)
    
    def list_jobs(self, page_size: int = 100) -> list:
        """
        List all batch jobs.
        
        Args:
            page_size: Maximum number of jobs to return per page
        
        Returns:
            list: List of batch job objects
        """
        return list(self.client.batches.list(config={"page_size": page_size}))
    
    def delete_job(self, job_name: str) -> None:
        """
        Delete a batch job.
        
        Args:
            job_name: The name/ID of the batch job to delete
        """
        self.client.batches.delete(name=job_name)
    
    @staticmethod
    def build_request(
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        tools: list[dict] | None = None,
        response_mime_type: str | None = None,
        response_schema: Any | None = None,
        response_modalities: list[str] | None = None
    ) -> dict:
        """
        Helper method to build a GenerateContentRequest dictionary.
        
        Supports all Gemini configuration options including tools, structured output,
        and multimodal responses.
        
        Args:
            prompt: The user prompt/question
            system_instruction: Optional system instruction
            temperature: Optional temperature setting (0.0 to 2.0)
            max_tokens: Optional maximum output tokens
            top_p: Optional top_p sampling parameter
            top_k: Optional top_k sampling parameter
            stop_sequences: Optional list of stop sequences
            tools: Optional list of tools to use (e.g., [{"google_search": {}}])
            response_mime_type: Optional response MIME type (e.g., "application/json")
            response_schema: Optional response schema for structured output (Pydantic model or dict)
            response_modalities: Optional list of response modalities (e.g., ["TEXT", "IMAGE"])
        
        Returns:
            dict: A properly formatted GenerateContentRequest
        
        Example (basic):
            >>> request = GeminiBatchService.build_request(
            ...     prompt="What is machine learning?",
            ...     system_instruction="You are a helpful tutor.",
            ...     temperature=0.7
            ... )
        
        Example (with tools):
            >>> request = GeminiBatchService.build_request(
            ...     prompt="Who won Euro 2024?",
            ...     tools=[{"google_search": {}}]
            ... )
        
        Example (structured output):
            >>> request = GeminiBatchService.build_request(
            ...     prompt="List 3 cookie recipes with ingredients",
            ...     response_mime_type="application/json",
            ...     response_schema={"type": "array", "items": {"type": "object"}}
            ... )
        
        Example (image generation):
            >>> request = GeminiBatchService.build_request(
            ...     prompt="Draw a cat playing piano",
            ...     response_modalities=["TEXT", "IMAGE"]
            ... )
        """
        request: dict = {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                    "role": "user"
                }
            ]
        }
        
        # Build config dict for advanced settings
        config: dict = {}
        
        # Add generation config if any parameters provided
        generation_config: dict = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        if top_p is not None:
            generation_config["top_p"] = top_p
        if top_k is not None:
            generation_config["top_k"] = top_k
        if stop_sequences is not None:
            generation_config["stop_sequences"] = stop_sequences
        if response_modalities is not None:
            generation_config["responseModalities"] = response_modalities
        
        if generation_config:
            request["generation_config"] = generation_config
        
        # Add system instruction to config
        if system_instruction:
            config["system_instruction"] = {"parts": [{"text": system_instruction}]}
        
        # Add tools to config
        if tools is not None:
            config["tools"] = tools
        
        # Add structured output settings to config
        if response_mime_type is not None:
            config["response_mime_type"] = response_mime_type
        if response_schema is not None:
            config["response_schema"] = response_schema
        
        # Add config to request if any settings were added
        if config:
            request["config"] = config
        
        return request
    
    @staticmethod
    def build_request_with_search(
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> dict:
        """
        Build a request with Google Search tool enabled.
        
        Useful for queries that require up-to-date information from the web.
        
        Args:
            prompt: The user prompt/question
            system_instruction: Optional system instruction
            temperature: Optional temperature setting
            max_tokens: Optional maximum output tokens
        
        Returns:
            dict: A GenerateContentRequest with Google Search enabled
        
        Example:
            >>> request = GeminiBatchService.build_request_with_search(
            ...     prompt="What are the latest news about AI?"
            ... )
        """
        return GeminiBatchService.build_request(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=[{"google_search": {}}]
        )
    
    @staticmethod
    def build_structured_request(
        prompt: str,
        response_schema: Any,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> dict:
        """
        Build a request for structured JSON output.
        
        The response will be formatted according to the provided schema.
        
        Args:
            prompt: The user prompt/question
            response_schema: Schema for the response. Can be:
                            - A Pydantic model class (e.g., list[Recipe])
                            - A dict with JSON Schema format
            system_instruction: Optional system instruction
            temperature: Optional temperature setting
            max_tokens: Optional maximum output tokens
        
        Returns:
            dict: A GenerateContentRequest configured for structured output
        
        Example (with Pydantic):
            >>> from pydantic import BaseModel
            >>> class Recipe(BaseModel):
            ...     name: str
            ...     ingredients: list[str]
            >>> 
            >>> request = GeminiBatchService.build_structured_request(
            ...     prompt="List 3 cookie recipes",
            ...     response_schema=list[Recipe]
            ... )
        
        Example (with dict schema):
            >>> schema = {
            ...     "type": "array",
            ...     "items": {
            ...         "type": "object",
            ...         "properties": {
            ...             "name": {"type": "string"},
            ...             "ingredients": {"type": "array", "items": {"type": "string"}}
            ...         }
            ...     }
            ... }
            >>> request = GeminiBatchService.build_structured_request(
            ...     prompt="List 3 cookie recipes",
            ...     response_schema=schema
            ... )
        """
        return GeminiBatchService.build_request(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=response_schema
        )
    
    @staticmethod
    def build_image_request(
        prompt: str,
        system_instruction: str | None = None,
        include_text: bool = True
    ) -> dict:
        """
        Build a request for image generation.
        
        Use with models that support image generation (e.g., gemini-3-pro-image-preview).
        
        Args:
            prompt: Description of the image to generate
            system_instruction: Optional system instruction
            include_text: Whether to include text in the response (default: True)
        
        Returns:
            dict: A GenerateContentRequest configured for image generation
        
        Example:
            >>> request = GeminiBatchService.build_image_request(
            ...     prompt="A cat playing piano in a cozy room"
            ... )
            >>> # Use with: model_name="gemini-3-pro-image-preview"
        """
        modalities = ["TEXT", "IMAGE"] if include_text else ["IMAGE"]
        return GeminiBatchService.build_request(
            prompt=prompt,
            system_instruction=system_instruction,
            response_modalities=modalities
        )
    
    @staticmethod
    def build_requests_from_prompts(
        prompts: list[str],
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        response_mime_type: str | None = None,
        response_schema: Any | None = None,
        response_modalities: list[str] | None = None
    ) -> list[dict]:
        """
        Convenience method to build multiple requests from a list of prompts.
        
        All configuration options are applied uniformly to all requests.
        
        Args:
            prompts: List of user prompts
            system_instruction: Optional system instruction (applied to all)
            temperature: Optional temperature (applied to all)
            max_tokens: Optional max tokens (applied to all)
            tools: Optional tools list (applied to all)
            response_mime_type: Optional response MIME type (applied to all)
            response_schema: Optional response schema (applied to all)
            response_modalities: Optional response modalities (applied to all)
        
        Returns:
            list[dict]: List of GenerateContentRequest dictionaries
        
        Example:
            >>> prompts = ["Question 1?", "Question 2?", "Question 3?"]
            >>> requests = GeminiBatchService.build_requests_from_prompts(
            ...     prompts,
            ...     system_instruction="Answer briefly.",
            ...     temperature=0.5
            ... )
        
        Example (with structured output):
            >>> requests = GeminiBatchService.build_requests_from_prompts(
            ...     prompts=["List 3 fruits", "List 3 vegetables"],
            ...     response_mime_type="application/json",
            ...     response_schema={"type": "array", "items": {"type": "string"}}
            ... )
        """
        return [
            GeminiBatchService.build_request(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
                response_modalities=response_modalities
            )
            for prompt in prompts
        ]


class OpenAIBatchService:
    """
    OpenAI Batch API service for asynchronous large-scale processing.
    
    The Batch API offers 50% cost discount compared to synchronous APIs,
    substantially higher rate limits, and 24-hour turnaround time.
    
    Supported endpoints:
        - /v1/chat/completions (Chat Completions API)
        - /v1/responses (Responses API)
        - /v1/embeddings (Embeddings API)
        - /v1/completions (Completions API)
        - /v1/moderations (Moderations API)
    
    Attributes:
        STATE_VALIDATING: Input file is being validated
        STATE_FAILED: Input file failed validation
        STATE_IN_PROGRESS: Batch is currently being processed
        STATE_FINALIZING: Batch completed, results being prepared
        STATE_COMPLETED: Batch completed, results ready
        STATE_EXPIRED: Batch did not complete within 24 hours
        STATE_CANCELLING: Batch is being cancelled
        STATE_CANCELLED: Batch was cancelled
    
    Example:
        >>> batch_service = OpenAIBatchService(api_key="your-api-key")
        >>> 
        >>> # Build requests
        >>> requests = OpenAIBatchService.build_requests_from_prompts(
        ...     prompts=["What is AI?", "Explain ML"],
        ...     model="gpt-4o-mini",
        ...     system_prompt="Answer concisely."
        ... )
        >>> 
        >>> # Create JSONL and upload
        >>> batch_service.create_jsonl_file(requests, "input.jsonl")
        >>> batch = batch_service.create_batch("input.jsonl")
        >>> 
        >>> # Wait and get results
        >>> completed = batch_service.wait_for_completion(batch.id)
        >>> results = batch_service.get_results(batch.id)
    """
    
    # Batch job states
    STATE_VALIDATING = "validating"
    STATE_FAILED = "failed"
    STATE_IN_PROGRESS = "in_progress"
    STATE_FINALIZING = "finalizing"
    STATE_COMPLETED = "completed"
    STATE_EXPIRED = "expired"
    STATE_CANCELLING = "cancelling"
    STATE_CANCELLED = "cancelled"
    
    # Supported endpoints
    ENDPOINT_CHAT_COMPLETIONS = "/v1/chat/completions"
    ENDPOINT_RESPONSES = "/v1/responses"
    ENDPOINT_EMBEDDINGS = "/v1/embeddings"
    ENDPOINT_COMPLETIONS = "/v1/completions"
    ENDPOINT_MODERATIONS = "/v1/moderations"
    
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Initialize the OpenAI Batch Service.
        
        Args:
            api_key: OpenAI API key for authentication. If None, will attempt
                    to use OPENAI_API_KEY environment variable.
            base_url: Optional custom base URL for API requests.
        
        Raises:
            ImportError: If openai library cannot be installed
        """
        if not install_library("openai"):
            raise ImportError(
                "The 'openai' library is required for OpenAIBatchService "
                "and could not be installed."
            )
        
        from openai import OpenAI
        
        self.api_key = api_key
        self.base_url = base_url
        
        client_kwargs: dict = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = OpenAI(**client_kwargs)
    
    def upload_file(self, file_path: str) -> Any:
        """
        Upload a JSONL file for batch processing.
        
        Args:
            file_path: Local path to the JSONL file to upload
        
        Returns:
            FileObject: The uploaded file object with file ID
        
        Example:
            >>> file = batch_service.upload_file("batch_input.jsonl")
            >>> print(f"File ID: {file.id}")
        """
        with open(file_path, "rb") as f:
            uploaded_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        return uploaded_file
    
    def create_batch(
        self,
        file_path: str | None = None,
        input_file_id: str | None = None,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: dict | None = None
    ) -> Any:
        """
        Create a batch processing job.
        
        Args:
            file_path: Local path to the JSONL file. Will be uploaded automatically.
                      Mutually exclusive with input_file_id.
            input_file_id: ID of an already-uploaded file. Mutually exclusive
                          with file_path.
            endpoint: API endpoint to use. One of:
                     - /v1/chat/completions (default)
                     - /v1/responses
                     - /v1/embeddings
                     - /v1/completions
                     - /v1/moderations
            completion_window: Time window for completion. Currently only "24h".
            metadata: Optional metadata dict for the batch.
        
        Returns:
            Batch: The created batch object with batch ID and status
        
        Example:
            >>> # Using local file (auto-upload)
            >>> batch = batch_service.create_batch(
            ...     file_path="batch_input.jsonl",
            ...     endpoint="/v1/chat/completions",
            ...     metadata={"description": "nightly eval job"}
            ... )
            >>> 
            >>> # Using already uploaded file
            >>> batch = batch_service.create_batch(
            ...     input_file_id="file-abc123",
            ...     endpoint="/v1/chat/completions"
            ... )
        """
        if file_path and input_file_id:
            raise ValueError("Provide either file_path or input_file_id, not both.")
        if not file_path and not input_file_id:
            raise ValueError("Either file_path or input_file_id must be provided.")
        
        # Upload file if local path provided
        if file_path:
            uploaded_file = self.upload_file(file_path)
            file_id = uploaded_file.id
        else:
            file_id = input_file_id
        
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata
        )
        
        return batch
    
    def get_batch(self, batch_id: str) -> Any:
        """
        Get the current state of a batch job.
        
        Args:
            batch_id: The ID of the batch (e.g., "batch_abc123")
        
        Returns:
            Batch: The batch object with current status and metadata
        """
        return self.client.batches.retrieve(batch_id)
    
    def get_batch_status(self, batch_id: str) -> str:
        """
        Get the current status of a batch job.
        
        Args:
            batch_id: The ID of the batch
        
        Returns:
            str: Status string (validating, failed, in_progress, finalizing,
                 completed, expired, cancelling, cancelled)
        """
        batch = self.get_batch(batch_id)
        return batch.status
    
    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: float = 30.0,
        timeout: float | None = None,
        callback: Callable[[Any, float], None] | None = None
    ) -> Any:
        """
        Wait for a batch job to complete, polling periodically.
        
        Args:
            batch_id: The ID of the batch
            poll_interval: Seconds between status checks (default: 30)
            timeout: Maximum seconds to wait. None for no timeout (default: None)
            callback: Optional function called on each poll with (batch, elapsed_seconds)
        
        Returns:
            Batch: The completed batch object
        
        Raises:
            TimeoutError: If timeout is reached before batch completes
        
        Example:
            >>> def progress(batch, elapsed):
            ...     print(f"Status: {batch.status} after {elapsed:.0f}s")
            ...     if batch.request_counts:
            ...         print(f"  Completed: {batch.request_counts.completed}/{batch.request_counts.total}")
            >>> 
            >>> completed = batch_service.wait_for_completion(
            ...     batch.id,
            ...     poll_interval=60,
            ...     timeout=3600,
            ...     callback=progress
            ... )
        """
        start_time = time.time()
        terminal_states = {
            self.STATE_COMPLETED, 
            self.STATE_FAILED, 
            self.STATE_CANCELLED,
            self.STATE_EXPIRED
        }
        
        while True:
            batch = self.get_batch(batch_id)
            elapsed = time.time() - start_time
            
            if callback:
                callback(batch, elapsed)
            
            if batch.status in terminal_states:
                return batch
            
            if timeout and elapsed >= timeout:
                raise TimeoutError(
                    f"Batch '{batch_id}' did not complete within {timeout} seconds. "
                    f"Current status: {batch.status}"
                )
            
            time.sleep(poll_interval)
    
    def get_results(
        self,
        batch_id: str,
        output_path: str | None = None
    ) -> list[dict] | str:
        """
        Get results from a completed batch.
        
        Args:
            batch_id: The ID of the completed batch
            output_path: Optional path to save results file. If None,
                        results are parsed and returned as a list.
        
        Returns:
            list[dict]: Parsed results if output_path is None
            str: Path to saved file if output_path is provided
        
        Raises:
            ValueError: If the batch has not completed successfully
        
        Example:
            >>> # Get as list
            >>> results = batch_service.get_results(batch.id)
            >>> for r in results:
            ...     print(f"ID: {r['custom_id']}, Response: {r['response']}")
            >>> 
            >>> # Save to file
            >>> path = batch_service.get_results(batch.id, "output.jsonl")
        """
        batch = self.get_batch(batch_id)
        
        if batch.status != self.STATE_COMPLETED:
            raise ValueError(
                f"Cannot get results from batch with status '{batch.status}'. "
                f"Batch must be in '{self.STATE_COMPLETED}' status."
            )
        
        if not batch.output_file_id:
            raise ValueError("Batch has no output file. Results may have been deleted.")
        
        # Download output file content
        content = self.client.files.content(batch.output_file_id)
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(content.content)
            return output_path
        
        # Parse JSONL and return
        results = []
        for line in content.text.strip().split('\n'):
            if line:
                results.append(json.loads(line))
        return results
    
    def get_errors(self, batch_id: str, output_path: str | None = None) -> list[dict] | str | None:
        """
        Get errors from a batch (if any).
        
        Args:
            batch_id: The ID of the batch
            output_path: Optional path to save error file. If None,
                        errors are parsed and returned as a list.
        
        Returns:
            list[dict]: Parsed errors if output_path is None and errors exist
            str: Path to saved file if output_path is provided
            None: If no error file exists
        """
        batch = self.get_batch(batch_id)
        
        if not batch.error_file_id:
            return None
        
        content = self.client.files.content(batch.error_file_id)
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(content.content)
            return output_path
        
        # Parse JSONL and return
        errors = []
        for line in content.text.strip().split('\n'):
            if line:
                errors.append(json.loads(line))
        return errors
    
    def cancel_batch(self, batch_id: str) -> Any:
        """
        Cancel an ongoing batch. May take up to 10 minutes.
        
        Args:
            batch_id: The ID of the batch to cancel
        
        Returns:
            Batch: The cancelled batch object
        """
        return self.client.batches.cancel(batch_id)
    
    def list_batches(self, limit: int = 20, after: str | None = None) -> list:
        """
        List all batches.
        
        Args:
            limit: Maximum number of batches to return (default: 20)
            after: Cursor for pagination (batch ID to start after)
        
        Returns:
            list: List of batch objects
        """
        kwargs: dict = {"limit": limit}
        if after:
            kwargs["after"] = after
        return list(self.client.batches.list(**kwargs))
    
    def create_jsonl_file(
        self,
        requests: list[dict],
        output_path: str
    ) -> str:
        """
        Create a JSONL file from a list of batch requests.
        
        Each request should have the format:
        {
            "custom_id": "unique-id",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {...}
        }
        
        Args:
            requests: List of batch request dictionaries
            output_path: Path where the JSONL file will be saved
        
        Returns:
            str: The path to the created JSONL file
        
        Example:
            >>> requests = OpenAIBatchService.build_requests_from_prompts(
            ...     prompts=["Q1?", "Q2?"],
            ...     model="gpt-4o-mini"
            ... )
            >>> path = batch_service.create_jsonl_file(requests, "input.jsonl")
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for request in requests:
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
        
        return output_path
    
    @staticmethod
    def build_request(
        prompt: str,
        model: str = "gpt-4o-mini",
        custom_id: str | None = None,
        system_prompt: str | None = None,
        endpoint: str = "/v1/chat/completions",
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        response_format: dict | None = None
    ) -> dict:
        """
        Build a batch request dictionary for chat completions.
        
        Args:
            prompt: The user message/prompt
            model: Model to use (default: gpt-4o-mini)
            custom_id: Unique identifier for this request. Auto-generated if None.
            system_prompt: Optional system message
            endpoint: API endpoint (default: /v1/chat/completions)
            temperature: Optional temperature setting
            max_tokens: Optional maximum tokens
            top_p: Optional top_p sampling parameter
            frequency_penalty: Optional frequency penalty
            presence_penalty: Optional presence penalty
            response_format: Optional response format (e.g., {"type": "json_object"})
        
        Returns:
            dict: A properly formatted batch request
        
        Example:
            >>> request = OpenAIBatchService.build_request(
            ...     prompt="What is machine learning?",
            ...     model="gpt-4o-mini",
            ...     custom_id="ml-question-1",
            ...     system_prompt="You are a helpful tutor.",
            ...     temperature=0.7
            ... )
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build body
        body: dict = {
            "model": model,
            "messages": messages
        }
        
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if response_format is not None:
            body["response_format"] = response_format
        
        # Generate custom_id if not provided
        if custom_id is None:
            custom_id = f"request-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": endpoint,
            "body": body
        }
    
    @staticmethod
    def build_requests_from_prompts(
        prompts: list[str],
        model: str = "gpt-4o-mini",
        system_prompt: str | None = None,
        endpoint: str = "/v1/chat/completions",
        temperature: float | None = None,
        max_tokens: int | None = None,
        custom_id_prefix: str = "request"
    ) -> list[dict]:
        """
        Build multiple batch requests from a list of prompts.
        
        Args:
            prompts: List of user prompts
            model: Model to use (default: gpt-4o-mini)
            system_prompt: Optional system prompt (applied to all)
            endpoint: API endpoint (default: /v1/chat/completions)
            temperature: Optional temperature (applied to all)
            max_tokens: Optional max tokens (applied to all)
            custom_id_prefix: Prefix for auto-generated IDs (default: "request")
        
        Returns:
            list[dict]: List of batch request dictionaries
        
        Example:
            >>> prompts = ["Question 1?", "Question 2?", "Question 3?"]
            >>> requests = OpenAIBatchService.build_requests_from_prompts(
            ...     prompts,
            ...     model="gpt-4o-mini",
            ...     system_prompt="Answer briefly.",
            ...     temperature=0.5
            ... )
        """
        return [
            OpenAIBatchService.build_request(
                prompt=prompt,
                model=model,
                custom_id=f"{custom_id_prefix}-{i+1}",
                system_prompt=system_prompt,
                endpoint=endpoint,
                temperature=temperature,
                max_tokens=max_tokens
            )
            for i, prompt in enumerate(prompts)
        ]
    
    @staticmethod
    def build_embedding_request(
        input_text: str | list[str],
        model: str = "text-embedding-3-small",
        custom_id: str | None = None,
        encoding_format: str | None = None,
        dimensions: int | None = None
    ) -> dict:
        """
        Build a batch request for embeddings.
        
        Args:
            input_text: Text or list of texts to embed
            model: Embedding model (default: text-embedding-3-small)
            custom_id: Unique identifier. Auto-generated if None.
            encoding_format: Optional format ("float" or "base64")
            dimensions: Optional number of dimensions
        
        Returns:
            dict: A properly formatted batch request for embeddings
        """
        body: dict = {
            "model": model,
            "input": input_text
        }
        
        if encoding_format:
            body["encoding_format"] = encoding_format
        if dimensions:
            body["dimensions"] = dimensions
        
        if custom_id is None:
            custom_id = f"embed-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": body
        }
    
    @staticmethod
    def build_moderation_request(
        input_content: str | list[dict],
        model: str = "omni-moderation-latest",
        custom_id: str | None = None
    ) -> dict:
        """
        Build a batch request for content moderation.
        
        Args:
            input_content: Text string or multimodal content array
            model: Moderation model (default: omni-moderation-latest)
            custom_id: Unique identifier. Auto-generated if None.
        
        Returns:
            dict: A properly formatted batch request for moderation
        
        Example:
            >>> # Text-only
            >>> req = OpenAIBatchService.build_moderation_request(
            ...     "This is a test sentence."
            ... )
            >>> 
            >>> # Multimodal
            >>> req = OpenAIBatchService.build_moderation_request([
            ...     {"type": "text", "text": "Describe this image"},
            ...     {"type": "image_url", "image_url": {"url": "https://..."}}
            ... ])
        """
        if custom_id is None:
            custom_id = f"mod-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/moderations",
            "body": {
                "model": model,
                "input": input_content
            }
        }


# Export for convenience
__all__ = ["GeminiBatchService", "OpenAIBatchService"]
