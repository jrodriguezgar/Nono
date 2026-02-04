"""
Data Stage Core - Core classes for data processing operations.

This module provides the core infrastructure for batch data operations
with intelligent throttling and TSV format support.

Refactored to use composition with TaskExecutor to avoid code duplication.

Author: DatamanEdge
License: MIT
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# Import from parent genai_tasker module - reuse existing infrastructure
from ..genai_tasker import (
    AIProvider, AIConfiguration, BaseAIClient,
    TaskExecutor,  # Use TaskExecutor for client management
    msg_log, event_log, logger, connector_genai
)


class DataFormat(Enum):
    """
    Enumeration of supported data formats.
    
    TSV: Tab-Separated Values - Most token-efficient for tabular data
    CSV: Comma-Separated Values - Good for simple data without commas
    PIPE: Pipe-delimited format - Alternative when tabs present in data
    JSON_COMPACT: Minimal JSON - Use when structure is complex
    """
    TSV = "tsv"           # KEY\tVALUE\tRESULT (recommended - 60% token savings)
    CSV = "csv"           # KEY,VALUE,RESULT
    PIPE = "pipe"         # KEY|VALUE|RESULT
    JSON_COMPACT = "json" # {"k":"001","v":"value"} (fallback for complex data)


class OperationType(Enum):
    """Enumeration of built-in operation types."""
    SEMANTIC_LOOKUP = "semantic_lookup"
    SPELL_CORRECTION = "spell_correction"
    CUSTOM = "custom"


@dataclass
class DataRecord:
    """
    Represents a single data record with key-value structure.
    
    The key is used for identification and result correlation.
    The value contains the data to be processed.
    The result field is populated after processing.
    
    Attributes:
        key: Unique identifier for the record (string or int).
        value: The data value to process.
        result: Processing result (populated after execution).
        metadata: Optional additional metadata dictionary.
    """
    key: Union[str, int]
    value: Any
    result: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_tsv_input(self) -> str:
        """Convert record to TSV input format (key<TAB>value)."""
        value_str = self._serialize_value(self.value)
        return f"{self.key}\t{value_str}"
    
    def to_tsv_output(self) -> str:
        """Convert record to TSV output format (key<TAB>value<TAB>result)."""
        value_str = self._serialize_value(self.value)
        result_str = self._serialize_value(self.result) if self.result is not None else ""
        return f"{self.key}\t{value_str}\t{result_str}"
    
    @staticmethod
    def _serialize_value(value: Any) -> str:
        """Serialize a value to string format, using pipe for lists."""
        if value is None:
            return ""
        if isinstance(value, list):
            return "|".join(str(v) for v in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        return str(value)
    
    @classmethod
    def from_tsv_line(cls, line: str, has_result: bool = False) -> 'DataRecord':
        """
        Parse a TSV line into a DataRecord.
        
        Args:
            line: TSV line to parse.
            has_result: Whether the line includes a result column.
        
        Returns:
            DataRecord instance.
        """
        parts = line.strip().split('\t')
        if len(parts) < 2:
            raise ValueError(f"Invalid TSV line: {line}")
        
        key = parts[0]
        value = parts[1]
        
        # Parse pipe-delimited lists
        if '|' in value and not value.startswith('{'):
            value = value.split('|')
        
        result = None
        if has_result and len(parts) >= 3:
            result = parts[2]
            if '|' in result and not result.startswith('{'):
                result = result.split('|')
        
        return cls(key=key, value=value, result=result)


@dataclass
class DataBatch:
    """
    Container for a batch of data records with throttling metadata.
    
    Attributes:
        records: List of DataRecord objects.
        batch_index: Index of this batch in the sequence.
        total_batches: Total number of batches.
        estimated_tokens: Estimated token count for this batch.
    """
    records: List[DataRecord]
    batch_index: int = 0
    total_batches: int = 1
    estimated_tokens: int = 0
    
    def to_tsv(self, include_result: bool = False) -> str:
        """
        Convert batch to TSV string.
        
        Args:
            include_result: Whether to include result column.
        
        Returns:
            TSV string representation of the batch.
        """
        if include_result:
            return '\n'.join(r.to_tsv_output() for r in self.records)
        return '\n'.join(r.to_tsv_input() for r in self.records)
    
    @classmethod
    def from_tsv(cls, tsv_string: str, has_result: bool = False) -> 'DataBatch':
        """
        Parse a TSV string into a DataBatch.
        
        Args:
            tsv_string: TSV string to parse.
            has_result: Whether the TSV includes result columns.
        
        Returns:
            DataBatch instance.
        """
        lines = [l for l in tsv_string.strip().split('\n') if l.strip()]
        records = [DataRecord.from_tsv_line(line, has_result) for line in lines]
        return cls(records=records)
    
    def __len__(self) -> int:
        return len(self.records)


@dataclass
class DataStageResult:
    """
    Result container for data stage operations.
    
    Attributes:
        success: Whether the operation completed successfully.
        records: List of processed DataRecord objects.
        total_input_records: Total number of input records.
        total_output_records: Total number of output records.
        batches_processed: Number of batches processed.
        total_tokens_used: Estimated total tokens consumed.
        errors: List of error messages if any.
        execution_time: Total execution time in seconds.
        operation_type: Type of operation executed.
    """
    success: bool
    records: List[DataRecord]
    total_input_records: int = 0
    total_output_records: int = 0
    batches_processed: int = 0
    total_tokens_used: int = 0
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    operation_type: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "total_input_records": self.total_input_records,
            "total_output_records": self.total_output_records,
            "batches_processed": self.batches_processed,
            "total_tokens_used": self.total_tokens_used,
            "errors": self.errors,
            "execution_time": self.execution_time,
            "operation_type": self.operation_type,
            "records": [
                {"key": r.key, "value": r.value, "result": r.result}
                for r in self.records
            ]
        }
    
    def get_results_as_dict(self) -> Dict[Union[str, int], Any]:
        """Get results as a dictionary keyed by record key."""
        return {r.key: r.result for r in self.records}
    
    def to_tsv(self) -> str:
        """Export results as TSV string."""
        return '\n'.join(r.to_tsv_output() for r in self.records)


class DataStageOperation(ABC):
    """
    Abstract base class for data stage operations.
    
    Subclasses must implement:
        - operation_type: The type identifier for the operation
        - build_prompt(): Construct the AI prompt for processing
        - parse_response(): Parse the AI response into results
    
    The base class handles batching, throttling, and execution.
    """
    
    def __init__(
        self,
        prompt_definition: Optional[Dict[str, Any]] = None,
        data_format: DataFormat = DataFormat.TSV
    ):
        """
        Initialize the operation.
        
        Args:
            prompt_definition: Dictionary with prompts configuration.
            data_format: Data format for I/O (default: TSV).
        """
        self.prompt_definition = prompt_definition
        self.data_format = data_format
    
    @property
    @abstractmethod
    def operation_type(self) -> OperationType:
        """Return the operation type identifier."""
        pass
    
    @abstractmethod
    def build_prompt(
        self,
        batch: DataBatch,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Build the AI prompt messages for processing a batch.
        
        Args:
            batch: DataBatch to process.
            context: Optional context for the operation (e.g., lookup list).
        
        Returns:
            List of message dictionaries for the AI.
        """
        pass
    
    @abstractmethod
    def parse_response(
        self,
        response: str,
        batch: DataBatch
    ) -> List[DataRecord]:
        """
        Parse the AI response and update records with results.
        
        Args:
            response: Raw response from the AI.
            batch: Original batch that was processed.
        
        Returns:
            List of DataRecord with results populated.
        """
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token)."""
        return len(text) // 4 + 1


class DataStageExecutor:
    """
    Main executor class for data stage operations.
    
    Handles batching, throttling, and AI execution.
    Uses composition with TaskExecutor to reuse client management infrastructure.
    
    Attributes:
        _task_executor: Internal TaskExecutor for AI client management.
        context_reserve_ratio: Ratio of context to reserve for response.
        max_batch_chars: Maximum characters per batch.
    """
    
    def __init__(
        self,
        provider: Union[str, AIProvider] = AIProvider.GEMINI,
        model_name: str = "gemini-3-flash-preview",
        api_key: Optional[str] = None,
        temperature: Union[str, float] = "data_cleaning",
        max_tokens: int = 4096,
        context_reserve_ratio: float = 0.3
    ):
        """
        Initialize the DataStageExecutor.
        
        Args:
            provider: AI provider to use.
            model_name: Model name.
            api_key: API key (auto-loaded if not provided).
            temperature: Preset string or float. Presets: 'coding', 'math', 'data_cleaning',
                'data_analysis', 'translation', 'conversation', 'creative', 'poetry'.
            max_tokens: Maximum output tokens.
            context_reserve_ratio: Ratio reserved for output (0.3 = 30%).
        """
        # Normalize provider
        if isinstance(provider, str):
            provider = AIProvider(provider.lower())
        
        # Resolve temperature preset to float value
        if isinstance(temperature, str):
            final_temperature = connector_genai.GenerativeAIService.get_recommended_temperature(temperature)
        else:
            final_temperature = float(temperature)
        
        # Create a temporary config file or use TaskExecutor directly
        # TaskExecutor handles API key loading internally
        self._task_executor = TaskExecutor(api_key=api_key)
        
        # Override config with our settings
        self._task_executor.config = AIConfiguration(
            provider=provider,
            model_name=model_name,
            api_key=api_key or self._task_executor.config.api_key,
            temperature=final_temperature,
            max_tokens=max_tokens
        )
        
        # Recreate client with updated config
        self._task_executor.client = self._task_executor._create_client()
        
        self.context_reserve_ratio = context_reserve_ratio
        self.max_batch_chars = self._calculate_max_batch_chars()
    
    @property
    def config(self) -> AIConfiguration:
        """Access the AI configuration from TaskExecutor."""
        return self._task_executor.config
    
    @property
    def client(self) -> BaseAIClient:
        """Access the AI client from TaskExecutor."""
        return self._task_executor.client
    
    def _get_service_class(self) -> Optional[type]:
        """Get the service class for current provider (delegates to TaskExecutor pattern)."""
        return self._task_executor._get_active_service_class(self.config.provider)
    
    def _calculate_max_batch_chars(self) -> int:
        """Calculate maximum characters per batch based on context window."""
        svc_class = self._get_service_class()
        if svc_class:
            max_chars = svc_class.get_max_input_chars(self.config.model_name)
            available = int(max_chars * (1 - self.context_reserve_ratio) * 0.8)
            return max(available, 1000)
        return 10000
    
    def _create_batches(
        self,
        records: List[DataRecord],
        operation: DataStageOperation,
        context: Optional[Dict[str, Any]] = None
    ) -> List[DataBatch]:
        """Split records into batches that fit within context limits."""
        prompts = operation.prompt_definition.get("prompts", {}) if operation.prompt_definition else {}
        system_len = len(prompts.get("system", ""))
        user_template_len = len(prompts.get("user", ""))
        
        # Context overhead
        context_overhead = 0
        if context:
            if 'reference_list' in context:
                ref_list = context['reference_list']
                if isinstance(ref_list, list):
                    context_overhead = sum(len(str(r)) + 1 for r in ref_list)
                else:
                    context_overhead = len(str(ref_list))
        
        prompt_overhead = system_len + user_template_len + context_overhead + 200
        available_chars = self.max_batch_chars - prompt_overhead
        
        if available_chars < 100:
            logger.warning("Limited space for data. Consider smaller context.")
            available_chars = 1000
        
        batches = []
        current_batch_records = []
        current_batch_size = 0
        
        for record in records:
            record_str = record.to_tsv_input()
            record_len = len(record_str) + 1
            
            if current_batch_size + record_len > available_chars and current_batch_records:
                batches.append(DataBatch(
                    records=current_batch_records,
                    estimated_tokens=current_batch_size // 4
                ))
                current_batch_records = []
                current_batch_size = 0
            
            if record_len > available_chars:
                logger.warning(f"Record {record.key} too large ({record_len} chars).")
            
            current_batch_records.append(record)
            current_batch_size += record_len
        
        if current_batch_records:
            batches.append(DataBatch(
                records=current_batch_records,
                estimated_tokens=current_batch_size // 4
            ))
        
        total_batches = len(batches)
        for i, batch in enumerate(batches):
            batch.batch_index = i
            batch.total_batches = total_batches
        
        return batches
    
    @event_log(message="Data Stage Execution")
    def execute(
        self,
        operation: DataStageOperation,
        data: Union[List[DataRecord], List[Dict[str, Any]], List[Tuple[Any, Any]], List[str]],
        context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DataStageResult:
        """
        Execute a data stage operation on the provided data.
        
        Args:
            operation: The DataStageOperation to execute.
            data: Input data as DataRecords, dicts, tuples, or strings.
            context: Additional context for the operation.
            progress_callback: Optional callback(current_batch, total_batches).
        
        Returns:
            DataStageResult with processed records.
        """
        start_time = time.time()
        
        records = self._normalize_input(data)
        
        if not records:
            return DataStageResult(
                success=True,
                records=[],
                total_input_records=0,
                operation_type=operation.operation_type.value
            )
        
        batches = self._create_batches(records, operation, context)
        
        logger.info(f"Processing {len(records)} records in {len(batches)} batches")
        
        all_results = []
        errors = []
        total_tokens = 0
        
        for batch in batches:
            if progress_callback:
                progress_callback(batch.batch_index + 1, batch.total_batches)
            
            logger.info(f"Processing batch {batch.batch_index + 1}/{batch.total_batches} ({len(batch)} records)")
            
            try:
                messages = operation.build_prompt(batch, context)
                response = self.client.generate_content(messages)
                batch_results = operation.parse_response(response, batch)
                all_results.extend(batch_results)
                
                prompt_len = sum(len(m.get("content", "")) for m in messages)
                total_tokens += (prompt_len + len(response)) // 4
                
            except Exception as e:
                error_msg = f"Batch {batch.batch_index + 1} failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                for record in batch.records:
                    record.result = {"error": str(e)}
                    all_results.append(record)
        
        execution_time = time.time() - start_time
        
        return DataStageResult(
            success=len(errors) == 0,
            records=all_results,
            total_input_records=len(records),
            total_output_records=len(all_results),
            batches_processed=len(batches),
            total_tokens_used=total_tokens,
            errors=errors,
            execution_time=execution_time,
            operation_type=operation.operation_type.value
        )
    
    def _normalize_input(
        self,
        data: Union[List[DataRecord], List[Dict[str, Any]], List[Tuple[Any, Any]], List[str]]
    ) -> List[DataRecord]:
        """Normalize various input formats to List[DataRecord]."""
        if not data:
            return []
        
        sample = data[0]
        
        if isinstance(sample, DataRecord):
            return data
        
        if isinstance(sample, dict):
            return [
                DataRecord(key=d.get('key', i), value=d.get('value', d))
                for i, d in enumerate(data)
            ]
        
        if isinstance(sample, tuple) and len(sample) >= 2:
            return [DataRecord(key=t[0], value=t[1]) for t in data]
        
        return [DataRecord(key=str(i), value=v) for i, v in enumerate(data)]
    
    def semantic_lookup(
        self,
        data: Union[List[str], List[Tuple[Any, Any]], List[DataRecord]],
        reference_list: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DataStageResult:
        """
        Convenience method for semantic lookup operation.
        
        Args:
            data: Input values to match.
            reference_list: Reference list to match against.
            progress_callback: Optional progress callback.
        
        Returns:
            DataStageResult with match results.
        """
        from .operations import SemanticLookupOperation
        operation = SemanticLookupOperation()
        context = {"reference_list": reference_list}
        return self.execute(operation, data, context, progress_callback)
    
    def spell_correction(
        self,
        data: Union[List[str], List[Tuple[Any, Any]], List[DataRecord]],
        language: str = "auto",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DataStageResult:
        """
        Convenience method for spell correction operation.
        
        Args:
            data: Input texts to correct.
            language: Language hint (default: auto-detect).
            progress_callback: Optional progress callback.
        
        Returns:
            DataStageResult with corrected texts.
        """
        from .operations import SpellCorrectionOperation
        operation = SpellCorrectionOperation(language=language)
        context = {"language": language}
        return self.execute(operation, data, context, progress_callback)


def create_data_stage(
    provider: str = "gemini",
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None
) -> DataStageExecutor:
    """
    Factory function to create a DataStageExecutor with common defaults.
    
    Args:
        provider: AI provider name.
        model: Model name.
        api_key: Optional API key.
    
    Returns:
        Configured DataStageExecutor.
    """
    return DataStageExecutor(
        provider=provider,
        model_name=model,
        api_key=api_key,
        temperature=0.1
    )
