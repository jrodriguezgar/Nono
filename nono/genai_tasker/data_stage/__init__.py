"""
Data Stage Operations - Extension for GenAI Tasker

This module extends genai_tasker with specialized data processing operations
using token-efficient TSV format and intelligent throttling.

Key Features:
    - TSV (Tab-Separated Values) format: 60-70% token reduction vs JSON
    - Smart throttling based on context window limits
    - Extensible operation base class
    - Built-in operations: SemanticLookup, SpellCorrection

Usage:
    from nono.genai_tasker.data_stage import DataStageExecutor
    
    executor = DataStageExecutor(provider="gemini")
    result = executor.semantic_lookup(data, reference_list)

Author: DatamanEdge
License: MIT
Version: 1.0.0
"""

from .core import (
    # Core Classes
    DataStageExecutor,
    DataStageOperation,
    DataRecord,
    DataBatch,
    DataStageResult,
    # Enums
    DataFormat,
    OperationType,
    # Factory
    create_data_stage,
)

from .operations import (
    # Built-in Operations
    SemanticLookupOperation,
    SpellCorrectionOperation,
)

__all__ = [
    # Core
    "DataStageExecutor",
    "DataStageOperation",
    "DataRecord",
    "DataBatch",
    "DataStageResult",
    "DataFormat",
    "OperationType",
    "create_data_stage",
    # Operations
    "SemanticLookupOperation",
    "SpellCorrectionOperation",
]
