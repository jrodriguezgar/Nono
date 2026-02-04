"""
Data Stage Operations - Built-in operations for data processing.

This module contains the built-in operations for genai_tasker data stage:
- SemanticLookupOperation: Match values against a reference list
- SpellCorrectionOperation: Correct spelling errors in text

Author: DatamanEdge
License: MIT
"""

from typing import Any, Dict, List, Optional

from .core import (
    DataStageOperation,
    DataBatch,
    DataRecord,
    DataFormat,
    OperationType,
)


class SemanticLookupOperation(DataStageOperation):
    """
    Operation for semantic lookup/matching in a reference list.
    
    Matches input values against a reference list using semantic similarity.
    Returns match type (exact, fuzzy, no_match) and matched value if found.
    
    Example:
        Input: ["Sevila", "Madriz", "Barzelona"]
        Reference: ["Madrid", "Barcelona", "Sevilla", "Valencia"]
        Output: [
            ("Sevila", "Sevilla", "fuzzy"),
            ("Madriz", "Madrid", "fuzzy"),
            ("Barzelona", "Barcelona", "fuzzy")
        ]
    """
    
    def __init__(
        self,
        prompt_definition: Optional[Dict[str, Any]] = None,
        data_format: DataFormat = DataFormat.TSV
    ):
        # Default prompt if none provided
        default_prompt = {
            "prompts": {
                "system": """You are a semantic matching expert. Match input values to the closest entry in the reference list.

RULES:
- Return ONLY TSV format: KEY<TAB>ORIGINAL<TAB>MATCHED_VALUE<TAB>MATCH_TYPE
- MATCH_TYPE must be: exact|fuzzy|none
- For "none" matches, leave MATCHED_VALUE empty
- Consider spelling errors, accents, abbreviations, synonyms
- Preserve the exact KEY from input
- One line per input, same order as input
- No headers, no explanations, just data lines""",
                "user": """Reference list (one per line):
{reference_list}

Input data (TSV: KEY<TAB>VALUE):
{input_data}

Output TSV (KEY<TAB>ORIGINAL<TAB>MATCHED<TAB>TYPE):"""
            }
        }
        super().__init__(prompt_definition or default_prompt, data_format)
    
    @property
    def operation_type(self) -> OperationType:
        return OperationType.SEMANTIC_LOOKUP
    
    def build_prompt(
        self,
        batch: DataBatch,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Build prompt for semantic lookup."""
        if not context or 'reference_list' not in context:
            raise ValueError("SemanticLookupOperation requires 'reference_list' in context")
        
        prompts = self.prompt_definition.get("prompts", {})
        system_prompt = prompts.get("system", "")
        user_template = prompts.get("user", "")
        
        # Format reference list (one per line for minimal tokens)
        reference_list = context['reference_list']
        if isinstance(reference_list, list):
            ref_str = '\n'.join(str(r) for r in reference_list)
        else:
            ref_str = str(reference_list)
        
        # Format input data as TSV
        input_data = batch.to_tsv(include_result=False)
        
        user_content = user_template.replace("{reference_list}", ref_str)
        user_content = user_content.replace("{input_data}", input_data)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def parse_response(
        self,
        response: str,
        batch: DataBatch
    ) -> List[DataRecord]:
        """Parse semantic lookup response."""
        results = []
        response_lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        
        # Build key -> record map
        key_map = {str(r.key): r for r in batch.records}
        
        for line in response_lines:
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            key = parts[0]
            matched = parts[2] if len(parts) > 2 else ""
            match_type = parts[3] if len(parts) > 3 else "none"
            
            if key in key_map:
                record = key_map[key]
                record.result = {
                    "matched_value": matched if matched else None,
                    "match_type": match_type.lower()
                }
                results.append(record)
        
        # Add missing records with error
        for key, record in key_map.items():
            if record not in results:
                record.result = {"matched_value": None, "match_type": "error"}
                results.append(record)
        
        return results


class SpellCorrectionOperation(DataStageOperation):
    """
    Operation for spelling correction on text values.
    
    Corrects spelling errors in input values and returns the corrected text
    along with a flag indicating if corrections were made.
    
    Example:
        Input: ["teh quick brwon fox", "hello world"]
        Output: [
            ("teh quick brwon fox", "the quick brown fox", True),
            ("hello world", "hello world", False)
        ]
    """
    
    def __init__(
        self,
        prompt_definition: Optional[Dict[str, Any]] = None,
        data_format: DataFormat = DataFormat.TSV,
        language: str = "auto"
    ):
        default_prompt = {
            "prompts": {
                "system": """You are a spelling correction expert. Correct spelling errors in input text.

RULES:
- Return ONLY TSV format: KEY<TAB>CORRECTED_TEXT<TAB>WAS_CORRECTED
- WAS_CORRECTED must be: 1 (yes) or 0 (no)
- Preserve punctuation and capitalization style
- Fix typos, missing letters, transposed letters
- Do NOT change proper nouns unless clearly misspelled
- Preserve the exact KEY from input
- One line per input, same order as input
- No headers, no explanations, just data lines""",
                "user": """Language hint: {language}

Input data (TSV: KEY<TAB>TEXT):
{input_data}

Output TSV (KEY<TAB>CORRECTED<TAB>CHANGED):"""
            }
        }
        super().__init__(prompt_definition or default_prompt, data_format)
        self.language = language
    
    @property
    def operation_type(self) -> OperationType:
        return OperationType.SPELL_CORRECTION
    
    def build_prompt(
        self,
        batch: DataBatch,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Build prompt for spell correction."""
        prompts = self.prompt_definition.get("prompts", {})
        system_prompt = prompts.get("system", "")
        user_template = prompts.get("user", "")
        
        language = self.language
        if context and 'language' in context:
            language = context['language']
        
        input_data = batch.to_tsv(include_result=False)
        
        user_content = user_template.replace("{language}", language)
        user_content = user_content.replace("{input_data}", input_data)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def parse_response(
        self,
        response: str,
        batch: DataBatch
    ) -> List[DataRecord]:
        """Parse spell correction response."""
        results = []
        response_lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        
        key_map = {str(r.key): r for r in batch.records}
        
        for line in response_lines:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            key = parts[0]
            corrected = parts[1] if len(parts) > 1 else ""
            was_corrected = parts[2] if len(parts) > 2 else "0"
            
            if key in key_map:
                record = key_map[key]
                record.result = {
                    "corrected_text": corrected,
                    "was_corrected": was_corrected in ("1", "true", "yes", "True", "Yes")
                }
                results.append(record)
        
        # Add missing records unchanged
        for key, record in key_map.items():
            if record not in results:
                record.result = {
                    "corrected_text": str(record.value),
                    "was_corrected": False
                }
                results.append(record)
        
        return results
