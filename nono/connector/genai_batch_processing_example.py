# -*- coding: utf-8 -*-
"""
Batch Processing Example - Gemini and OpenAI Batch APIs

This example demonstrates how to use GeminiBatchService and OpenAIBatchService
for processing large volumes of requests asynchronously at 50% cost.

Author: DatamanEdge
License: MIT
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connector.batch_processing import GeminiBatchService, OpenAIBatchService


def example_inline_batch(api_key: str) -> None:
    """
    Example 1: Inline batch processing for small sets of requests.
    Suitable for batches under 20 MB total size.
    """
    print("=" * 60)
    print("Example 1: Inline Batch Processing")
    print("=" * 60)
    
    batch_service = GeminiBatchService(api_key=api_key)
    
    # Define prompts to process
    prompts = [
        "¿Cuál es la capital de España?",
        "Explica la fotosíntesis en una oración.",
        "¿Cuántos planetas hay en el sistema solar?",
        "¿Quién escribió Don Quijote?",
        "¿Cuál es la velocidad de la luz?"
    ]
    
    # Build requests using the helper method
    requests = GeminiBatchService.build_requests_from_prompts(
        prompts=prompts,
        system_instruction="Responde de forma concisa en español, máximo 2 oraciones.",
        temperature=0.5,
        max_tokens=100
    )
    
    print(f"Creating batch job with {len(requests)} requests...")
    
    # Create the batch job
    job = batch_service.create_inline_batch(
        requests=requests,
        model_name="gemini-3-flash-preview",
        display_name="ejemplo-inline-batch"
    )
    
    print(f"✓ Job created: {job.name}")
    print(f"  Initial state: {job.state}")
    
    # Wait for completion with progress callback
    def progress_callback(job, elapsed_seconds):
        print(f"  [{elapsed_seconds:.0f}s] State: {job.state}")
    
    print("\nWaiting for completion...")
    completed_job = batch_service.wait_for_completion(
        job_name=job.name,
        poll_interval=10,  # Check every 10 seconds
        timeout=600,       # Max 10 minutes
        callback=progress_callback
    )
    
    # Process results
    if completed_job.state == GeminiBatchService.STATE_SUCCEEDED:
        print("\n✓ Batch completed successfully!")
        results = batch_service.get_inline_results(job.name)
        
        print("\nResults:")
        print("-" * 40)
        for i, (prompt, result) in enumerate(zip(prompts, results), 1):
            print(f"\n[{i}] Q: {prompt}")
            print(f"    A: {result}")
    else:
        print(f"\n✗ Batch failed with state: {completed_job.state}")


def example_file_batch(api_key: str) -> None:
    """
    Example 2: File-based batch processing for large sets of requests.
    Recommended for batches over 100 requests or up to 2 GB.
    """
    print("\n" + "=" * 60)
    print("Example 2: File-Based Batch Processing")
    print("=" * 60)
    
    batch_service = GeminiBatchService(api_key=api_key)
    
    # Create a larger set of requests
    prompts = [
        "Describe qué es Python en una oración.",
        "Describe qué es JavaScript en una oración.",
        "Describe qué es Rust en una oración.",
        "Describe qué es Go en una oración.",
        "Describe qué es Java en una oración.",
        "Describe qué es C++ en una oración.",
        "Describe qué es Ruby en una oración.",
        "Describe qué es Swift en una oración.",
        "Describe qué es Kotlin en una oración.",
        "Describe qué es TypeScript en una oración.",
    ]
    
    requests = GeminiBatchService.build_requests_from_prompts(
        prompts=prompts,
        system_instruction="Eres un experto en programación. Responde en español.",
        temperature=0.7
    )
    
    # Create JSONL file
    jsonl_path = "batch_requests.jsonl"
    keys = [f"lang-{i+1}" for i in range(len(prompts))]
    
    print(f"Creating JSONL file with {len(requests)} requests...")
    batch_service.create_jsonl_file(
        requests=requests,
        output_path=jsonl_path,
        keys=keys
    )
    print(f"✓ JSONL file created: {jsonl_path}")
    
    # Create batch job from file
    print("Creating batch job from file...")
    job = batch_service.create_file_batch(
        file_path=jsonl_path,
        model_name="gemini-3-flash-preview",
        display_name="ejemplo-file-batch"
    )
    
    print(f"✓ Job created: {job.name}")
    print(f"  Initial state: {job.state}")
    
    # Wait for completion
    print("\nWaiting for completion...")
    completed_job = batch_service.wait_for_completion(
        job_name=job.name,
        poll_interval=15,
        timeout=600,
        callback=lambda j, t: print(f"  [{t:.0f}s] State: {j.state}")
    )
    
    # Process results
    if completed_job.state == GeminiBatchService.STATE_SUCCEEDED:
        print("\n✓ Batch completed successfully!")
        
        # Save results to file
        output_path = "batch_results.jsonl"
        batch_service.get_file_results(job.name, output_path)
        print(f"Results saved to: {output_path}")
        
        # Or get as list
        results = batch_service.get_file_results(job.name)
        print(f"\nProcessed {len(results)} responses")
        for result in results[:3]:  # Show first 3
            print(f"  Key: {result.get('key')}")
    else:
        print(f"\n✗ Batch failed with state: {completed_job.state}")
    
    # Cleanup
    import os
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)
        print(f"\nCleaned up {jsonl_path}")


def example_build_requests() -> None:
    """
    Example 3: Building requests with different configurations.
    Shows various ways to construct batch requests.
    """
    print("\n" + "=" * 60)
    print("Example 3: Building Custom Requests")
    print("=" * 60)
    
    # Method 1: Single request with all options
    request1 = GeminiBatchService.build_request(
        prompt="Explica la teoría de la relatividad.",
        system_instruction="Eres un profesor de física.",
        temperature=0.7,
        max_tokens=200,
        top_p=0.95,
        top_k=40,
        stop_sequences=[".", "\n\n"]
    )
    print("\n1. Request with full configuration:")
    print(f"   Keys: {list(request1.keys())}")
    
    # Method 2: Simple request
    request2 = GeminiBatchService.build_request(
        prompt="¿Qué es la inteligencia artificial?",
    )
    print("\n2. Simple request (minimal):")
    print(f"   Keys: {list(request2.keys())}")
    
    # Method 3: Batch from prompts
    prompts = ["Pregunta 1", "Pregunta 2", "Pregunta 3"]
    requests = GeminiBatchService.build_requests_from_prompts(
        prompts=prompts,
        system_instruction="Responde brevemente.",
        temperature=0.5
    )
    print(f"\n3. Batch from prompts: {len(requests)} requests")
    
    # Method 4: Request with Google Search tool
    search_request = GeminiBatchService.build_request_with_search(
        prompt="¿Quién ganó la Eurocopa 2024?",
        temperature=0.3
    )
    print("\n4. Request with Google Search:")
    print(f"   Has tools: {'config' in search_request and 'tools' in search_request.get('config', {})}")
    
    # Method 5: Structured output request
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "ingredients": {"type": "array", "items": {"type": "string"}}
            }
        }
    }
    structured_request = GeminiBatchService.build_structured_request(
        prompt="Lista 3 recetas de galletas con ingredientes",
        response_schema=schema,
        temperature=0.5
    )
    print("\n5. Structured output request:")
    print(f"   Response MIME type: {structured_request.get('config', {}).get('response_mime_type')}")
    print(f"   Has response schema: {'response_schema' in structured_request.get('config', {})}")
    
    # Method 6: Image generation request
    image_request = GeminiBatchService.build_image_request(
        prompt="Un gato tocando el piano en una habitación acogedora",
        system_instruction="Genera imágenes artísticas y coloridas."
    )
    print("\n6. Image generation request:")
    print(f"   Response modalities: {image_request.get('generation_config', {}).get('responseModalities')}")
    
    # Method 7: Manual request with config (new format)
    manual_request = {
        "contents": [
            {
                "parts": [{"text": "¿Cuál es tu color favorito?"}],
                "role": "user"
            }
        ],
        "config": {
            "system_instruction": {"parts": [{"text": "Eres un asistente creativo."}]},
            "tools": [{"google_search": {}}]
        },
        "generation_config": {
            "temperature": 0.9,
            "max_output_tokens": 50
        }
    }
    print("\n7. Manual request with config:")
    print(f"   Contents: {len(manual_request['contents'])} message(s)")
    print(f"   Has config: True")
    print(f"   Has tools in config: True")


def example_job_management(api_key: str) -> None:
    """
    Example 4: Managing batch jobs (list, check status, cancel, delete).
    """
    print("\n" + "=" * 60)
    print("Example 4: Job Management")
    print("=" * 60)
    
    batch_service = GeminiBatchService(api_key=api_key)
    
    # List existing jobs
    print("\nListing existing jobs...")
    jobs = batch_service.list_jobs(page_size=10)
    
    if jobs:
        print(f"Found {len(jobs)} job(s):")
        for job in jobs[:5]:  # Show first 5
            print(f"  - {job.name}: {job.state}")
    else:
        print("  No jobs found")
    
    # Example: Create a job and then check/cancel it
    print("\nCreating a test job...")
    requests = GeminiBatchService.build_requests_from_prompts(
        prompts=["Test question 1", "Test question 2"],
        temperature=0.5
    )
    
    job = batch_service.create_inline_batch(
        requests=requests,
        display_name="test-management-job"
    )
    print(f"✓ Created job: {job.name}")
    
    # Check status
    status = batch_service.get_job_status(job.name)
    print(f"  Current status: {status}")
    
    # Get full job details
    job_details = batch_service.get_job(job.name)
    print(f"  Display name: {job_details.display_name if hasattr(job_details, 'display_name') else 'N/A'}")
    
    # Cancel the job (optional - uncomment to test)
    # print("\nCancelling job...")
    # cancelled = batch_service.cancel_job(job.name)
    # print(f"  New status: {cancelled.state}")
    
    # Delete the job (optional - uncomment to test)
    # print("\nDeleting job...")
    # batch_service.delete_job(job.name)
    # print("  Job deleted")


# ============================================================================
# OpenAI Batch API Examples
# ============================================================================

def example_openai_chat_batch(api_key: str) -> None:
    """
    Example 5: OpenAI Chat Completions Batch Processing.
    """
    print("\n" + "=" * 60)
    print("Example 5: OpenAI Chat Completions Batch")
    print("=" * 60)
    
    batch_service = OpenAIBatchService(api_key=api_key)
    
    # Define prompts to process
    prompts = [
        "What is the capital of Spain?",
        "Explain photosynthesis in one sentence.",
        "How many planets are in the solar system?",
        "Who wrote Don Quixote?",
        "What is the speed of light?"
    ]
    
    # Build requests using the helper method
    requests = OpenAIBatchService.build_requests_from_prompts(
        prompts=prompts,
        model="gpt-4o-mini",
        system_prompt="Answer concisely in English, maximum 2 sentences.",
        temperature=0.5,
        max_tokens=100,
        custom_id_prefix="question"
    )
    
    print(f"Creating batch with {len(requests)} requests...")
    
    # Create JSONL file
    jsonl_path = "openai_batch_input.jsonl"
    batch_service.create_jsonl_file(requests, jsonl_path)
    print(f"✓ JSONL file created: {jsonl_path}")
    
    # Create the batch job
    batch = batch_service.create_batch(
        file_path=jsonl_path,
        endpoint="/v1/chat/completions",
        metadata={"description": "example-chat-batch"}
    )
    
    print(f"✓ Batch created: {batch.id}")
    print(f"  Initial status: {batch.status}")
    
    # Wait for completion with progress callback
    def progress_callback(batch, elapsed_seconds):
        counts = batch.request_counts
        print(f"  [{elapsed_seconds:.0f}s] Status: {batch.status} - {counts.completed}/{counts.total}")
    
    print("\nWaiting for completion...")
    completed = batch_service.wait_for_completion(
        batch_id=batch.id,
        poll_interval=10,  # Check every 10 seconds
        timeout=3600,      # Max 1 hour
        callback=progress_callback
    )
    
    # Process results
    if completed.status == OpenAIBatchService.STATE_COMPLETED:
        print("\n✓ Batch completed successfully!")
        results = batch_service.get_results(batch.id)
        
        print("\nResults:")
        print("-" * 40)
        for r in results:
            custom_id = r["custom_id"]
            content = r["response"]["body"]["choices"][0]["message"]["content"]
            print(f"\n[{custom_id}]: {content}")
        
        # Check for errors
        errors = batch_service.get_errors(batch.id)
        if errors:
            print(f"\n⚠️  {len(errors)} request(s) failed:")
            for e in errors:
                print(f"  {e['custom_id']}: {e['error']}")
    else:
        print(f"\n✗ Batch ended with status: {completed.status}")
    
    # Cleanup
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)
        print(f"\nCleaned up {jsonl_path}")


def example_openai_embeddings_batch(api_key: str) -> None:
    """
    Example 6: OpenAI Embeddings Batch Processing.
    """
    print("\n" + "=" * 60)
    print("Example 6: OpenAI Embeddings Batch")
    print("=" * 60)
    
    batch_service = OpenAIBatchService(api_key=api_key)
    
    # Texts to embed
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming industries.",
        "Python is a popular programming language.",
        "Artificial intelligence enables new possibilities.",
        "Data science combines statistics and programming."
    ]
    
    # Build embedding requests
    requests = [
        OpenAIBatchService.build_embedding_request(
            input_text=text,
            model="text-embedding-3-small",
            custom_id=f"embed-{i+1}"
        )
        for i, text in enumerate(texts)
    ]
    
    print(f"Creating embedding batch with {len(requests)} requests...")
    
    # Create JSONL and batch
    jsonl_path = "openai_embedding_batch.jsonl"
    batch_service.create_jsonl_file(requests, jsonl_path)
    
    batch = batch_service.create_batch(
        file_path=jsonl_path,
        endpoint="/v1/embeddings",
        metadata={"description": "example-embedding-batch"}
    )
    
    print(f"✓ Batch created: {batch.id}")
    
    # Wait and get results
    completed = batch_service.wait_for_completion(
        batch.id,
        poll_interval=10,
        callback=lambda b, t: print(f"  [{t:.0f}s] {b.status}")
    )
    
    if completed.status == OpenAIBatchService.STATE_COMPLETED:
        print("\n✓ Batch completed!")
        results = batch_service.get_results(batch.id)
        
        for r in results:
            embedding = r["response"]["body"]["data"][0]["embedding"]
            print(f"  {r['custom_id']}: {len(embedding)} dimensions")
    
    # Cleanup
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)


def example_openai_build_requests() -> None:
    """
    Example 7: Building OpenAI batch requests with different configurations.
    Shows various ways to construct batch requests.
    """
    print("\n" + "=" * 60)
    print("Example 7: Building OpenAI Batch Requests")
    print("=" * 60)
    
    # Method 1: Chat completion request with all options
    request1 = OpenAIBatchService.build_request(
        prompt="Explain the theory of relativity.",
        model="gpt-4o-mini",
        custom_id="physics-question-1",
        system_prompt="You are a physics teacher.",
        temperature=0.7,
        max_tokens=200,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        response_format={"type": "text"}
    )
    print("\n1. Chat request with full configuration:")
    print(f"   Custom ID: {request1['custom_id']}")
    print(f"   Endpoint: {request1['url']}")
    print(f"   Model: {request1['body']['model']}")
    
    # Method 2: Simple chat request
    request2 = OpenAIBatchService.build_request(
        prompt="What is AI?",
        model="gpt-4o-mini"
    )
    print("\n2. Simple chat request (minimal):")
    print(f"   Custom ID: {request2['custom_id']}")
    
    # Method 3: Batch from prompts
    prompts = ["Question 1?", "Question 2?", "Question 3?"]
    requests = OpenAIBatchService.build_requests_from_prompts(
        prompts=prompts,
        model="gpt-4o-mini",
        system_prompt="Answer briefly.",
        temperature=0.5,
        custom_id_prefix="q"
    )
    print(f"\n3. Batch from prompts: {len(requests)} requests")
    for r in requests:
        print(f"   - {r['custom_id']}")
    
    # Method 4: Embedding request
    embed_request = OpenAIBatchService.build_embedding_request(
        input_text="Hello world",
        model="text-embedding-3-small",
        custom_id="embed-hello",
        dimensions=512
    )
    print("\n4. Embedding request:")
    print(f"   Custom ID: {embed_request['custom_id']}")
    print(f"   Endpoint: {embed_request['url']}")
    
    # Method 5: Moderation request (text)
    mod_request = OpenAIBatchService.build_moderation_request(
        input_content="This is a test sentence.",
        custom_id="mod-test-1"
    )
    print("\n5. Moderation request (text):")
    print(f"   Custom ID: {mod_request['custom_id']}")
    print(f"   Endpoint: {mod_request['url']}")
    
    # Method 6: Moderation request (multimodal)
    mod_mm_request = OpenAIBatchService.build_moderation_request(
        input_content=[
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ],
        custom_id="mod-multimodal-1"
    )
    print("\n6. Moderation request (multimodal):")
    print(f"   Custom ID: {mod_mm_request['custom_id']}")
    print(f"   Input types: text, image_url")


def example_openai_batch_management(api_key: str) -> None:
    """
    Example 8: Managing OpenAI batches (list, check status, cancel).
    """
    print("\n" + "=" * 60)
    print("Example 8: OpenAI Batch Management")
    print("=" * 60)
    
    batch_service = OpenAIBatchService(api_key=api_key)
    
    # List existing batches
    print("\nListing existing batches...")
    batches = batch_service.list_batches(limit=10)
    
    if batches:
        print(f"Found {len(batches)} batch(es):")
        for batch in batches[:5]:  # Show first 5
            counts = batch.request_counts
            print(f"  - {batch.id}: {batch.status} ({counts.completed}/{counts.total})")
    else:
        print("  No batches found")


def main():
    """Main function to run all examples."""
    print("=" * 60)
    print("Gemini and OpenAI Batch Processing Examples")
    print("=" * 60)
    
    # Get API key from environment or config
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        # Try to read from config.toml
        try:
            import tomllib
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config.toml"
            )
            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    config = tomllib.load(f)
                    api_key = config.get("gemini", {}).get("api_key")
        except Exception:
            pass
    
    if not api_key:
        print("\n⚠️  No API key found!")
        print("Set GOOGLE_API_KEY environment variable or configure config.toml")
        print("\nRunning examples that don't require API key...")
        example_build_requests()
        example_openai_build_requests()
        return
    
    print(f"\n✓ API key found (ending in ...{api_key[-4:]})")
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print(f"✓ OpenAI API key found (ending in ...{openai_api_key[-4:]})")
    
    # Run examples (uncomment the ones you want to test)
    try:
        # ============================================
        # Gemini Examples
        # ============================================
        
        # Example 1: Inline batch (small sets)
        # example_inline_batch(api_key)
        
        # Example 2: File-based batch (large sets)
        # example_file_batch(api_key)
        
        # Example 3: Building requests (no API call needed)
        example_build_requests()
        
        # Example 4: Job management
        # example_job_management(api_key)
        
        # ============================================
        # OpenAI Examples
        # ============================================
        
        # Example 5: OpenAI Chat Completions Batch
        # if openai_api_key:
        #     example_openai_chat_batch(openai_api_key)
        
        # Example 6: OpenAI Embeddings Batch
        # if openai_api_key:
        #     example_openai_embeddings_batch(openai_api_key)
        
        # Example 7: Building OpenAI requests (no API call needed)
        example_openai_build_requests()
        
        # Example 8: OpenAI Batch Management
        # if openai_api_key:
        #     example_openai_batch_management(openai_api_key)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
