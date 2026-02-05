# Batch Processing with Gemini and OpenAI Batch APIs

> Process large volumes of requests asynchronously at **50% of the standard cost** using the Google Gemini and OpenAI Batch APIs.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Cost](https://img.shields.io/badge/cost-50%25%20savings-brightgreen)

## Table of Contents

- [Supported Providers](#supported-providers)
- [When to Use Batch Processing](#when-to-use-batch-processing)
- [Quick Start - Gemini](#quick-start---gemini)
- [Quick Start - OpenAI](#quick-start---openai)
- [Gemini Batch API](#gemini-batch-api)
  - [Inline Requests](#gemini-inline-requests)
  - [Google Search](#gemini-with-google-search)
  - [Structured Output](#gemini-structured-output)
  - [Image Generation](#gemini-image-generation)
  - [File-Based Requests](#gemini-file-based-requests)
- [OpenAI Batch API](#openai-batch-api)
  - [Chat Completions](#openai-chat-completions)
  - [Embeddings](#openai-embeddings)
  - [Moderations](#openai-moderations)
- [Monitoring and Results](#monitoring-and-results)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)

---

## Supported Providers

| Provider | Class | Max File Size | Features |
|:---------|:------|:--------------|:---------|
| **Google Gemini** | `GeminiBatchService` | 20 MB inline, 2 GB file | Inline & file-based, Google Search, Structured Output, Image Generation |
| **OpenAI** | `OpenAIBatchService` | 200 MB | Chat, Embeddings, Moderations |

Both providers offer:
- **50% cost savings** compared to synchronous APIs
- **Higher rate limits** (separate pool)
- **24-hour turnaround** (usually much faster)
- **48-hour expiration** for Gemini jobs

---

## When to Use Batch Processing

### ✅ Use Batch for:

- Running evaluations and benchmarks
- Classifying large datasets
- Embedding content repositories
- Bulk document translation
- Large-scale text classification
- Sentiment analysis at scale
- Content moderation in bulk
- **Bulk image generation** (Gemini)
- **Web search queries at scale** (Gemini with Google Search)
- **Structured data extraction** (Gemini with response schemas)
- Any task that doesn't require immediate response

### ❌ Do NOT use Batch for:

- Real-time interactive applications
- Chatbots requiring immediate response
- Critical features with strict SLAs
- Processing few items (< 10 requests)

---

## Quick Start - Gemini

```python
from nono.connector.genai_batch_processing import GeminiBatchService

# Initialize the service
batch_service = GeminiBatchService(api_key="your-api-key")

# Create simple requests
prompts = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "How many planets are in the solar system?"
]

# Build requests from prompts
requests = GeminiBatchService.build_requests_from_prompts(
    prompts,
    system_instruction="Answer concisely.",
    temperature=0.7
)

# Create inline batch job
job = batch_service.create_inline_batch(
    requests=requests,
    display_name="my-first-batch"
)

print(f"✓ Job created: {job.name}")

# Wait for completion
completed_job = batch_service.wait_for_completion(
    job.name,
    poll_interval=30,
    callback=lambda j, t: print(f"Status: {j.state} ({t:.0f}s)")
)

# Get results
if completed_job.state == GeminiBatchService.STATE_SUCCEEDED:
    results = batch_service.get_inline_results(job.name)
    for i, result in enumerate(results):
        print(f"Response {i+1}: {result}")
```

---

## Quick Start - OpenAI

```python
from nono.connector.genai_batch_processing import OpenAIBatchService

# Initialize the service
batch_service = OpenAIBatchService(api_key="your-api-key")

# Create simple requests
prompts = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "How many planets are in the solar system?"
]

# Build requests from prompts
requests = OpenAIBatchService.build_requests_from_prompts(
    prompts,
    model="gpt-4o-mini",
    system_prompt="Answer concisely.",
    temperature=0.7
)

# Create JSONL file and start batch
batch_service.create_jsonl_file(requests, "batch_input.jsonl")
batch = batch_service.create_batch(
    file_path="batch_input.jsonl",
    metadata={"description": "my-first-batch"}
)

print(f"✓ Batch created: {batch.id}")

# Wait for completion
completed = batch_service.wait_for_completion(
    batch.id,
    poll_interval=30,
    callback=lambda b, t: print(f"Status: {b.status} ({t:.0f}s)")
)

# Get results
if completed.status == OpenAIBatchService.STATE_COMPLETED:
    results = batch_service.get_results(batch.id)
    for r in results:
        content = r["response"]["body"]["choices"][0]["message"]["content"]
        print(f"{r['custom_id']}: {content}")
```
```

---

## Gemini Batch API

### Gemini Inline Requests

Suitable for smaller batches that keep total request size under **20 MB**.

```python
from nono.connector.genai_batch_processing import GeminiBatchService

batch_service = GeminiBatchService(api_key="your-api-key")

# Method 1: Build requests manually
inline_requests = [
    {
        "contents": [
            {
                "parts": [{"text": "Tell me a short joke."}],
                "role": "user"
            }
        ]
    },
    {
        "contents": [
            {
                "parts": [{"text": "Why is the sky blue?"}],
                "role": "user"
            }
        ],
        "generation_config": {
            "temperature": 0.5
        }
    }
]

# Method 2: Use the build_request helper
requests_via_helper = [
    GeminiBatchService.build_request(
        prompt="Explain gravity.",
        system_instruction="You are a physics teacher.",
        temperature=0.7,
        max_tokens=200
    ),
    GeminiBatchService.build_request(
        prompt="What is a black hole?",
        system_instruction="You are a physics teacher.",
        temperature=0.7
    )
]

# Create the job
job = batch_service.create_inline_batch(
    requests=inline_requests,
    model_name="gemini-3-flash-preview",
    display_name="science-batch"
)

print(f"Job created: {job.name}")
print(f"Initial state: {job.state}")
```

### Gemini with Google Search

Use the Google Search tool for queries that require up-to-date information:

```python
# Method 1: Using helper
request = GeminiBatchService.build_request_with_search(
    prompt="Who won Euro 2024?",
    temperature=0.3
)

# Method 2: Manual configuration
request_manual = GeminiBatchService.build_request(
    prompt="What are the latest AI developments?",
    tools=[{"google_search": {}}]
)

# Create batch with search-enabled requests
job = batch_service.create_inline_batch(
    requests=[request],
    display_name="search-batch"
)
```

### Gemini Structured Output

Get responses in structured JSON format with a defined schema:

```python
from pydantic import BaseModel

# Option 1: Using Pydantic models
class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]

request = GeminiBatchService.build_structured_request(
    prompt="List 3 popular cookie recipes with ingredients.",
    response_schema=list[Recipe],
    temperature=0.5
)

# Option 2: Using dict schema
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "recipe_name": {"type": "string"},
            "ingredients": {"type": "array", "items": {"type": "string"}}
        }
    }
}

request = GeminiBatchService.build_structured_request(
    prompt="List 3 popular cookie recipes with ingredients.",
    response_schema=schema
)

# Option 3: Manual configuration
request_manual = GeminiBatchService.build_request(
    prompt="List 3 recipes",
    response_mime_type="application/json",
    response_schema=schema
)

job = batch_service.create_inline_batch(
    requests=[request],
    display_name="structured-output-batch"
)
```

### Gemini Image Generation

Generate images in batch using models like `gemini-3-pro-image-preview`:

```python
# Build image generation request
request = GeminiBatchService.build_image_request(
    prompt="A cat playing piano in a cozy room",
    system_instruction="Generate artistic, colorful images."
)

# Create batch with image model
job = batch_service.create_inline_batch(
    requests=[request],
    model_name="gemini-3-pro-image-preview",
    display_name="image-generation-batch"
)

# For file-based batch (recommended for many images)
requests = [
    GeminiBatchService.build_image_request(f"A big letter {letter} surrounded by animals")
    for letter in "ABC"
]

batch_service.create_jsonl_file(
    requests=requests,
    output_path="image_requests.jsonl",
    keys=["letter-A", "letter-B", "letter-C"]
)

job = batch_service.create_file_batch(
    file_path="image_requests.jsonl",
    model_name="gemini-3-pro-image-preview",
    display_name="bulk-image-generation"
)
```

### Gemini File-Based Requests

Recommended for larger request sets. Maximum file size: **2 GB**.

#### JSONL File Format

Each line must be a JSON object with:
- `key`: Unique identifier to correlate with the response
- `request`: A valid GenerateContentRequest object

```json
{"key": "question-1", "request": {"contents": [{"parts": [{"text": "What is Python?"}]}], "generation_config": {"temperature": 0.7}}}
{"key": "question-2", "request": {"contents": [{"parts": [{"text": "What is JavaScript?"}]}]}}
{"key": "question-3", "request": {"contents": [{"parts": [{"text": "What is Rust?"}]}]}}
```

#### Create and Upload JSONL File

```python
from nono.connector.genai_batch_processing import GeminiBatchService

batch_service = GeminiBatchService(api_key="your-api-key")

# Option A: Create file manually
requests = [
    {"contents": [{"parts": [{"text": "Describe photosynthesis."}]}]},
    {"contents": [{"parts": [{"text": "Ingredients in a Margherita pizza."}]}]},
    {"contents": [{"parts": [{"text": "What is machine learning?"}]}]}
]

# Use helper to create JSONL file
file_path = batch_service.create_jsonl_file(
    requests=requests,
    output_path="my-requests.jsonl",
    keys=["photo", "pizza", "ml"]  # Custom keys (optional)
)

# Create job from local file
job = batch_service.create_file_batch(
    file_path=file_path,
    model_name="gemini-3-flash-preview",
    display_name="batch-from-file"
)

print(f"Job created: {job.name}")
```

#### Use Already Uploaded File

```python
# If you already have a file uploaded via the File API
job = batch_service.create_file_batch(
    uploaded_file_name="files/abc123xyz",
    model_name="gemini-3-flash-preview",
    display_name="batch-existing-file"
)
```

---

## OpenAI Batch API

The OpenAI Batch API supports multiple endpoints with 50% cost savings and separate rate limit pools.

### Supported Endpoints

| Endpoint | Use Case |
|:---------|:---------|
| `/v1/chat/completions` | Chat completions (default) |
| `/v1/responses` | Responses API |
| `/v1/embeddings` | Text embeddings |
| `/v1/completions` | Legacy completions |
| `/v1/moderations` | Content moderation |

### OpenAI Chat Completions

```python
from nono.connector.genai_batch_processing import OpenAIBatchService

batch_service = OpenAIBatchService(api_key="your-api-key")

# Method 1: Build requests from prompts
requests = OpenAIBatchService.build_requests_from_prompts(
    prompts=[
        "Explain quantum computing.",
        "What is machine learning?",
        "Describe blockchain technology."
    ],
    model="gpt-4o-mini",
    system_prompt="You are a tech expert. Answer in 2-3 sentences.",
    temperature=0.7
)

# Method 2: Build individual request
request = OpenAIBatchService.build_request(
    prompt="What is artificial intelligence?",
    model="gpt-4o-mini",
    custom_id="ai-question-1",
    system_prompt="You are a tech expert.",
    temperature=0.5,
    max_tokens=150
)

# Create JSONL file
batch_service.create_jsonl_file(requests, "chat_batch.jsonl")

# Create batch (file is auto-uploaded)
batch = batch_service.create_batch(
    file_path="chat_batch.jsonl",
    endpoint="/v1/chat/completions",
    metadata={"description": "tech questions batch"}
)

print(f"Batch ID: {batch.id}")
print(f"Status: {batch.status}")
```

### OpenAI Embeddings

```python
from nono.connector.genai_batch_processing import OpenAIBatchService

batch_service = OpenAIBatchService(api_key="your-api-key")

# Build embedding requests
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming industries.",
    "Python is a popular programming language."
]

requests = [
    OpenAIBatchService.build_embedding_request(
        input_text=text,
        model="text-embedding-3-small",
        custom_id=f"embed-{i+1}"
    )
    for i, text in enumerate(texts)
]

# Create and submit
batch_service.create_jsonl_file(requests, "embedding_batch.jsonl")
batch = batch_service.create_batch(
    file_path="embedding_batch.jsonl",
    endpoint="/v1/embeddings"
)

# Wait and get results
completed = batch_service.wait_for_completion(batch.id)
if completed.status == OpenAIBatchService.STATE_COMPLETED:
    results = batch_service.get_results(batch.id)
    for r in results:
        embedding = r["response"]["body"]["data"][0]["embedding"]
        print(f"{r['custom_id']}: {len(embedding)} dimensions")
```

### OpenAI Moderations

```python
from nono.connector.genai_batch_processing import OpenAIBatchService

batch_service = OpenAIBatchService(api_key="your-api-key")

# Text-only moderation
text_requests = [
    OpenAIBatchService.build_moderation_request(
        input_content="This is a harmless test sentence.",
        custom_id="mod-text-1"
    ),
    OpenAIBatchService.build_moderation_request(
        input_content="Another test for moderation.",
        custom_id="mod-text-2"
    )
]

# Multimodal moderation (text + image)
multimodal_request = OpenAIBatchService.build_moderation_request(
    input_content=[
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ],
    custom_id="mod-multimodal-1"
)

all_requests = text_requests + [multimodal_request]

# Create and submit
batch_service.create_jsonl_file(all_requests, "moderation_batch.jsonl")
batch = batch_service.create_batch(
    file_path="moderation_batch.jsonl",
    endpoint="/v1/moderations"
)
```

### OpenAI JSONL File Format

Each line in the JSONL file must contain:

```json
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello!"}]}}
```

| Field | Description |
|:------|:------------|
| `custom_id` | Unique identifier for correlating results |
| `method` | HTTP method (always "POST") |
| `url` | API endpoint |
| `body` | Request body (same as synchronous API) |

---

## Monitoring and Results

### Gemini Job States

| State | Description |
|:------|:------------|
| `JOB_STATE_PENDING` | Queued, waiting for processing |
| `JOB_STATE_RUNNING` | Currently being processed |
| `JOB_STATE_SUCCEEDED` | Completed successfully |
| `JOB_STATE_FAILED` | Failed |
| `JOB_STATE_CANCELLED` | Cancelled |
| `JOB_STATE_EXPIRED` | Job expired (ran or pending for more than 48 hours) |

### OpenAI Batch States

| State | Description |
|:------|:------------|
| `validating` | Input file is being validated |
| `failed` | Input file failed validation |
| `in_progress` | Batch is currently being processed |
| `finalizing` | Batch completed, results being prepared |
| `completed` | Batch completed, results ready |
| `expired` | Batch did not complete within 24 hours |
| `cancelling` | Batch is being cancelled (up to 10 min) |
| `cancelled` | Batch was cancelled |

### Check Status - Gemini

```python
# Simple status check
status = batch_service.get_job_status(job.name)
print(f"Status: {status}")

# Get complete job object
job_details = batch_service.get_job(job.name)
print(f"Status: {job_details.state}")
print(f"Name: {job_details.name}")
```

### Check Status - OpenAI

```python
# Simple status check
status = batch_service.get_batch_status(batch.id)
print(f"Status: {status}")

# Get complete batch object
batch_details = batch_service.get_batch(batch.id)
print(f"Status: {batch_details.status}")
print(f"Total: {batch_details.request_counts.total}")
print(f"Completed: {batch_details.request_counts.completed}")
print(f"Failed: {batch_details.request_counts.failed}")
```

### Wait for Completion - Gemini

```python
# Wait with progress callback
def progress(job, elapsed_seconds):
    print(f"[{elapsed_seconds:.0f}s] Status: {job.state}")

completed_job = batch_service.wait_for_completion(
    job_name=job.name,
    poll_interval=30,        # Check every 30 seconds
    timeout=3600,            # Max 1 hour (None = no limit)
    callback=progress
)

if completed_job.state == GeminiBatchService.STATE_SUCCEEDED:
    print("✓ Job completed successfully!")
elif completed_job.state == GeminiBatchService.STATE_FAILED:
    print("✗ Job failed")
```

### Wait for Completion - OpenAI

```python
# Wait with progress callback showing request counts
def progress(batch, elapsed_seconds):
    counts = batch.request_counts
    print(f"[{elapsed_seconds:.0f}s] Status: {batch.status} - {counts.completed}/{counts.total}")

completed = batch_service.wait_for_completion(
    batch_id=batch.id,
    poll_interval=30,
    timeout=3600,
    callback=progress
)

if completed.status == OpenAIBatchService.STATE_COMPLETED:
    print("✓ Batch completed successfully!")
elif completed.status == OpenAIBatchService.STATE_FAILED:
    print("✗ Batch failed validation")
elif completed.status == OpenAIBatchService.STATE_EXPIRED:
    print("✗ Batch expired (24h timeout)")
```

### Get Results - Gemini

#### For inline jobs:

```python
if completed_job.state == GeminiBatchService.STATE_SUCCEEDED:
    results = batch_service.get_inline_results(job.name)
    
    for i, response in enumerate(results):
        print(f"Response {i+1}: {response}")
```

#### For file-based jobs:

```python
if completed_job.state == GeminiBatchService.STATE_SUCCEEDED:
    # Option 1: Get as list of dictionaries
    results = batch_service.get_file_results(job.name)
    for result in results:
        print(f"Key: {result.get('key')}, Response: {result.get('response')}")
    
    # Option 2: Save to file
    output_path = batch_service.get_file_results(
        job.name, 
        output_path="results.jsonl"
    )
    print(f"Results saved to: {output_path}")
```

### Get Results - OpenAI

```python
if completed.status == OpenAIBatchService.STATE_COMPLETED:
    # Option 1: Get as list of dictionaries
    results = batch_service.get_results(batch.id)
    for r in results:
        custom_id = r["custom_id"]
        response = r["response"]["body"]
        content = response["choices"][0]["message"]["content"]
        print(f"{custom_id}: {content}")
    
    # Option 2: Save to file
    output_path = batch_service.get_results(batch.id, "output.jsonl")
    print(f"Results saved to: {output_path}")

# Get errors (if any)
errors = batch_service.get_errors(batch.id)
if errors:
    for e in errors:
        print(f"Error for {e['custom_id']}: {e['error']}")
```
```

---

## Helpers and Utilities

### Build Individual Request

```python
request = GeminiBatchService.build_request(
    prompt="What is the speed of light?",
    system_instruction="You are a scientist.",
    temperature=0.5,
    max_tokens=100,
    top_p=0.9,
    top_k=40,
    stop_sequences=[".", "\n"]
)
```

### Build Multiple Requests from Prompts

```python
prompts = [
    "Explain the theory of relativity.",
    "What is quantum mechanics?",
    "Describe the Big Bang."
]

requests = GeminiBatchService.build_requests_from_prompts(
    prompts=prompts,
    system_instruction="Explain as if to a 10-year-old.",
    temperature=0.8,
    max_tokens=150
)
```

### Create JSONL File

```python
# With auto-generated keys (request-1, request-2, ...)
path = batch_service.create_jsonl_file(
    requests=requests,
    output_path="requests.jsonl"
)

# With custom keys
path = batch_service.create_jsonl_file(
    requests=requests,
    output_path="requests.jsonl",
    keys=["relativity", "quantum", "bigbang"]
)
```

### List All Jobs

```python
jobs = batch_service.list_jobs(page_size=50)
for job in jobs:
    print(f"{job.name}: {job.state}")
```

### Cancel a Job

```python
cancelled_job = batch_service.cancel_job(job.name)
print(f"Job cancelled: {cancelled_job.state}")
```

### Delete a Job

```python
batch_service.delete_job(job.name)
print("Job deleted")
```

---

## Complete Examples

### Example 1: Bulk Text Classification

```python
from nono.connector.genai_batch_processing import GeminiBatchService

batch_service = GeminiBatchService(api_key="your-api-key")

# Texts to classify
texts = [
    "I loved the movie, highly recommend it!",
    "Terrible service, never coming back to this restaurant.",
    "The product is okay, does what it's supposed to.",
    "Amazing experience, exceeded my expectations!",
    "Didn't like it at all, very disappointed."
]

# Build classification requests
requests = [
    GeminiBatchService.build_request(
        prompt=f'Classify the following text as POSITIVE, NEGATIVE, or NEUTRAL:\n\n"{text}"',
        system_instruction="You are a sentiment classifier. Respond with one word only: POSITIVE, NEGATIVE, or NEUTRAL.",
        temperature=0.1  # Low temperature for consistency
    )
    for text in texts
]

# Create job
job = batch_service.create_inline_batch(
    requests=requests,
    display_name="sentiment-classification"
)

# Wait and get results
completed = batch_service.wait_for_completion(job.name, poll_interval=10)

if completed.state == GeminiBatchService.STATE_SUCCEEDED:
    results = batch_service.get_inline_results(job.name)
    
    for text, result in zip(texts, results):
        print(f"Text: {text[:50]}...")
        print(f"Sentiment: {result}\n")
```

### Example 2: Bulk Translation with File

```python
from nono.connector.genai_batch_processing import GeminiBatchService

batch_service = GeminiBatchService(api_key="your-api-key")

# English texts to translate
english_texts = [
    "Hello, how are you?",
    "The weather is beautiful today.",
    "I love programming in Python.",
    "Artificial intelligence is the future.",
    # ... hundreds more texts
]

# Build requests
requests = [
    GeminiBatchService.build_request(
        prompt=f"Translate to Spanish:\n\n{text}",
        system_instruction="You are a professional translator. Provide only the translation, no explanations.",
        temperature=0.3
    )
    for text in english_texts
]

# Create JSONL file (recommended for large volumes)
keys = [f"text-{i+1}" for i in range(len(english_texts))]
file_path = batch_service.create_jsonl_file(
    requests=requests,
    output_path="translations.jsonl",
    keys=keys
)

# Create job from file
job = batch_service.create_file_batch(
    file_path=file_path,
    display_name="bulk-translation-es"
)

print(f"Job created: {job.name}")
print("Waiting for completion (this may take time for large batches)...")

# Wait with 2-hour timeout
try:
    completed = batch_service.wait_for_completion(
        job.name,
        poll_interval=60,
        timeout=7200,
        callback=lambda j, t: print(f"[{t/60:.1f} min] {j.state}")
    )
    
    # Save results
    if completed.state == GeminiBatchService.STATE_SUCCEEDED:
        batch_service.get_file_results(job.name, "translations_result.jsonl")
        print("✓ Translations saved to translations_result.jsonl")
        
except TimeoutError as e:
    print(f"Timeout: {e}")
    # Option: cancel or continue waiting later
```

### Example 3: Data Processing Pipeline

```python
from nono.connector.genai_batch_processing import GeminiBatchService
import json

def process_dataset_with_batch(data: list[dict], api_key: str) -> list[dict]:
    """
    Process a dataset using batch processing.
    Extracts structured information from each record.
    """
    batch_service = GeminiBatchService(api_key=api_key)
    
    # Build prompts for each record
    requests = []
    for record in data:
        prompt = f"""
        Extract the following information from the text:
        - Person's name
        - Age (if mentioned)
        - Location (if mentioned)
        
        Text: {record['text']}
        
        Respond in JSON format.
        """
        requests.append(GeminiBatchService.build_request(
            prompt=prompt,
            system_instruction="You are a data extractor. Respond only with valid JSON.",
            temperature=0.1
        ))
    
    # Process in batch
    job = batch_service.create_inline_batch(
        requests=requests,
        display_name="data-extraction"
    )
    
    completed = batch_service.wait_for_completion(job.name)
    
    if completed.state == GeminiBatchService.STATE_SUCCEEDED:
        results = batch_service.get_inline_results(job.name)
        
        # Combine results with original data
        for i, result in enumerate(results):
            try:
                data[i]['extracted'] = json.loads(result)
            except json.JSONDecodeError:
                data[i]['extracted'] = {"error": "Could not parse"}
    
    return data

# Usage
dataset = [
    {"id": 1, "text": "John is 25 years old and lives in New York."},
    {"id": 2, "text": "Mary works as an engineer in San Francisco."},
    {"id": 3, "text": "Peter is looking for a job."}
]

results = process_dataset_with_batch(dataset, "your-api-key")
for r in results:
    print(f"ID {r['id']}: {r.get('extracted', 'N/A')}")
```

---

## API Reference

### Class `GeminiBatchService`

#### Constructor

```python
GeminiBatchService(api_key: str | None = None)
```

#### Main Methods

| Method | Description |
|:-------|:------------|
| `create_inline_batch()` | Create job with inline requests (< 20 MB) |
| `create_file_batch()` | Create job from JSONL file (< 2 GB) |
| `get_job()` | Get the complete job object |
| `get_job_status()` | Get only the job status |
| `wait_for_completion()` | Blocking wait until job completes |
| `get_inline_results()` | Get results from inline job |
| `get_file_results()` | Get results from file-based job |
| `cancel_job()` | Cancel a running job |
| `list_jobs()` | List all jobs |
| `delete_job()` | Delete a job |
| `create_jsonl_file()` | Create JSONL file from request list |

#### Static Methods (Helpers)

| Method | Description |
|:-------|:------------|
| `build_request()` | Build a GenerateContentRequest with full config options |
| `build_requests_from_prompts()` | Build multiple requests from prompt list |
| `build_request_with_search()` | Build request with Google Search tool enabled |
| `build_structured_request()` | Build request for structured JSON output |
| `build_image_request()` | Build request for image generation |

#### `build_request()` Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `prompt` | `str` | The user prompt/question (required) |
| `system_instruction` | `str` | System instruction |
| `temperature` | `float` | Temperature setting (0.0 to 2.0) |
| `max_tokens` | `int` | Maximum output tokens |
| `top_p` | `float` | Top-p sampling parameter |
| `top_k` | `int` | Top-k sampling parameter |
| `stop_sequences` | `list[str]` | Stop sequences |
| `tools` | `list[dict]` | Tools (e.g., `[{"google_search": {}}]`) |
| `response_mime_type` | `str` | Response MIME type (`"application/json"`) |
| `response_schema` | `Any` | Response schema (Pydantic model or dict) |
| `response_modalities` | `list[str]` | Response modalities (`["TEXT", "IMAGE"]`) |

#### State Constants

```python
GeminiBatchService.STATE_PENDING     # "JOB_STATE_PENDING"
GeminiBatchService.STATE_RUNNING     # "JOB_STATE_RUNNING"
GeminiBatchService.STATE_SUCCEEDED   # "JOB_STATE_SUCCEEDED"
GeminiBatchService.STATE_FAILED      # "JOB_STATE_FAILED"
GeminiBatchService.STATE_CANCELLED   # "JOB_STATE_CANCELLED"
GeminiBatchService.STATE_EXPIRED     # "JOB_STATE_EXPIRED"
```

### Class `OpenAIBatchService`

#### Constructor

```python
OpenAIBatchService(api_key: str | None = None, base_url: str | None = None)
```

#### Main Methods

| Method | Description |
|:-------|:------------|
| `upload_file()` | Upload a JSONL file for batch processing |
| `create_batch()` | Create a batch job (auto-uploads file if path given) |
| `get_batch()` | Get the complete batch object |
| `get_batch_status()` | Get only the batch status |
| `wait_for_completion()` | Blocking wait until batch completes |
| `get_results()` | Get results from completed batch |
| `get_errors()` | Get errors from batch (if any) |
| `cancel_batch()` | Cancel an ongoing batch |
| `list_batches()` | List all batches with pagination |
| `create_jsonl_file()` | Create JSONL file from request list |

#### Static Methods (Helpers)

| Method | Description |
|:-------|:------------|
| `build_request()` | Build a chat completion batch request |
| `build_requests_from_prompts()` | Build multiple requests from prompt list |
| `build_embedding_request()` | Build an embedding batch request |
| `build_moderation_request()` | Build a moderation batch request |

#### Endpoint Constants

```python
OpenAIBatchService.ENDPOINT_CHAT_COMPLETIONS  # "/v1/chat/completions"
OpenAIBatchService.ENDPOINT_RESPONSES         # "/v1/responses"
OpenAIBatchService.ENDPOINT_EMBEDDINGS        # "/v1/embeddings"
OpenAIBatchService.ENDPOINT_COMPLETIONS       # "/v1/completions"
OpenAIBatchService.ENDPOINT_MODERATIONS       # "/v1/moderations"
```

#### State Constants

```python
OpenAIBatchService.STATE_VALIDATING   # "validating"
OpenAIBatchService.STATE_FAILED       # "failed"
OpenAIBatchService.STATE_IN_PROGRESS  # "in_progress"
OpenAIBatchService.STATE_FINALIZING   # "finalizing"
OpenAIBatchService.STATE_COMPLETED    # "completed"
OpenAIBatchService.STATE_EXPIRED      # "expired"
OpenAIBatchService.STATE_CANCELLING   # "cancelling"
OpenAIBatchService.STATE_CANCELLED    # "cancelled"
```

---

## Best Practices

### 1. Choose the Right Provider

| Use Case | Recommended |
|:---------|:------------|
| Large file support (> 200 MB) | Gemini |
| Inline requests (no file needed) | Gemini |
| Embeddings at scale | OpenAI |
| Content moderation | OpenAI |
| Chat completions | Both |

### 2. Choose the Right Method

| Number of Requests | Gemini | OpenAI |
|:-------------------|:-------|:-------|
| < 100 | Inline requests | JSONL file |
| 100 - 10,000 | JSONL file | JSONL file |
| > 10,000 | JSONL file (partition) | Multiple batches |

### 3. Optimize Polling

```python
# For small jobs (< 100 requests)
poll_interval = 10  # 10 seconds

# For medium jobs (100 - 1000)
poll_interval = 30  # 30 seconds

# For large jobs (> 1000)
poll_interval = 60  # 1 minute or more
```

### 4. Handle Errors Appropriately

#### Gemini:

```python
try:
    completed = batch_service.wait_for_completion(job.name, timeout=3600)
except TimeoutError:
    status = batch_service.get_job_status(job.name)
    if status == GeminiBatchService.STATE_RUNNING:
        pass  # Still running, wait more
    else:
        batch_service.cancel_job(job.name)
```

#### OpenAI:

```python
try:
    completed = batch_service.wait_for_completion(batch.id, timeout=3600)
except TimeoutError:
    status = batch_service.get_batch_status(batch.id)
    if status == OpenAIBatchService.STATE_IN_PROGRESS:
        pass  # Still running, wait more
    else:
        batch_service.cancel_batch(batch.id)

# Always check for errors
errors = batch_service.get_errors(batch.id)
if errors:
    for e in errors:
        print(f"Failed: {e['custom_id']} - {e['error']}")
```

### 5. Use Meaningful IDs

```python
# ❌ Bad: generic keys/IDs
keys = ["1", "2", "3"]

# ✅ Good: keys that identify the content
# Gemini
keys = ["product-A-review-1", "product-A-review-2", "product-B-review-1"]

# OpenAI
custom_id_prefix = "sentiment-analysis"  # Results in "sentiment-analysis-1", etc.
```

### 6. Monitor Progress

#### Gemini:

```python
def my_callback(job, elapsed):
    minutes = elapsed / 60
    print(f"[{minutes:.1f} min] Status: {job.state}")
    
batch_service.wait_for_completion(job.name, callback=my_callback)
```

#### OpenAI:

```python
def my_callback(batch, elapsed):
    minutes = elapsed / 60
    counts = batch.request_counts
    print(f"[{minutes:.1f} min] {batch.status}: {counts.completed}/{counts.total}")
    
batch_service.wait_for_completion(batch.id, callback=my_callback)
```

---

## Rate Limits

### Gemini Batch API

- No specific per-batch limits documented
- Uses separate quota from synchronous API

### OpenAI Batch API

| Limit | Value |
|:------|:------|
| Requests per batch | 50,000 |
| Input file size | 200 MB |
| Embedding inputs per batch | 50,000 |
| Batch creation rate | 2,000/hour |

---

## Troubleshooting

### Gemini: "Request size exceeds limit"

**Solution**: Switch from inline to file-based, or split into multiple jobs.

### Gemini: Job stays in PENDING

**Possible causes**: High queue volume or high-demand model.
**Solution**: Wait or use an alternative model.

### OpenAI: Batch stuck in "validating"

**Check**: 
1. JSONL format is correct
2. All requests have unique `custom_id`
3. All requests target the same endpoint
4. Model is batch-compatible

### OpenAI: Batch expired

**Cause**: Batch did not complete within 24 hours.
**Solution**: 
- Split into smaller batches
- Errors are available via `get_errors()`

### Empty results

**Check**:
1. State is `STATE_SUCCEEDED` (Gemini) or `STATE_COMPLETED` (OpenAI)
2. Using correct method: `get_inline_results()` vs `get_file_results()` (Gemini)
3. Check `output_file_id` exists (OpenAI)

---

## Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| `google-genai` | >= 1.0.0 | Google Gemini SDK ([docs](https://ai.google.dev/gemini-api/docs)) |
| `openai` | >= 1.0.0 | OpenAI SDK for batch API |
| `requests` | >= 2.28.0 | HTTP library for API calls |

---

## Contact

- **Author**: [DatamanEdge](https://github.com/DatamanEdge)
- **Email**: [jrodriguezga@outlook.com](mailto:jrodriguezga@outlook.com)
- **LinkedIn**: [Javier Rodríguez](https://es.linkedin.com/in/javier-rodriguez-ga)

---

## License

MIT © 2026 DatamanEdge. See [LICENSE](../../LICENSE).
