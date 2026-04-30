---
name: generating-api-docs
description: >
  Generate API documentation in OpenAPI format or structured Markdown.
  Use when the user asks to document an API, create API reference,
  generate endpoint docs, or produce OpenAPI/Swagger specs.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - code
  - api
  - documentation
  - openapi
version: "1.0.0"
temperature: 0.1
output_format: json
tools:
  - name: validate_schema
    script: scripts/validate_schema.py
    description: Validate generated OpenAPI schema against the specification.
---

# Generate API Documentation

Produce comprehensive API documentation from code, route definitions,
or natural language descriptions.

## Output formats

| Format | Use case |
|---|---|
| **OpenAPI 3.1** (default) | Machine-readable spec for Swagger UI, code generation |
| **Markdown** | Human-readable API reference for READMEs |
| **Both** | Spec + readable docs combined |

## Guidelines

1. **Complete descriptions**: Every endpoint, parameter, and response
   must have a clear `description`.
2. **Examples**: Include `example` values for request bodies and responses.
3. **Error responses**: Document 4xx and 5xx responses, not just 200.
4. **Authentication**: Document required auth (Bearer, API key, OAuth).
5. **Data types**: Use precise types and formats (`string/date-time`,
   `integer/int64`, `string/email`).
6. **Tags**: Group endpoints by resource or domain.

## OpenAPI template

```yaml
openapi: "3.1.0"
info:
  title: API Name
  description: Brief API description.
  version: "1.0.0"
servers:
  - url: https://api.example.com/v1
    description: Production
paths:
  /resource:
    get:
      tags: [Resources]
      summary: List resources
      description: Retrieve a paginated list of resources.
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
          description: Page number.
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
          description: Items per page.
      responses:
        "200":
          description: Successful response.
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: "#/components/schemas/Resource"
                  total:
                    type: integer
              example:
                data:
                  - id: 1
                    name: "Example"
                total: 42
        "401":
          description: Unauthorized — missing or invalid API key.
components:
  schemas:
    Resource:
      type: object
      required: [id, name]
      properties:
        id:
          type: integer
          description: Unique identifier.
        name:
          type: string
          description: Resource name.
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
security:
  - ApiKeyAuth: []
```

## Example

**Input**: "Document a REST API for a task manager with endpoints: create task, list tasks, get task, update task, delete task"

**Output** (abbreviated):
```yaml
openapi: "3.1.0"
info:
  title: Task Manager API
  version: "1.0.0"
paths:
  /tasks:
    get:
      summary: List all tasks
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [pending, in_progress, completed]
      responses:
        "200":
          description: List of tasks.
    post:
      summary: Create a new task
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/TaskCreate"
      responses:
        "201":
          description: Task created.
  /tasks/{id}:
    get:
      summary: Get task by ID
    put:
      summary: Update task
    delete:
      summary: Delete task
```
