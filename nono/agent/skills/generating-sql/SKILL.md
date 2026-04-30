---
name: generating-sql
description: >
  Generate SQL queries from natural language descriptions.
  Use when the user asks to write, create, or build SQL queries,
  convert questions to SQL, or query a database.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - code
  - sql
  - data
  - database
version: "1.0.0"
temperature: 0.1
output_format: json
---

# Generate SQL

Convert natural language questions into correct, optimized SQL queries.

## Supported dialects

- **PostgreSQL** (default)
- MySQL / MariaDB
- SQLite
- SQL Server (T-SQL)
- BigQuery (Standard SQL)

If the user specifies a dialect, generate dialect-specific syntax.
Otherwise default to standard SQL / PostgreSQL.

## Guidelines

1. **Safety first**: Never generate `DROP`, `TRUNCATE`, or `DELETE` without
   explicit `WHERE` clauses. Always warn for destructive operations.
2. **Parameterized queries**: Use placeholders (`$1`, `?`, `:param`) instead
   of inline values to prevent SQL injection.
3. **Performance**: Prefer indexed columns in `WHERE` and `JOIN` conditions.
   Avoid `SELECT *` — list specific columns.
4. **Readability**: Use uppercase for SQL keywords, meaningful aliases,
   and proper indentation.
5. **Schema awareness**: If the user provides a schema, respect column names
   and types exactly. If not, use reasonable defaults and note assumptions.

## Output format

```json
{
  "query": "SELECT u.name, COUNT(o.id) AS order_count\nFROM users u\nJOIN orders o ON o.user_id = u.id\nWHERE o.created_at >= $1\nGROUP BY u.name\nORDER BY order_count DESC\nLIMIT 10;",
  "dialect": "postgresql",
  "parameters": ["2024-01-01"],
  "explanation": "Finds the top 10 users by order count since the given date.",
  "warnings": [],
  "assumptions": ["Table 'users' has columns: id, name", "Table 'orders' has columns: id, user_id, created_at"]
}
```

## Examples

**Input**: "Show me the top 5 customers who spent the most last month"

**Output**:
```json
{
  "query": "SELECT\n    c.name,\n    SUM(o.total_amount) AS total_spent\nFROM customers c\nJOIN orders o ON o.customer_id = c.id\nWHERE o.order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')\n  AND o.order_date < DATE_TRUNC('month', CURRENT_DATE)\nGROUP BY c.name\nORDER BY total_spent DESC\nLIMIT 5;",
  "dialect": "postgresql",
  "parameters": [],
  "explanation": "Aggregates order totals per customer for the previous calendar month and returns the top 5 spenders.",
  "warnings": [],
  "assumptions": ["Table 'customers' has column: name", "Table 'orders' has columns: customer_id, total_amount, order_date"]
}
```
