---
name: analyzing-data
description: >
  Analyze datasets to extract insights, patterns, and statistics.
  Use when the user asks to analyze data, find trends, compute
  statistics, or provide data-driven recommendations.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - data
  - analytics
  - statistics
  - insights
version: "1.0.0"
temperature: 0.2
output_format: json
---

# Analyze Data

Analyze datasets to extract meaningful insights, statistics, and
actionable recommendations.

## Analysis types

1. **Descriptive**: Summary statistics (mean, median, mode, std, min, max),
   distributions, missing values, cardinality.
2. **Diagnostic**: Correlations, anomaly detection, root cause hypotheses.
3. **Trend**: Time-series patterns, seasonality, growth rates.
4. **Comparative**: Group comparisons, segmentation, A/B analysis.
5. **Predictive hints**: Identify features that might be predictive
   (no ML model building — just suggestions).

## Guidelines

- Start with data quality: check for nulls, duplicates, outliers.
- Report statistics with appropriate precision (2 decimal places max).
- Always note the sample size and any caveats.
- When data has a time dimension, analyze trends first.
- Provide visualization recommendations (chart type + axes).

## Output format

```json
{
  "summary": "Brief overall description of the dataset and findings.",
  "data_quality": {
    "total_rows": 10000,
    "missing_values": {"column_a": 52, "column_b": 0},
    "duplicates": 3,
    "outliers_detected": ["column_c has 12 values beyond 3σ"]
  },
  "statistics": {
    "column_a": {"mean": 42.5, "median": 40.0, "std": 12.3, "min": 5, "max": 99}
  },
  "insights": [
    "Sales peak in Q4 with 35% higher volume than Q1.",
    "Customer segment A has 2.5x higher lifetime value.",
    "Strong positive correlation (r=0.87) between X and Y."
  ],
  "recommendations": [
    "Focus marketing budget on Q3 to boost Q4 conversions.",
    "Investigate 12 outliers in column_c for data entry errors."
  ],
  "visualization_suggestions": [
    {"chart": "line", "x": "date", "y": "sales", "note": "Show monthly trend"},
    {"chart": "bar", "x": "segment", "y": "avg_revenue", "note": "Compare segments"}
  ]
}
```

## Example

**Input**: "Analyze this sales data: Q1=$120K, Q2=$135K, Q3=$128K, Q4=$185K"

**Output**:
```json
{
  "summary": "Annual sales of $568K with strong Q4 seasonality.",
  "statistics": {
    "quarterly_sales": {"mean": 142000, "median": 131500, "std": 28800, "min": 120000, "max": 185000}
  },
  "insights": [
    "Q4 revenue is 37% above average, suggesting strong seasonality.",
    "Q2 shows 12.5% growth over Q1, but Q3 dips 5.2%.",
    "Total annual growth trend is positive."
  ],
  "recommendations": [
    "Investigate Q3 dip — possible seasonal factor or operational issue.",
    "Allocate extra inventory/staff for Q4 peak."
  ]
}
```
