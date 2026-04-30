# Classification Reference

## Category Design Guidelines

- **Mutually exclusive**: Each input should map to exactly one primary category.
- **Exhaustive**: All possible inputs should have at least one matching category.
- **Clear boundaries**: Avoid overlapping categories; when overlap exists, document tie-breaking rules.
- **Consistent granularity**: Categories should be at the same level of abstraction.

## Confidence Scoring

| Range | Meaning | Action |
|-------|---------|--------|
| 0.90 – 1.00 | Very high confidence | Accept directly |
| 0.70 – 0.89 | High confidence | Accept with secondary labels |
| 0.50 – 0.69 | Moderate confidence | Flag for review |
| 0.00 – 0.49 | Low confidence | Escalate / request more context |

## Common Classification Domains

| Domain | Typical Categories |
|--------|-------------------|
| Sentiment | positive, negative, neutral, mixed |
| Intent | question, command, statement, greeting |
| Topic | technology, finance, health, education, sports |
| Priority | critical, high, medium, low |
| Language | en, es, fr, de, pt, zh, ja, ko |
| Content safety | safe, sensitive, toxic, spam |
