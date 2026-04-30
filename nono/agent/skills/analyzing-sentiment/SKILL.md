---
name: analyzing-sentiment
description: >
  Analyze text sentiment, emotion, and tone.
  Use when the user asks to detect sentiment, emotions, mood,
  or overall tone in text, reviews, feedback, or social media posts.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - text
  - sentiment
  - nlp
  - analysis
version: "1.0.0"
temperature: 0.2
output_format: json
---

# Analyze Sentiment

Detect sentiment, emotion, and tone in text with structured output.

## Analysis dimensions

1. **Sentiment**: Positive, negative, neutral, or mixed — with a confidence
   score between 0.0 and 1.0.
2. **Emotion**: Primary emotion detected (joy, anger, sadness, fear, surprise,
   disgust, trust, anticipation).
3. **Tone**: Professional, casual, sarcastic, urgent, formal, friendly, etc.
4. **Intensity**: How strong the sentiment is on a 1–5 scale.

## Guidelines

- Analyze the entire text holistically, not sentence by sentence.
- When sentiment is mixed, report the dominant sentiment and note the
  secondary sentiment in the `notes` field.
- Sarcasm and irony should be detected; mark `sarcasm_detected: true`.
- For multilingual text, detect the language and analyze in context.

## Additional resources

- For emotion taxonomy and scoring guidelines, see [references/REFERENCE.md](references/REFERENCE.md)

## Output format

```json
{
  "sentiment": "positive|negative|neutral|mixed",
  "confidence": 0.92,
  "emotion": "joy",
  "tone": "enthusiastic",
  "intensity": 4,
  "sarcasm_detected": false,
  "language": "en",
  "notes": "Optional additional observations"
}
```

## Examples

**Input**: "This product is absolutely amazing! Best purchase I've made all year."
**Output**:
```json
{
  "sentiment": "positive",
  "confidence": 0.97,
  "emotion": "joy",
  "tone": "enthusiastic",
  "intensity": 5,
  "sarcasm_detected": false,
  "language": "en",
  "notes": ""
}
```

**Input**: "Oh great, another update that breaks everything. Just what I needed."
**Output**:
```json
{
  "sentiment": "negative",
  "confidence": 0.89,
  "emotion": "anger",
  "tone": "sarcastic",
  "intensity": 4,
  "sarcasm_detected": true,
  "language": "en",
  "notes": "Sarcastic tone masking frustration"
}
```
