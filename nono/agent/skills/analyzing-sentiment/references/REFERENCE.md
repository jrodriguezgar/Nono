# Sentiment Analysis Reference

## Emotion Taxonomy (Plutchik's Wheel)

| Primary Emotion | Opposite     | Mild Form      | Intense Form   |
|-----------------|-------------|----------------|----------------|
| Joy             | Sadness     | Serenity       | Ecstasy        |
| Trust           | Disgust     | Acceptance     | Admiration     |
| Fear            | Anger       | Apprehension   | Terror         |
| Surprise        | Anticipation| Distraction    | Amazement      |
| Sadness         | Joy         | Pensiveness    | Grief          |
| Disgust         | Trust       | Boredom        | Loathing       |
| Anger           | Fear        | Annoyance      | Rage           |
| Anticipation    | Surprise    | Interest       | Vigilance      |

## Confidence Scoring

| Range     | Label        | Meaning                                    |
|-----------|-------------|---------------------------------------------|
| 0.90–1.00 | Very High   | Clear, unambiguous sentiment                |
| 0.75–0.89 | High        | Strong indicators with minor ambiguity      |
| 0.50–0.74 | Moderate    | Mixed signals, context-dependent            |
| 0.25–0.49 | Low         | Predominantly neutral or unclear            |
| 0.00–0.24 | Very Low    | Insufficient signal to determine sentiment  |

## Intensity Scale

| Score | Description                                         |
|-------|-----------------------------------------------------|
| 1     | Very mild — barely detectable sentiment             |
| 2     | Mild — subtle but present                           |
| 3     | Moderate — clearly expressed                        |
| 4     | Strong — emphatic language, punctuation, emphasis   |
| 5     | Very strong — extreme language, all caps, superlatives |

## Sarcasm Indicators

- Contradiction between literal meaning and context
- Excessive superlatives in negative contexts ("Just *wonderful*")
- Quotation marks around positive words
- "Oh great", "Sure, because...", "Thanks for nothing"
