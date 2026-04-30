---
name: writing-emails
description: >
  Compose professional emails with appropriate tone and structure.
  Use when the user asks to write, draft, or compose an email,
  message, or professional communication.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - text
  - communication
  - writing
  - email
version: "1.0.0"
temperature: 0.5
---

# Write Emails

Compose clear, professional emails adapted to context, audience,
and purpose.

## Email types

| Type | Tone | Key element |
|---|---|---|
| **Business** | Formal, concise | Clear call to action |
| **Follow-up** | Polite, persistent | Reference to previous context |
| **Request** | Respectful, specific | What, why, and when |
| **Announcement** | Informative, positive | Key dates and impact |
| **Apology** | Sincere, solution-focused | Acknowledgment + corrective action |
| **Thank you** | Warm, genuine | Specific contribution noted |
| **Cold outreach** | Engaging, brief | Value proposition in first line |

## Guidelines

1. **Subject line**: Specific and actionable (max 60 characters).
   Bad: "Update" → Good: "Q3 Budget Review — Action Needed by Friday"
2. **Opening**: State purpose in the first sentence. No filler.
3. **Body**: One idea per paragraph. Use bullet points for lists.
4. **Call to action**: End with a clear, specific next step.
5. **Tone matching**: Match the user's requested formality level.
6. **Length**: Keep under 200 words unless complexity requires more.
7. **Sign-off**: Match the formality (Best regards / Thanks / Cheers).

## Structure

```
Subject: [Specific, actionable subject line]

Hi [Name],

[One-sentence purpose statement.]

[Supporting details — 1-2 short paragraphs or bullet points.]

[Clear call to action with specific deadline if applicable.]

[Professional sign-off],
[Sender name]
```

## Examples

**Input**: "Write an email to a client requesting a meeting to discuss project delays"

**Output**:
```
Subject: Project Timeline Discussion — Meeting Request

Hi Sarah,

I'd like to schedule a 30-minute call this week to discuss the current
project timeline and align on adjusted milestones.

We've identified two areas where the original schedule needs revision:
- The data migration phase is taking longer than estimated.
- We're waiting on API credentials from the third-party vendor.

I have availability on Thursday 2-4 PM or Friday morning.
Could you let me know what works for your schedule?

Best regards,
[Your name]
```

**Input**: "Write a thank-you email after a job interview"

**Output**:
```
Subject: Thank You — Software Engineer Interview

Hi David,

Thank you for taking the time to meet with me today. I enjoyed
learning about the team's work on the real-time data pipeline
and the challenges around scaling the event processing system.

Our conversation reinforced my enthusiasm for the role. My experience
with Apache Kafka and distributed systems aligns well with the
problems you described, and I'd be excited to contribute.

I look forward to hearing about next steps. Please don't hesitate
to reach out if you need any additional information.

Best regards,
[Your name]
```
