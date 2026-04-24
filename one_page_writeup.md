# Ask First Pattern Detection — One Page Writeup

## Approach to the Reasoning Problem

The core challenge is not retrieval — it is causal temporal reasoning. Two events in a
conversation history are only meaningfully connected if the time gap between them is
consistent with known biological or behavioral mechanisms. A symptom appearing 8 weeks
after a dietary change means something. The same symptom appearing 2 days before that
change means nothing.

The system approaches this in three layers:

1. Full-history context: Each user's entire 3-month conversation history is sent to the
model in a single call, with exact timestamps. This ensures the model can compute actual
time deltas rather than reasoning about vague sequence. Session ordering alone is
insufficient — the gap between Session 1 (Jan 8) and Session 6 (Feb 19) being 6 weeks
is what makes the hair fall connection to calorie restriction clinically meaningful.

2. Structured temporal reasoning prompt: The system prompt explicitly instructs the model
to reason across time, show what it ruled out, and justify why a connection is causal
rather than coincidental. This forces the model to surface the mechanism (e.g., telogen
effluvium, cortisol-prostaglandin interaction) not just the association.

3. Streaming reasoning trace: The output is streamed live, making the model's
consideration process visible before the final JSON is produced. This is not cosmetic —
it surfaces where the model is uncertain or over-reaches.

The context management decision was to send full user history in one shot rather than
chunking. Chunking would be fatal to temporal reasoning: a pattern where cause and effect
are 6–8 weeks apart would be split across chunks and never connected. The dataset is small
enough that single-shot is both feasible and correct.

## Where the System Fails or Hallucinates Confidently

1. Symptom timing hallucination: The model knows clinical facts (e.g., "telogen effluvium
manifests 6-12 weeks after deficiency onset") from training. If the actual timestamp gap
in the data is 4 weeks, the model may still confidently assign a causal connection and
retroactively fit the clinical timeline. It cannot truly verify calendar arithmetic — it
reasons about timestamps but does not compute them with precision. Fix: pre-process all
timestamps into explicit delta annotations (e.g., "42 days after S01") before sending
to the model.

2. Absence-of-evidence blindness: The model can observe patterns when symptoms appear but
struggles to reason about sessions where symptoms were notably absent. Pattern P1 (Arjun's
stomach pain) is stronger because there are sessions with no stomach complaint during
non-deadline weeks — but the model may not explicitly track and cite those absent sessions
as negative evidence. Fix: add a preprocessing step that annotates each session with which
symptom tags are absent relative to the user's known symptom vocabulary.

3. Confounding variable collapse: In USR003, the model correctly identifies sleep
deprivation as the driver of cramping, separating it from work stress. But this isolation
required three months of natural variation. In a shorter history, the model would likely
collapse both variables into a single "stress" pattern and miss the more specific sleep
mechanism. Fix: a multi-hypothesis scoring approach that explicitly tests each candidate
variable independently.

4. Confident medium-quality patterns: The model will produce patterns with "high"
confidence even when session count is low (2 sessions). The confidence score is
self-reported by the model, not computed from statistical support. Fix: override confidence
to "low" automatically when fewer than 3 sessions are involved, regardless of what the
model reports.

## What I Would Build Differently With More Time

- Vector store for sessions: Embed each session, retrieve relevant context per candidate
symptom rather than full history. Scales to real users with years of data.

- Structured temporal preprocessor: Compute all pairwise session time deltas before the
LLM call. Feed the model a pre-computed delta table alongside raw sessions. This makes
temporal reasoning exact, not approximate.

- Negative evidence pass: A second LLM pass that asks "in which sessions did this symptom
NOT appear?" and uses absence as confirmatory signal.

- Pattern confidence calibration: Cross-validate model confidence against a held-out set
of labeled patterns. Apply a calibration correction to the raw confidence scores.

- Incremental session ingestion: Instead of batch analysis, process each new session as
it arrives and update pattern confidence scores in real time. This is how Clary would
actually work in production.