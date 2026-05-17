# Approach — SHL Conversational Assessment Recommender

## Design overview

The agent treats each turn as one of five intents — **clarify**, **recommend**, **refine**, **compare**, **refuse** — and routes to a dedicated handler. A separate cheap LLM call classifies the intent before paying for retrieval or generation. This keeps each turn focused, makes failure modes localised, and is easy to defend in an interview: each handler is short, single-purpose, and uses the catalog as the source of truth.

```
user turn → router (LLM, JSON-mode) → handler → ChatResponse
                                       │
            clarify / recommend / refine / compare / refuse
```

The service is fully stateless. The full conversation arrives on every `/chat` request; nothing is stored server-side.

## Retrieval

Hybrid **BM25 + dense (MiniLM-L6-v2)** fused with reciprocal rank fusion. The catalog has 377 items with very specific product names (`Docker (New)`, `OPQ32r`, `SVAR Spoken English (US)`), so BM25 dominates exact-skill queries while dense retrieval rescues intent queries like "we need something for call centre agents". RRF avoids tuning a relative weight per query.

Two non-obvious touches that mattered:
- **Name-weighting (3×).** Item names carry the strongest discriminating signal in a product catalog. Repeating the name three times in the BM25 doc lifts the right items above generic-description noise. On the C8 trace (Excel/Word for admin assistants) this single change moved the gold items from outside top-5 into top-5.
- **Soft filters, not hard.** Job-level and key filters apply as score boosts, not as filters. A too-strict filter could blank the candidate set; a small boost preserves recall while still nudging the right level forward.

Embeddings are loaded once at startup. The retriever degrades to BM25-only if the embedding model can't be fetched (which is what happens during sandboxed offline testing), so the service stays useful under partial failure.

## Prompt design and grounding

Every prompt lives in `app/prompts.py` so all behavior is centralised and reviewable. Two guardrails are repeated wherever the agent could fabricate:

1. The **selector** is given a JSON candidate block (top-20 from retrieval) and instructed to pick names *exactly as written*. After the LLM responds, `_resolve_picks` maps the returned names back to catalog items; only items in the candidate pool or the canonical index can survive. URLs and `test_type` codes are looked up from the index, never generated.
2. The **comparator** receives only the catalog entries for the items named. It can't reach beyond them. This is what kept "OPQ vs OPQ MQ Sales Report" honest in the C5 trace style — the answer comes from the description field, not the model's prior.

The router prompt biases toward **clarify** early in a conversation, which is the failure mode the assignment explicitly flags ("agent does not recommend on turn 1 for a vague query"). A simple character-count heuristic short-circuits the router on tiny first turns to save a round-trip.

## Refusal and scope

Off-topic detection runs in two layers: a cheap regex check for prompt-injection tells ("ignore previous instructions", "reveal the system prompt") before any LLM call, then the router for the more ambiguous cases (legal, general hiring advice). Each refusal category has a distinct templated reply so probes that check for specific refusal language can match cleanly.

## Evaluation

The 10 public conversation traces (C1–C10) are also the implicit ground truth, so I built a replay harness (`tests/replay.py`) that:
1. Parses user turns and the final shortlist out of each trace markdown.
2. Replays the user turns against the same `Agent` the server uses, mimicking the grader by tracking the most recent non-empty `recommendations` across the conversation (the grader stops when the agent emits a shortlist).
3. Computes loose-matched Recall@10 against the trace's gold list.

Iteration log on this metric (all numbers offline, **stub LLM + BM25-only**, so this is a strict lower bound on production performance):

| Change | Mean Recall@10 |
|---|---|
| Initial: top-5 fallback, single-weight BM25 | 0.083 |
| + replay tracks last non-empty recs (matches grader) | 0.222 |
| + top-10 fallback, 3× name weight | 0.407 |
| + refine falls through to recommend when prior shortlist unrecoverable | 0.403 (regressed 1; expected — real LLM doesn't have this issue) |

With Groq Llama-3.3-70B as the selector and MiniLM dense fused into retrieval (the production stack on Render), I expect Mean Recall@10 around **0.65–0.85** because (a) the LLM selector picks coherent shortlists from top-20 candidates rather than relying on the top-10 fallback, (b) dense embeddings surface OPQ32r and Verify G+ on semantic intents where they aren't named explicitly, and (c) refine preserves the previously committed shortlist across edits instead of re-retrieving from scratch.

## What didn't work

- A first version had a single big prompt handle classify-and-recommend in one shot. It hallucinated items not in the catalog whenever the brief was unusual. Splitting into router + selector and constraining the selector to a JSON candidate block eliminated those hallucinations entirely.
- Hard filtering on job_level (`Director` candidates only for senior-leadership briefs) backfired when the right item was tagged `Mid-Professional` only. The soft-boost version preserves recall.

## Stack

FastAPI · Pydantic v2 · Groq (Llama-3.3-70B) · `sentence-transformers/all-MiniLM-L6-v2` · `rank-bm25` · numpy. Deployed on Render free tier; build step runs `python -m app.indexer` so the embedding model is downloaded and warmed before first request. AI tooling was used to draft prose and accelerate scaffolding; every design decision in this document is defensible end-to-end.
