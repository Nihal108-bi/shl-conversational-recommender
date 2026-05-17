"""FastAPI service. Endpoints per the assignment spec:

  GET  /health  -> {"status": "ok"}
  POST /chat    -> {"reply": ..., "recommendations": [...], "end_of_conversation": ...}

Plus a landing page at / so the root URL is informative, not a 404.

The /chat call is stateless. The full conversation history comes in every
request; we keep no per-conversation state on the server.
"""

from __future__ import annotations
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .schemas import ChatRequest, ChatResponse, Health
from .retriever import get_retriever
from .llm import GroqClient
from .agent import Agent


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("shl-recommender")


# Process-wide singletons. Built on startup so the first /chat call doesn't
# pay the embedding-model load cost.
_AGENT: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _AGENT
    log.info("Building retriever (loading catalog + embeddings)…")
    retriever = get_retriever()
    log.info("Retriever ready: %d items indexed.", len(retriever.items))
    llm = GroqClient()
    _AGENT = Agent(retriever=retriever, llm=llm)
    log.info("Agent ready. Model=%s stub=%s", llm.model, llm._stub)
    yield


# ----- API metadata for a polished /docs page -------------------------------

API_DESCRIPTION = """
A conversational agent that recommends SHL assessments. Takes a hiring brief
through dialogue — clarifying when vague, recommending when ready, refining
on edits, comparing items on request — and always grounds recommendations in
the SHL product catalog.

**Built for the SHL Labs AI Intern take-home by Nihal Jaiswal.**

## How to try it

Use the `POST /chat` endpoint below. The request body is a list of messages
in OpenAI chat format. The service is **stateless** — send the full
conversation history on every call.

### Example: vague brief (the agent will clarify)
```json
{
  "messages": [
    {"role": "user", "content": "We need an assessment for senior leadership."}
  ]
}
```

### Example: full job description (the agent will recommend)
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a senior backend engineer. Core Java, Spring, SQL, AWS, Docker. Senior IC."}
  ]
}
```

### Example: multi-turn refinement
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a senior backend engineer. Java, Spring, SQL, AWS, Docker."},
    {"role": "assistant", "content": "Here's a shortlist: Core Java Advanced, Spring, SQL, AWS Development, Docker, Verify G+, OPQ32r."},
    {"role": "user", "content": "Drop AWS — they won't touch infrastructure."}
  ]
}
```
"""

app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    description=API_DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    contact={
        "name": "Nihal Jaiswal",
        "url": "https://github.com/Nihal108-bi",
    },
)

# Permissive CORS — the grading harness calls from a different origin and
# nothing here is sensitive.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Landing page ---------------------------------------------------------

_LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SHL Conversational Assessment Recommender</title>
  <style>
    :root {
      --bg: #0c0a09;
      --bg-2: #1c1917;
      --ink: #f5f1eb;
      --ink-2: #c9c1b5;
      --muted: #8c8478;
      --accent: #ff5722;
      --line: #2a2724;
      --code-bg: #14110f;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.6;
      padding: 60px 24px;
      min-height: 100vh;
    }
    .container { max-width: 760px; margin: 0 auto; }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      background: var(--bg-2);
      border: 1px solid var(--line);
      border-radius: 4px;
      font-size: 12px;
      color: var(--accent);
      letter-spacing: 0.5px;
      text-transform: uppercase;
      margin-bottom: 20px;
    }
    h1 {
      font-size: 32px;
      font-weight: 700;
      margin-bottom: 12px;
      line-height: 1.2;
    }
    .subtitle {
      color: var(--ink-2);
      font-size: 17px;
      margin-bottom: 32px;
    }
    h2 {
      font-size: 18px;
      font-weight: 600;
      margin-top: 32px;
      margin-bottom: 12px;
      color: var(--ink);
    }
    p { color: var(--ink-2); margin-bottom: 12px; }
    .links { display: flex; gap: 12px; margin-bottom: 32px; flex-wrap: wrap; }
    .link {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 10px 16px;
      background: var(--bg-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink);
      text-decoration: none;
      font-size: 14px;
      font-weight: 500;
      transition: border-color 0.15s;
    }
    .link:hover { border-color: var(--accent); }
    .link.primary { background: var(--accent); border-color: var(--accent); color: white; }
    .link.primary:hover { background: #e64a19; border-color: #e64a19; }
    pre {
      background: var(--code-bg);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 16px;
      overflow-x: auto;
      font-family: 'SF Mono', Menlo, Consolas, monospace;
      font-size: 13px;
      color: var(--ink-2);
      margin-bottom: 16px;
    }
    code { color: var(--accent); }
    pre code { color: var(--ink-2); }
    ul { color: var(--ink-2); margin-left: 20px; margin-bottom: 16px; }
    li { margin-bottom: 6px; }
    .footer {
      margin-top: 48px;
      padding-top: 24px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }
    .footer a { color: var(--ink-2); text-decoration: none; border-bottom: 1px dotted var(--muted); }
  </style>
</head>
<body>
  <div class="container">
    <span class="badge">SHL Labs AI Intern · Take-home</span>
    <h1>SHL Conversational Assessment Recommender</h1>
    <p class="subtitle">
      A FastAPI service that takes a hiring manager from a vague brief to a grounded
      shortlist of SHL assessments through dialogue.
    </p>

    <div class="links">
      <a class="link primary" href="/docs">Open API Docs →</a>
      <a class="link" href="/health">Check Health</a>
      <a class="link" href="/redoc">ReDoc</a>
    </div>

    <h2>What it does</h2>
    <ul>
      <li><strong>Clarifies</strong> when the brief is too vague to act on.</li>
      <li><strong>Recommends</strong> 1–10 assessments from the SHL catalog once it has context.</li>
      <li><strong>Refines</strong> when the user changes constraints mid-conversation.</li>
      <li><strong>Compares</strong> specific assessments using catalog descriptions.</li>
      <li><strong>Stays in scope</strong> — refuses legal advice, off-topic queries, and prompt injection.</li>
    </ul>

    <h2>Try it from the command line</h2>
    <pre><code>curl -X POST $URL/chat \\
  -H 'Content-Type: application/json' \\
  -d '{"messages":[{"role":"user","content":"Hiring a senior Java engineer with Spring and SQL"}]}'</code></pre>

    <h2>Architecture in one sentence</h2>
    <p>
      An LLM router classifies each turn (clarify / recommend / refine / compare / refuse), then a
      dedicated handler runs hybrid retrieval (BM25 + dense MiniLM, fused with reciprocal rank fusion)
      over the 377-item SHL catalog, and a second LLM call picks the final shortlist from the top-20
      candidates — never allowed to invent items outside what retrieval surfaced.
    </p>

    <div class="footer">
      Built by <a href="https://github.com/Nihal108-bi" target="_blank">Nihal Jaiswal</a>
      &nbsp;·&nbsp; Stack: FastAPI + Groq (Llama 3.3 70B) + sentence-transformers + rank-bm25
    </div>
  </div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing() -> HTMLResponse:
    return HTMLResponse(content=_LANDING_HTML)


# ----- API endpoints --------------------------------------------------------


@app.get(
    "/health",
    response_model=Health,
    tags=["status"],
    summary="Health check",
    description="Returns `{\"status\": \"ok\"}` once the agent is ready to serve. "
                "The assignment evaluator polls this before grading.",
)
def health() -> Health:
    return Health(status="ok")


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Send a chat turn",
    description="Stateless turn handler. Pass the full conversation history as `messages` and "
                "receive the next assistant `reply`, optional `recommendations`, and an "
                "`end_of_conversation` flag.",
)
def chat(req: ChatRequest) -> ChatResponse:
    if _AGENT is None:
        # Should not happen — lifespan builds it before serving — but be defensive.
        raise HTTPException(status_code=503, detail="agent not ready")
    # The spec caps conversations at 8 turns. We don't enforce a 400 — we
    # just respond. The evaluator drives turn count from its side.
    try:
        return _AGENT.respond(req.messages)
    except Exception as e:
        log.exception("chat failed: %s", e)
        # Never break the schema. Return a clean response even on failure.
        return ChatResponse(
            reply="Something went wrong on my side. Could you rephrase the request?",
            recommendations=None,
            end_of_conversation=False,
        )