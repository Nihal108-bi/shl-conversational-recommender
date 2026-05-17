"""Offline replay against the 10 public conversation traces (C1-C10).

The assignment evaluator uses a similar mechanism on its side; this script
gives us a local proxy for Recall@10 so we can iterate before submission.

For each trace we:
  1. Extract the user turns and the FINAL ground-truth shortlist (last table
     in the trace).
  2. Replay the user turns through our /chat handler (the same Agent the
     server uses).
  3. Take the recommendations from our last response and compute
     Recall@10 against the ground-truth names.

Run:
    GROQ_API_KEY=gsk_... python -m tests.replay
    # or in offline mode (LLM stubbed; tests retrieval only):
    GROQ_API_KEY=stub python -m tests.replay
"""

from __future__ import annotations
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

from app.schemas import Message
from app.retriever import get_retriever
from app.llm import GroqClient
from app.agent import Agent


TRACE_SEARCH_DIRS = [
    Path("data/traces"),
    Path("/mnt/user-data/uploads"),
]
TRACE_FILES = [f"C{i}.md" for i in range(1, 11)]


def _find_trace(fname: str) -> Path | None:
    for d in TRACE_SEARCH_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None


# ---- trace parsing ----------------------------------------------------------


_USER_RE = re.compile(
    r"\*\*User\*\*\s*\n+>\s*(.+?)(?=\n\*\*Agent\*\*|\Z)",
    re.DOTALL,
)
_TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|", re.MULTILINE)


def parse_trace(path: Path) -> Tuple[List[str], List[str]]:
    """Return (user_turns, final_ground_truth_names)."""
    text = path.read_text(encoding="utf-8")

    # User turns: every blockquote after a "**User**" header.
    user_turns: List[str] = []
    for m in _USER_RE.finditer(text):
        # Collect contiguous "> " quote lines.
        block = m.group(1)
        # Strip leading ">" markers from continuation lines.
        cleaned = "\n".join(
            line.lstrip("> ").rstrip()
            for line in block.splitlines()
        ).strip()
        user_turns.append(cleaned)

    # Ground truth = the last markdown table's name column.
    tables = list(re.finditer(
        r"(\|\s*#\s*\|\s*Name\s*\|.*?)(?=\n\n|\Z)",
        text,
        re.DOTALL,
    ))
    ground_truth: List[str] = []
    if tables:
        last_table = tables[-1].group(1)
        for row in _TABLE_ROW_RE.finditer(last_table):
            name = row.group(1).strip()
            if name:
                ground_truth.append(name)
    return user_turns, ground_truth


# ---- replay -----------------------------------------------------------------


def replay_trace(agent: Agent, user_turns: List[str]):
    """Replay user turns sequentially.

    The grader's evaluator stops when the agent emits a shortlist. We mimic
    that by tracking the most recent non-empty recommendations across the
    conversation — that's what's scored on Recall@10.
    """
    history: List[Message] = []
    last_resp = None
    last_with_recs = None
    for utext in user_turns[:8]:  # honour the 8-turn cap
        history.append(Message(role="user", content=utext))
        resp = agent.respond(history)
        last_resp = resp
        if resp.recommendations:
            last_with_recs = resp
        history.append(Message(role="assistant", content=resp.reply))
    return last_with_recs or last_resp


# ---- metric -----------------------------------------------------------------


def _normalize(name: str) -> str:
    """Loose name-matching for evaluation: lower, collapse whitespace, drop punctuation."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def recall_at_k(predicted: List[str], gold: List[str], k: int = 10) -> float:
    if not gold:
        return 0.0
    pred_set = {_normalize(p) for p in predicted[:k]}
    gold_set = {_normalize(g) for g in gold}
    hits = sum(1 for g in gold_set if g in pred_set)
    return hits / len(gold_set)


# ---- main -------------------------------------------------------------------


def main() -> int:
    print("Loading retriever…")
    retriever = get_retriever()
    llm = GroqClient()
    agent = Agent(retriever=retriever, llm=llm)
    print(f"Retriever: {len(retriever.items)} items. LLM stub mode: {llm._stub}\n")

    recalls = []
    for fname in TRACE_FILES:
        path = _find_trace(fname)
        if path is None:
            print(f"SKIP {fname}: not found in {TRACE_SEARCH_DIRS}")
            continue

        user_turns, gold = parse_trace(path)
        t0 = time.time()
        resp = replay_trace(agent, user_turns)
        elapsed = time.time() - t0

        predicted = [r.name for r in (resp.recommendations or [])]
        r10 = recall_at_k(predicted, gold, k=10)
        recalls.append(r10)

        print(f"=== {fname} ===  ({elapsed:.1f}s, {len(user_turns)} user turns)")
        print(f"  gold ({len(gold)}): {gold}")
        print(f"  pred ({len(predicted)}): {predicted}")
        print(f"  Recall@10 = {r10:.2f}\n")

    if recalls:
        mean = sum(recalls) / len(recalls)
        print(f"Mean Recall@10 over {len(recalls)} traces: {mean:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
