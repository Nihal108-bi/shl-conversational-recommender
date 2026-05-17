"""The agent: routes each turn to one of {clarify, recommend, refine, compare,
refuse} and produces the ChatResponse.

Stateless by design. The full conversation history comes in on every call,
matching the spec. No per-conversation state on the server.
"""

from __future__ import annotations
import json
import re
from typing import List, Dict

from .schemas import Message, ChatResponse, Recommendation
from .catalog import CatalogItem, index_by_name
from .retriever import HybridRetriever
from .llm import GroqClient
from . import prompts


# ---------- helpers ----------------------------------------------------------


def _user_turns(messages: List[Message]) -> List[Message]:
    return [m for m in messages if m.role == "user"]


def _last_user(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content.strip()
    return ""


def _conversation_text(messages: List[Message]) -> str:
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)


def _previous_shortlist(
    messages: List[Message], by_name: Dict[str, CatalogItem]
) -> List[CatalogItem]:
    """Walk back through assistant turns and find the most recent shortlist.

    Items recommended earlier appear by name in the assistant's prose; we
    detect them by substring match against the canonical catalog name.
    """
    for m in reversed(messages):
        if m.role != "assistant":
            continue
        text_lower = m.content.lower()
        found: List[CatalogItem] = []
        seen_ids = set()
        # Longest names first so "Microsoft Excel 365 - Essentials" wins
        # over "Microsoft Excel" when both substrings are present.
        for name_l in sorted(by_name.keys(), key=len, reverse=True):
            if name_l in text_lower:
                item = by_name[name_l]
                if item.entity_id not in seen_ids:
                    found.append(item)
                    seen_ids.add(item.entity_id)
        if found:
            return found
    return []


def _items_to_recs(items: List[CatalogItem]) -> List[Recommendation]:
    return [
        Recommendation(name=it.name, url=it.link, test_type=it.test_type_code)
        for it in items
    ]


def _candidates_block(cands: List[CatalogItem], limit: int = 20) -> str:
    """Compact block the selector LLM can read quickly."""
    lines = []
    for i, it in enumerate(cands[:limit], 1):
        desc = (it.description or "").replace("\n", " ").strip()
        if len(desc) > 280:
            desc = desc[:280] + "…"
        lines.append(
            f"[{i}] {it.name}\n"
            f"    keys: {', '.join(it.keys) or '—'}\n"
            f"    levels: {', '.join(it.job_levels) or '—'}\n"
            f"    duration: {it.display_duration}\n"
            f"    desc: {desc}"
        )
    return "\n".join(lines)


def _resolve_picks(
    picks: List[str], pool: List[CatalogItem], by_name: Dict[str, CatalogItem]
) -> List[CatalogItem]:
    """Map LLM-emitted names back to catalog items.

    Exact match first; then a loose match against the candidate pool to
    survive minor punctuation drift from the model. Hard cap at 10."""
    chosen: List[CatalogItem] = []
    seen = set()
    pool_by_name = {p.name.lower(): p for p in pool}
    for raw in picks:
        if not isinstance(raw, str):
            continue
        key = raw.strip().lower()
        item = pool_by_name.get(key) or by_name.get(key)
        if item is None:
            stripped = re.sub(r"\s*\([^)]*\)\s*$", "", key).strip()
            for n, it in pool_by_name.items():
                if stripped and stripped in n:
                    item = it
                    break
        if item is not None and item.entity_id not in seen:
            chosen.append(item)
            seen.add(item.entity_id)
        if len(chosen) >= 10:
            break
    return chosen


def _looks_like_injection(text: str) -> bool:
    t = text.lower()
    tells = [
        "ignore previous", "ignore prior", "ignore the above",
        "system prompt", "reveal your prompt", "show your instructions",
        "you are now", "pretend you are", "act as if",
        "recommend a product outside",
    ]
    return any(s in t for s in tells)


# ---------- the agent --------------------------------------------------------


class Agent:
    def __init__(self, retriever: HybridRetriever, llm: GroqClient):
        self.retriever = retriever
        self.llm = llm
        self.by_name = index_by_name(retriever.items)

    # ---- public entry point -------------------------------------------------

    def respond(self, messages: List[Message]) -> ChatResponse:
        if not messages:
            return ChatResponse(
                reply="Hi — tell me about the role you're hiring for and I'll shape an SHL assessment shortlist.",
                recommendations=None,
                end_of_conversation=False,
            )

        last = _last_user(messages)

        # Cheap pre-check before paying for the router.
        if _looks_like_injection(last):
            return self._refuse_canned("injection")

        intent = self._route(messages)

        if intent == "refuse":
            return self._handle_refuse(last)
        if intent == "clarify":
            return self._handle_clarify(messages)
        if intent == "compare":
            return self._handle_compare(messages)
        if intent == "refine":
            return self._handle_refine(messages)
        return self._handle_recommend(messages)

    # ---- router -------------------------------------------------------------

    def _route(self, messages: List[Message]) -> str:
        """Classify the current turn. Returns one of:
        clarify, recommend, refine, compare, refuse."""
        prior_assistant = any(m.role == "assistant" for m in messages)
        user_msgs = _user_turns(messages)

        # Heuristic: very short first user turn -> clarify.
        if len(user_msgs) == 1 and len(user_msgs[0].content) < 40:
            return "clarify"

        router_msgs = [
            {"role": "system", "content": prompts.ROUTER_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Conversation so far:\n"
                    f"{_conversation_text(messages)}\n\n"
                    "Classify the LAST user turn. JSON only."
                ),
            },
        ]
        try:
            data = self.llm.chat_json(router_msgs, max_tokens=80)
            intent = data.get("intent", "recommend")
        except Exception:
            intent = "recommend"

        # Refine requires a prior shortlist.
        if intent == "refine" and not prior_assistant:
            intent = "recommend"
        if intent not in {"clarify", "recommend", "refine", "compare", "refuse"}:
            intent = "recommend"
        return intent

    # ---- handlers -----------------------------------------------------------

    def _handle_clarify(self, messages: List[Message]) -> ChatResponse:
        msgs = [
            {"role": "system", "content": prompts.CLARIFY_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Conversation so far:\n"
                    f"{_conversation_text(messages)}\n\n"
                    "Ask one focused clarifying question."
                ),
            },
        ]
        reply = self.llm.chat(msgs, max_tokens=150, temperature=0.3).strip()
        return ChatResponse(reply=reply, recommendations=None, end_of_conversation=False)

    def _handle_recommend(self, messages: List[Message]) -> ChatResponse:
        query = self._build_search_query(messages)
        candidates = [it for it, _ in self.retriever.search(query, k=20)]

        msgs = [
            {"role": "system", "content": prompts.SELECTOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Conversation:\n"
                    f"{_conversation_text(messages)}\n\n"
                    "CANDIDATES:\n"
                    f"{_candidates_block(candidates)}\n\n"
                    "Pick 1-10. JSON only."
                ),
            },
        ]
        try:
            data = self.llm.chat_json(msgs, max_tokens=600, temperature=0.1)
        except Exception:
            data = {"reply": "Here are matches from the catalog.", "picks": []}

        picks_raw = data.get("picks") or []
        chosen = _resolve_picks(picks_raw, candidates, self.by_name)
        # Safety net so we never emit an empty shortlist on a recommend turn.
        # Use top-10 — we're scored on Recall@10 and the harness caps at 10.
        if not chosen:
            chosen = candidates[:10]
        reply = (data.get("reply") or "Here are matches from the catalog.").strip()
        return ChatResponse(
            reply=reply,
            recommendations=_items_to_recs(chosen),
            end_of_conversation=False,
        )

    def _handle_refine(self, messages: List[Message]) -> ChatResponse:
        previous = _previous_shortlist(messages, self.by_name)
        # If we can't recover any prior shortlist from the conversation text,
        # the "refine" classification was over-confident. Fall through to a
        # fresh recommend on the full conversation history.
        if not previous:
            return self._handle_recommend(messages)

        edit_query = _last_user(messages)
        new_cands = [it for it, _ in self.retriever.search(edit_query, k=15)]
        prev_ids = {p.entity_id for p in previous}
        pool = previous + [c for c in new_cands if c.entity_id not in prev_ids]
        prev_names = [p.name for p in previous]

        msgs = [
            {"role": "system", "content": prompts.REFINE_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Conversation:\n"
                    f"{_conversation_text(messages)}\n\n"
                    f"PREVIOUS SHORTLIST: {json.dumps(prev_names)}\n\n"
                    "CANDIDATES (for any additions):\n"
                    f"{_candidates_block(new_cands, limit=15)}\n\n"
                    "Apply the edit. JSON only."
                ),
            },
        ]
        try:
            data = self.llm.chat_json(msgs, max_tokens=500, temperature=0.1)
        except Exception:
            data = {"reply": "Updated.", "picks": prev_names}

        picks_raw = data.get("picks") or prev_names
        chosen = _resolve_picks(picks_raw, pool, self.by_name)
        if not chosen:
            chosen = previous
        reply = (data.get("reply") or "Updated.").strip()
        return ChatResponse(
            reply=reply,
            recommendations=_items_to_recs(chosen),
            end_of_conversation=False,
        )

    def _handle_compare(self, messages: List[Message]) -> ChatResponse:
        last = _last_user(messages)
        items = self._items_named_in(last)
        if len(items) < 2:
            previous = _previous_shortlist(messages, self.by_name)
            items = (items + previous)[:3]
        if len(items) < 2:
            return ChatResponse(
                reply="Which two assessments would you like compared? Name them and I'll pull the details from the catalog.",
                recommendations=None,
                end_of_conversation=False,
            )

        entries_block = "\n\n".join(
            f"NAME: {it.name}\n"
            f"KEYS: {', '.join(it.keys)}\n"
            f"DURATION: {it.display_duration}\n"
            f"LEVELS: {', '.join(it.job_levels)}\n"
            f"DESCRIPTION: {it.description}"
            for it in items[:3]
        )
        msgs = [
            {"role": "system", "content": prompts.COMPARE_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Conversation:\n"
                    f"{_conversation_text(messages)}\n\n"
                    "CATALOG ENTRIES:\n"
                    f"{entries_block}\n\n"
                    "Write the comparison."
                ),
            },
        ]
        reply = self.llm.chat(msgs, max_tokens=400, temperature=0.2).strip()
        return ChatResponse(
            reply=reply,
            recommendations=None,
            end_of_conversation=False,
        )

    def _handle_refuse(self, last: str) -> ChatResponse:
        msgs = [
            {"role": "system", "content": prompts.REFUSE_CLASSIFY_SYSTEM},
            {"role": "user", "content": last},
        ]
        try:
            data = self.llm.chat_json(msgs, max_tokens=40)
            cat = data.get("category", "off_topic")
        except Exception:
            cat = "off_topic"
        if cat not in prompts.REFUSE_REPLIES:
            cat = "off_topic"
        return ChatResponse(
            reply=prompts.REFUSE_REPLIES[cat],
            recommendations=None,
            end_of_conversation=False,
        )

    def _refuse_canned(self, category: str) -> ChatResponse:
        return ChatResponse(
            reply=prompts.REFUSE_REPLIES.get(category, prompts.REFUSE_REPLIES["off_topic"]),
            recommendations=None,
            end_of_conversation=False,
        )

    # ---- query construction & item lookup ----------------------------------

    def _build_search_query(self, messages: List[Message]) -> str:
        """Concatenate the most informative user turns into a retrieval query."""
        user_msgs = _user_turns(messages)
        if not user_msgs:
            return ""
        recent = user_msgs[-4:]
        return "  ".join(m.content for m in recent)

    def _items_named_in(self, text: str) -> List[CatalogItem]:
        """Find catalog items whose canonical name appears in the text.
        Longest match first so subsuming names win."""
        text_l = text.lower()
        scored: List[tuple[int, CatalogItem]] = []
        for name_l, item in self.by_name.items():
            if name_l in text_l:
                scored.append((len(name_l), item))
        scored.sort(key=lambda x: -x[0])
        chosen: List[CatalogItem] = []
        seen_ids = set()
        for _, item in scored:
            if item.entity_id in seen_ids:
                continue
            chosen.append(item)
            seen_ids.add(item.entity_id)
        return chosen
