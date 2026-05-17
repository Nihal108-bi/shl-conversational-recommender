"""Tiny wrapper around the Groq SDK.

Adds:
  - retries with exponential backoff
  - JSON-mode for the router
  - an offline stub for tests (set GROQ_API_KEY=stub)
"""

from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional


GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


class GroqClient:
    def __init__(self, api_key: Optional[str] = None, model: str = GROQ_MODEL):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self._stub = self.api_key == "stub" or not self.api_key
        if not self._stub:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)

    # ----- core chat ------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> str:
        if self._stub:
            return self._stub_response(messages, json_mode=json_mode)

        last_err = None
        for attempt in range(3):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # network, rate limit, etc.
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"Groq call failed after retries: {last_err}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> Dict[str, Any]:
        raw = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # The model sometimes wraps JSON in ```json``` — strip and retry parse.
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)

    # ----- offline stub for unit tests ------------------------------------

    def _stub_response(self, messages, json_mode: bool) -> str:
        """Predictable canned replies so unit tests don't need network."""
        last = messages[-1]["content"].lower() if messages else ""
        if json_mode:
            # Router stub: classify by keyword.
            if "compare" in last or " vs " in last or "difference" in last:
                return json.dumps({"intent": "compare", "items": []})
            if any(w in last for w in ["drop", "remove", "replace", "add ", "swap"]):
                return json.dumps({"intent": "refine"})
            if len(last) < 60 and "?" not in last:
                return json.dumps({"intent": "clarify"})
            if any(w in last for w in ["legal", "law", "sue"]):
                return json.dumps({"intent": "refuse"})
            return json.dumps({"intent": "recommend"})
        return "Stub reply."
