"""Catalog loader + code mapping.

The raw catalog JSON stores test categories under `keys` as full strings
("Knowledge & Skills", "Personality & Behavior"). The assignment example and
the conversation traces use single-letter codes ("K", "P"). This module owns
that mapping so the rest of the code stays clean.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Letter code -> human key. Drawn from the C1-C10 conversation traces.
KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",   # not seen in traces; chosen to avoid clash
}


@dataclass
class CatalogItem:
    entity_id: str
    name: str
    link: str
    description: str
    keys: List[str] = field(default_factory=list)
    job_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    duration_raw: str = ""
    duration: str = ""
    remote: str = ""
    adaptive: str = ""

    @property
    def test_type_code(self) -> str:
        """Comma-joined letter codes, e.g. 'K' or 'K,S' or 'P,C'."""
        codes = [KEY_TO_CODE[k] for k in self.keys if k in KEY_TO_CODE]
        return ",".join(codes) if codes else ""

    @property
    def display_duration(self) -> str:
        if self.duration_raw and self.duration_raw.strip():
            d = self.duration_raw.strip()
            return d if "minute" in d.lower() else f"{d} minutes"
        return "—"

    def to_search_doc(self) -> str:
        """Concatenated text used for BM25 + dense embedding.

        Name is repeated 3x because in a product catalog the name carries
        the strongest discriminating signal (it contains the tool/skill the
        test is for: "Docker", "MS Excel", "OPQ32r"), and we don't want
        common description words to drown it out in BM25 scoring.
        """
        parts = [
            self.name,
            self.name,
            self.name,
            self.description,
            "Categories: " + ", ".join(self.keys),
            "Levels: " + ", ".join(self.job_levels),
            f"Duration: {self.display_duration}",
        ]
        return "\n".join(p for p in parts if p)


def _data_path() -> Path:
    """Resolve the catalog JSON location.

    We support two layouts: `data/shl_product_catalog.json` (preferred) and
    the upload path used during development.
    """
    here = Path(__file__).resolve().parent.parent
    candidates = [
        here / "data" / "shl_product_catalog.json",
        Path("/mnt/user-data/uploads/shl_product_catalog.json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "shl_product_catalog.json not found. Place it under data/."
    )


def load_catalog() -> List[CatalogItem]:
    path = _data_path()
    with open(path, "r", encoding="utf-8") as f:
        # The provided JSON contains a stray control character; strict=False handles it.
        raw = json.load(f, strict=False)
    items: List[CatalogItem] = []
    for r in raw:
        if r.get("status") != "ok":
            continue
        items.append(
            CatalogItem(
                entity_id=str(r.get("entity_id", "")),
                name=r.get("name", "").strip(),
                link=r.get("link", "").strip(),
                description=(r.get("description") or "").strip(),
                keys=r.get("keys") or [],
                job_levels=r.get("job_levels") or [],
                languages=r.get("languages") or [],
                duration_raw=r.get("duration_raw") or "",
                duration=r.get("duration") or "",
                remote=r.get("remote") or "",
                adaptive=r.get("adaptive") or "",
            )
        )
    return items


def index_by_name(items: List[CatalogItem]) -> dict[str, CatalogItem]:
    """Lower-cased name -> item, for compare/lookup paths."""
    return {it.name.lower(): it for it in items}
