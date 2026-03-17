from __future__ import annotations

import re


def split_steps(reasoning_text: str) -> list[str]:
    text = reasoning_text.strip()
    parts = re.split(r"(?:^|\n)(?:step\s*\d+[:.)-]?|\d+[.)-])\s*", text, flags=re.I)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts

    parts = [p.strip() for p in text.split("\n") if p.strip()]
    if len(parts) >= 2:
        return parts

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def canon(step: str) -> str:
    s = step.lower().strip()
    s = re.sub(r"^step\s*\d+[:.)-]?\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\.\+\-\*\/=]", "", s)
    return s
