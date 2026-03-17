from __future__ import annotations

import re


_STEP_MARKER_RE = r"(?:step\s*\d+[:.)-]?|\d+[.)-])"


def _clean_parts(parts: list[str]) -> list[str]:
    cleaned = []
    for part in parts:
        text = re.sub(rf"^\s*{_STEP_MARKER_RE}\s*", "", part, flags=re.I).strip()
        if text:
            cleaned.append(text)
    return cleaned


def split_steps(reasoning_text: str) -> list[str]:
    text = reasoning_text.strip()
    if not text:
        return []

    parts = re.split(rf"(?im)^\s*{_STEP_MARKER_RE}\s*", text)
    parts = _clean_parts(parts)
    if len(parts) >= 2:
        return parts

    parts = re.split(r"(?i)\s+(?=step\s*\d+[:.)-]?\s*)", text)
    parts = _clean_parts(parts)
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
