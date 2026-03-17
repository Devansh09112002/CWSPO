from __future__ import annotations

import re


def extract_final_answer(text: str) -> str:
    # Prefer boxed answers, then 'answer is', then final number.
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    ans = re.findall(r"answer\s*(?:is|=)\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.I)
    if ans:
        return ans[-1].strip()

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1].strip()

    return text.strip().splitlines()[-1].strip()


def normalize_answer(ans: str) -> str:
    s = ans.strip().lower()
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s)
    m = re.fullmatch(r"[-+]?\d+(?:\.0+)?", s)
    if m:
        s = s.rstrip("0").rstrip(".") if "." in s else s
    return s


def is_correct_answer(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)
