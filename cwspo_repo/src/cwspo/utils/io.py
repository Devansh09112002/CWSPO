from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: str | Path, model: Type[T] | None = None) -> list[T] | list[dict]:
    rows: list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(model.model_validate(obj) if model is not None else obj)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, BaseModel):
                f.write(row.model_dump_json() + "\n")
            else:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
