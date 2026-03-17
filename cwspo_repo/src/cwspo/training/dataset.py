from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cwspo.schemas import PairRecord


@dataclass
class EncodedPair:
    prefix_ids: list[int]
    pref_ids: list[int]
    disp_ids: list[int]
    weight: float


class PairDataset(Dataset):
    def __init__(self, rows: list[PairRecord], tokenizer: PreTrainedTokenizerBase, max_length: int = 2048):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)[: self.max_length]

    def __getitem__(self, idx: int) -> EncodedPair:
        row = self.rows[idx]
        prefix_text = row.prompt.strip() + "\n\n" + "\n".join(row.prefix_steps).strip()
        pref_text = "\n".join(row.preferred_steps).strip()
        disp_text = "\n".join(row.dispreferred_steps).strip()
        return EncodedPair(
            prefix_ids=self._encode_text(prefix_text),
            pref_ids=self._encode_text(pref_text),
            disp_ids=self._encode_text(disp_text),
            weight=float(row.weight),
        )


def _pad_2d(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def collate_pairs(batch: list[EncodedPair], pad_id: int) -> dict[str, torch.Tensor]:
    return {
        "prefix_ids": _pad_2d([b.prefix_ids for b in batch], pad_id),
        "pref_ids": _pad_2d([b.pref_ids for b in batch], pad_id),
        "disp_ids": _pad_2d([b.disp_ids for b in batch], pad_id),
        "weights": torch.tensor([b.weight for b in batch], dtype=torch.float32),
    }
