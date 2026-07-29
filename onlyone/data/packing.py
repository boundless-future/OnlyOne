"""Sequence packing for SFT (ONLY for SFT — see design doc §2).

Concatenates encoded samples into fixed-length blocks so padding waste drops
to ~zero. Labels keep their -100 prompt masks, so each packed block is a valid
completion-only training sequence.

Caveat (accepted trade-off, same as TRL's packing): samples in a block attend
to each other unless the model is given block-diagonal attention. With causal
masking the contamination is one-directional and empirically harmless for SFT.
"""

from __future__ import annotations

from typing import Any, Iterator


def pack_examples(examples: Iterator[dict[str, Any]], max_len: int) -> Iterator[dict[str, Any]]:
    """Greedily fill max_len blocks with encoded examples."""
    buf_ids: list[int] = []
    buf_labels: list[int] = []

    def flush() -> dict[str, Any]:
        nonlocal buf_ids, buf_labels
        block = {
            "input_ids": buf_ids,
            "attention_mask": [1] * len(buf_ids),
            "labels": buf_labels,
        }
        buf_ids, buf_labels = [], []
        return block

    for ex in examples:
        ids, labels = ex["input_ids"], ex["labels"]
        if len(ids) > max_len:  # already truncated upstream, defensive
            ids, labels = ids[:max_len], labels[:max_len]
        if len(buf_ids) + len(ids) > max_len:
            yield flush()
        buf_ids.extend(ids)
        buf_labels.extend(labels)
    if buf_ids:
        yield flush()


class PackedSFTDataset:
    """Eager wrapper: pack a full SFTDataset into fixed-length blocks."""

    def __init__(self, dataset, max_len: int):
        self.blocks = list(pack_examples(
            (dataset[i] for i in range(len(dataset))), max_len
        ))

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.blocks[idx]
