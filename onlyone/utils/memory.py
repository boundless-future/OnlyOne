"""VRAM monitoring helpers.

Trainers call `peak_vram_gb()` at logging steps so the capability matrix in
README is backed by measured numbers, not guesses.
"""

from __future__ import annotations

import torch


def current_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3


def peak_vram_gb(reset: bool = False) -> float:
    if not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return peak
