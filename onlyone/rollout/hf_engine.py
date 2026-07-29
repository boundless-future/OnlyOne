"""HuggingFace generate()-based rollout engine.

The always-available fallback (design doc §2.5): no vLLM dependency, works
everywhere transformers works. Slower than vLLM but numerically identical in
distribution, so the whole RAFT/GRPO stack is validated against this engine
first.
"""

from __future__ import annotations

import logging

import torch

from onlyone.data.templates import get_template
from onlyone.rollout.base import RolloutEngine

logger = logging.getLogger("onlyone")


class HFRolloutEngine(RolloutEngine):
    def __init__(self, um, tokenizer, template: str = "chatml",
                 device: str = "cpu", batch_size: int = 8):
        self.um = um
        self.tokenizer = tokenizer
        self.template = get_template(template)
        self.device = device
        self.batch_size = batch_size

    def generate(
        self,
        prompts: list[str],
        n_per_prompt: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[str]]:
        # Expand: each prompt appears n_per_prompt times consecutively, so the
        # output regroups into per-prompt candidate lists by simple slicing.
        expanded = [p for p in prompts for _ in range(n_per_prompt)]
        rendered = [self.template.render_prompt(p) for p in expanded]

        results: list[str] = []
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"  # generation convention
        try:
            with self.um.as_generator() as model:
                for i in range(0, len(rendered), self.batch_size):
                    chunk = rendered[i : i + self.batch_size]
                    enc = self.tokenizer(
                        chunk, return_tensors="pt", padding=True,
                        add_special_tokens=False,
                    ).to(self.device)
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        top_p=top_p if temperature > 0 else None,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    prompt_len = enc["input_ids"].shape[1]
                    for row in out:
                        results.append(self.tokenizer.decode(
                            row[prompt_len:], skip_special_tokens=True
                        ))
        finally:
            self.tokenizer.padding_side = old_padding_side

        return [
            results[i * n_per_prompt : (i + 1) * n_per_prompt]
            for i in range(len(prompts))
        ]
