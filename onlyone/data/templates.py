"""Prompt templates.

Each template turns (prompt, completion) into (prompt_text, full_text).
The trainer tokenizes `prompt_text` and `full_text` separately and derives the
completion mask from the length difference — so templates MUST guarantee that
`full_text` starts with exactly `prompt_text`.

NOTE: modern chat models ship an official `tokenizer.apply_chat_template`.
These minimal templates exist for tiny/base models in tests and as a fallback;
prefer the tokenizer's own template for real models (template="tokenizer").
"""

from __future__ import annotations

from typing import Protocol


class Template(Protocol):
    def render_prompt(self, prompt: str) -> str: ...
    def render_full(self, prompt: str, completion: str) -> str: ...


class ChatMLTemplate:
    def render_prompt(self, prompt: str) -> str:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def render_full(self, prompt: str, completion: str) -> str:
        return self.render_prompt(prompt) + completion + "<|im_end|>"


class AlpacaTemplate:
    def render_prompt(self, prompt: str) -> str:
        return f"### Instruction:\n{prompt}\n\n### Response:\n"

    def render_full(self, prompt: str, completion: str) -> str:
        return self.render_prompt(prompt) + completion


class RawTemplate:
    """No formatting: prompt and completion are concatenated directly."""

    def render_prompt(self, prompt: str) -> str:
        return prompt

    def render_full(self, prompt: str, completion: str) -> str:
        return prompt + completion


_REGISTRY: dict[str, type[Template]] = {
    "chatml": ChatMLTemplate,
    "alpaca": AlpacaTemplate,
    "raw": RawTemplate,
}


def get_template(name: str) -> Template:
    if name not in _REGISTRY:
        raise KeyError(f"未知模版 '{name}'，可选: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
