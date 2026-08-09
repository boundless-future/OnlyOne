"""vLLM colocate rollout engine (Linux only, M4 performance path).

Single-card coexistence strategy (design doc §3.2c):
- vLLM loads the BF16 BASE model ONCE with `enable_lora=True`; per-step weight
  sync saves only the LoRA adapter (~160MB for r=16) and hot-swaps it via
  `add_lora`/`remove_lora`. This replaces the old merge-and-reload approach,
  which cost 30-60s per step AND was unsound under QLoRA (merge_and_unload
  on 4bit shares modules with the training model — probe 2026-08-07).
- Sleep/wake ordering (vllm sleep-mode docs, huggingface/trl#5142):
  `wake_up() -> swap adapter -> generate() -> sleep()`. Mutating a SLEEPING
  engine touches offloaded/freed GPU memory and segfaults on wake.
- Requires `use_lora: true` — without an adapter there is nothing to swap;
  use `engine: hf` for full-parameter runs.

Requires: pip install "onlyone[vllm]" on Linux. Not importable on Windows —
all vllm imports are lazy so the module itself stays importable everywhere.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import tempfile
from pathlib import Path

import torch

from onlyone.data.templates import get_template
from onlyone.models.unified import DEFAULT_ADAPTER
from onlyone.rollout.base import RolloutEngine

logger = logging.getLogger("onlyone")

# vLLM >=0.10 defaults to the V1 engine; the LoRA hot-swap + sleep/wake path
# here is validated on the legacy V0 engine only (0.10.2, RTX 5090).
os.environ.setdefault("VLLM_USE_V1", "0")


class VLLMRolloutEngine(RolloutEngine):
    def __init__(self, um, tokenizer, template: str = "chatml",
                 device: str = "cuda", gpu_mem_util: float = 0.35):
        try:
            from vllm import LLM  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "vLLM 引擎需要 Linux + `pip install onlyone[vllm]`;"
                "Windows 请使用 engine: hf"
            ) from e
        if not getattr(um.cfg, "use_lora", False):
            raise RuntimeError(
                "vLLM 引擎依赖 LoRA adapter 热插拔,需要 use_lora: true;"
                "全量训练请使用 engine: hf"
            )

        self.um = um
        self.tokenizer = tokenizer
        self.template = get_template(template)
        self.device = device
        self.gpu_mem_util = gpu_mem_util
        self._llm = None  # created lazily on first sync
        self._lora_id = 0          # bumped every swap; vLLM requires unique ids
        self._active_lora = None   # LoRARequest for the current policy adapter
        self._tmp_dir: tempfile.TemporaryDirectory | None = None

    # ---------------------------------------------------------- weight sync

    def sync_weights(self) -> None:
        """Push the current policy adapter into vLLM.

        Saves ONLY the default adapter (never the frozen ref adapter) and
        hot-swaps it into the running engine under a fresh lora id. Each step
        writes a NEW adapter directory: overwriting the file vLLM may still
        have mmap'd is a use-after-write race.
        """
        if self._tmp_dir is None:
            self._tmp_dir = tempfile.TemporaryDirectory(prefix="onlyone_vllm_")
        path = Path(self._tmp_dir.name) / f"policy_adapter_{self._lora_id + 1}"
        self.um.model.save_pretrained(path, selected_adapters=[DEFAULT_ADAPTER])

        from vllm.lora.request import LoRARequest

        if self._llm is None:
            from vllm import LLM
            self._llm = LLM(
                model=str(self.um.cfg.name_or_path),
                enable_lora=True,
                max_lora_rank=self.um.cfg.lora_r,
                max_loras=2,               # old + new during a swap
                gpu_memory_utilization=self.gpu_mem_util,
                enable_sleep_mode=True,    # sleep/wake colocate (vllm >=0.9)
                max_model_len=4096,
                enforce_eager=True,        # LoRA+cudagraph 未验证,先 eager 求稳
            )
            logger.info("vLLM engine created (gpu_mem_util=%.2f, lora r=%d)",
                        self.gpu_mem_util, self.um.cfg.lora_r)
        else:
            # Release PyTorch's free-but-reserved blocks before waking: after a
            # training step the caching allocator holds most of the training
            # peak, and vLLM's cuMem wake needs physically free GPU memory
            # (CUDA OOM in cumem_allocator otherwise — smoke 2026-08-07).
            gc.collect()
            torch.cuda.empty_cache()
            # Wake BEFORE swapping — the engine slept after the last rollout,
            # and mutating a sleeping engine is UB (segfault on wake).
            self._llm.wake_up()

        old_id = self._lora_id
        self._lora_id += 1
        request = LoRARequest("policy", self._lora_id, str(path))
        self._llm.llm_engine.add_lora(request)
        if old_id > 0:
            self._llm.llm_engine.remove_lora(old_id)
        self._active_lora = request
        # Drop adapter dirs from 2+ swaps ago (keep current and previous —
        # previous may still be referenced by vLLM internals). Bounds /tmp
        # growth to ~2 adapters (~320MB) over a long run.
        stale = Path(self._tmp_dir.name) / f"policy_adapter_{self._lora_id - 2}"
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)

    # -------------------------------------------------------------- generate

    def generate(self, prompts, n_per_prompt, max_new_tokens, temperature, top_p):
        # GRPO 每步都更新 policy,必须每轮 rollout 前同步 adapter。
        self.sync_weights()

        from vllm import SamplingParams

        rendered = [self.template.render_prompt(p) for p in prompts]
        params = SamplingParams(
            n=n_per_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        try:
            outputs = self._llm.generate(rendered, params,
                                         lora_request=self._active_lora)
        finally:
            self._llm.sleep()  # release weights/KV back to training
        return [[o.text for o in out.outputs] for out in outputs]

    def __del__(self):
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
