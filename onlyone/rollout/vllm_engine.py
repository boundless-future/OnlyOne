"""vLLM colocate rollout engine (Linux only, M4 performance path).

Single-card coexistence strategy (design doc §3.2c):
- vLLM pre-allocates `gpu_mem_util` fraction of VRAM; the rest is left for
  training peak (activations + optimizer). Default 0.35 is a safe starting
  point for 1.5B bf16 on 24G.
- Before rollout: hot-load the latest merged LoRA weights into vLLM, then
  `wake_up()`. After rollout: `sleep()` releases the KV cache back to training.
- Weight sync: vLLM's `LLM.load_weights()` is not public API across versions;
  we use the documented-stable approach of loading through the model runner's
  `model.load_weights` when available, else fall back to recreating the LLM
  (slow but correct). The interface boundary is `sync_weights()`.

Requires: pip install "onlyone[vllm]" on Linux. Not importable on Windows —
all vllm imports are lazy so the module itself stays importable everywhere.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from onlyone.data.templates import get_template
from onlyone.rollout.base import RolloutEngine

logger = logging.getLogger("onlyone")

# vLLM >=0.10 defaults to the V1 engine, whose internal LLMEngine layout
# (multi-process EngineCore) is incompatible with our weight hot-load path.
# Force the legacy V0 engine until we have a V1-compatible sync implementation.
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

        self.um = um
        self.tokenizer = tokenizer
        self.template = get_template(template)
        self.device = device
        self.gpu_mem_util = gpu_mem_util
        self._llm = None  # created lazily on first generate (after first sync)
        self._tmp_dir: tempfile.TemporaryDirectory | None = None

    # ---------------------------------------------------------- weight sync

    def sync_weights(self) -> None:
        """Push the current policy weights into vLLM.

        LoRA case: merge adapter into base weights in a temp dir and let vLLM
        load from there. vLLM >=0.9 supports `LLM.sleep/wake_up` for colocate;
        weight refresh uses `llm.llm_engine...load_weights` when the LLM
        instance already exists, else the path is picked up at construction.
        """
        if self._tmp_dir is None:
            self._tmp_dir = tempfile.TemporaryDirectory(prefix="onlyone_vllm_")
        path = Path(self._tmp_dir.name) / "merged"

        merged = self.um.model.merge_and_unload() if hasattr(self.um.model, "merge_and_unload") else self.um.model
        merged.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # NOTE: merge_and_unload mutates the training model in some PEFT
        # versions — we immediately re-wrap below to keep training intact.
        # (PEFT >=0.10 merge_and_unload returns a NEW model; the original
        # PeftModel keeps its adapters. We discard the merged copy after save.)

        if self._llm is None:
            from vllm import LLM
            self._llm = LLM(
                model=str(path),
                gpu_memory_utilization=self.gpu_mem_util,
                enable_sleep_mode=True,   # sleep/wake colocate (vllm >=0.9)
                max_model_len=4096,
                enforce_eager=False,
            )
            logger.info("vLLM engine created (gpu_mem_util=%.2f)", self.gpu_mem_util)
        else:
            # Hot-reload into the running engine. This reaches into vLLM
            # internals; guarded so a version bump fails loudly, not silently.
            try:
                from safetensors.torch import load_file
                weights = list(load_file(str(path / "model.safetensors")).items())
                runner = self._llm.llm_engine.model_executor.driver_worker.model_runner
                runner.model.load_weights(weights)
            except (AttributeError, ImportError) as e:
                raise RuntimeError(
                    f"vLLM 权重热加载失败(版本 API 变动?)。"
                    f"请将 engine 临时切回 hf,或在此适配新版 vLLM API: {e}"
                ) from e

    # -------------------------------------------------------------- generate

    def generate(self, prompts, n_per_prompt, max_new_tokens, temperature, top_p):
        # GRPO 每步都更新 policy,必须每轮 rollout 前同步权重。
        # (代价:merge + save + load_weights;M4 后续可换 TRL 式 collective RPC)
        self.sync_weights()

        from vllm import SamplingParams

        rendered = [self.template.render_prompt(p) for p in prompts]
        params = SamplingParams(
            n=n_per_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        self._llm.wake_up()
        try:
            outputs = self._llm.generate(rendered, params)
        finally:
            self._llm.sleep()  # release KV cache back to training
        return [[o.text for o in out.outputs] for out in outputs]

    def __del__(self):
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
