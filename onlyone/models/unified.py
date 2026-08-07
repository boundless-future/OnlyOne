"""UnifiedModel: one backbone, multiple roles.

The single most important memory decision in OnlyOne: the GPU holds exactly
ONE copy of the backbone. The reference policy is a frozen LoRA snapshot in a
second adapter slot ("ref"), switched with `set_adapter` — zero extra VRAM for
a full reference model.

Invariants enforced here:
- `snapshot_ref()` must be called BEFORE any optimizer step, otherwise the
  "reference" would silently become the partially-trained policy.
- While inside `as_reference()`, gradients are globally impossible for the
  model and the previous train/eval state is restored on exit.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel

from onlyone.models.loading import load_backbone
from onlyone.utils.config import ModelConfig

logger = logging.getLogger("onlyone")

DEFAULT_ADAPTER = "default"
REF_ADAPTER = "ref"


class UnifiedModel:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._init_from_backbone(load_backbone(cfg))

    @classmethod
    def from_model(cls, base: PreTrainedModel, cfg: ModelConfig) -> "UnifiedModel":
        """Wrap an already-instantiated backbone (used by tests with tiny
        randomly-initialized models, and by the TRL comparison script)."""
        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj._init_from_backbone(base)
        return obj

    def _init_from_backbone(self, base: PreTrainedModel) -> None:
        cfg = self.cfg
        self.model: PreTrainedModel | PeftModel
        if cfg.use_lora:
            lora_cfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base, lora_cfg, adapter_name=DEFAULT_ADAPTER)
        else:
            self.model = base
        self._ref_snapshotted = False
        self._trained_steps = 0

        trainable, total = self._count_params()
        logger.info(
            "model loaded: %s | trainable %.2fM / total %.2fM (%.2f%%)",
            cfg.name_or_path, trainable / 1e6, total / 1e6, 100 * trainable / max(total, 1),
        )

    # ------------------------------------------------------------------ roles

    def snapshot_ref(self) -> None:
        """Freeze the CURRENT adapter weights into the "ref" slot.

        Must be called before training starts. After this, the ref slot is
        detached from the graph and never updated.
        """
        if not self.cfg.use_lora:
            raise RuntimeError("snapshot_ref 需要 LoRA；全量训练时请单独加载冻结副本（v1 不支持）")
        if self._trained_steps > 0:
            raise RuntimeError(
                "snapshot_ref() 必须在任何训练 step 之前调用，"
                "否则 reference 会被污染成部分训练后的 policy"
            )
        peft_model: PeftModel = self.model  # type: ignore[assignment]
        if REF_ADAPTER not in peft_model.peft_config:
            ref_cfg = peft_model.peft_config[DEFAULT_ADAPTER]
            peft_model.add_adapter(REF_ADAPTER, ref_cfg)

        # Copy default -> ref. PEFT stores per-adapter weights in the same
        # module with the adapter name in the parameter key.
        state = peft_model.state_dict()
        ref_state = {
            k.replace(f".{DEFAULT_ADAPTER}.", f".{REF_ADAPTER}."): v.detach().clone()
            for k, v in state.items()
            if f".{DEFAULT_ADAPTER}." in k and "lora_" in k
        }
        missing, unexpected = peft_model.load_state_dict(ref_state, strict=False)
        if unexpected:
            raise RuntimeError(f"ref 快照写入了意外参数: {unexpected[:5]}")
        del missing  # base weights are shared, expected to be "missing"

        for name, p in peft_model.named_parameters():
            if f".{REF_ADAPTER}." in name:
                p.requires_grad_(False)

        peft_model.set_adapter(DEFAULT_ADAPTER)
        self._ref_snapshotted = True
        logger.info("reference snapshot frozen into adapter slot '%s'", REF_ADAPTER)

    @contextlib.contextmanager
    def as_reference(self):
        """Yield the model as the frozen reference policy (no grad, eval)."""
        if not self._ref_snapshotted:
            raise RuntimeError("as_reference() 之前必须先 snapshot_ref()")
        peft_model: PeftModel = self.model  # type: ignore[assignment]
        was_training = self.model.training
        peft_model.set_adapter(REF_ADAPTER)
        self.model.eval()
        with torch.no_grad():
            yield self.model
        peft_model.set_adapter(DEFAULT_ADAPTER)
        if was_training:
            self.model.train()

    @contextlib.contextmanager
    def as_generator(self):
        """Yield the model ready for rollout generation.

        Training keeps use_cache=False (incompatible with gradient
        checkpointing); generation needs the KV cache. This context flips
        use_cache on, switches to eval + no_grad, and restores everything on
        exit so the training loop is unaffected.
        """
        was_training = self.model.training
        old_use_cache = self.model.config.use_cache
        self.model.eval()
        self.model.config.use_cache = True
        with torch.no_grad():
            yield self.model
        self.model.config.use_cache = old_use_cache
        if was_training:
            self.model.train()

    # ------------------------------------------------------------------ math

    def token_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        logps_chunk_size: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked per-token log probs and per-sequence completion counts.

        Returns (token_logps (B, T-1), counts (B,)). Masked positions (prompt,
        padding) are exactly 0 in the output. This is the single forward pass
        that all losses are derived from — SFT uses the sum, ORPO uses sum
        AND per-token average, GRPO will use the full matrix.

        The forward is chunked over the batch dimension: full-vocab logits for
        B sequences at fp32 cost B×T×V×4 bytes (7B/152k vocab: ~12G at B=32,
        which OOMs a 32G card). Chunking keeps numerics identical while
        bounding the peak to chunk_size×T×V×4 (~1.5G at chunk 4).
        """
        chunks: list[torch.Tensor] = []
        counts: list[torch.Tensor] = []
        for start in range(0, input_ids.shape[0], logps_chunk_size):
            sl = slice(start, start + logps_chunk_size)
            out = self.model(input_ids=input_ids[sl], attention_mask=attention_mask[sl])
            logits = out.logits[:, :-1, :]
            shifted_labels = labels[sl, 1:]
            mask = shifted_labels != -100

            logp_per_token = torch.log_softmax(logits.float(), dim=-1)
            # Replace -100 labels by 0 before gather, but also zero out the gathered
            # value for masked positions afterwards so arbitrary garbage at masked
            # indices cannot leak into the sum.
            safe_labels = shifted_labels.clamp(min=0)
            token_logps = torch.gather(
                logp_per_token, dim=-1, index=safe_labels.unsqueeze(-1)
            ).squeeze(-1)
            chunks.append(token_logps * mask)
            counts.append(mask.sum(dim=-1))
        return torch.cat(chunks), torch.cat(counts)

    def logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        average_per_token: bool = False,
    ) -> torch.Tensor:
        """Per-sequence summed log p(completion tokens | prompt).

        `labels` uses -100 for positions that must be ignored (prompt tokens,
        padding). Logits are shifted internally: position t predicts token t+1.

        Returns shape (batch,). If `average_per_token`, divide by the number
        of completion tokens (SimPO-style length normalization).
        """
        token_logps, counts = self.token_logps(input_ids, attention_mask, labels)
        seq_logps = token_logps.sum(dim=-1)
        if average_per_token:
            seq_logps = seq_logps / counts.clamp(min=1)
        return seq_logps

    def mark_trained(self, steps: int) -> None:
        self._trained_steps += steps

    # ------------------------------------------------------------------ misc

    def _count_params(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def save(self, path: str) -> None:
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path, selected_adapters=[DEFAULT_ADAPTER])
        else:
            self.model.save_pretrained(path)

    def __getattr__(self, name: str):
        # Delegate unknown attributes (generate, config, device, ...) to the
        # wrapped model so callers can treat UnifiedModel as the model itself.
        if name in ("model", "cfg", "_ref_snapshotted", "_trained_steps"):
            raise AttributeError(name)
        return getattr(self.model, name)
