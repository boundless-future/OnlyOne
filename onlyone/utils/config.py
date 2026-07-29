"""Pydantic-based configuration: YAML -> validated dataclass-like models.

Design rules:
- Every trainer has its own config section; shared sections (model/data/train)
  are composed rather than duplicated.
- Unknown YAML keys are rejected (extra="forbid") so typos fail fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Base):
    """Backbone loading and LoRA setup."""

    name_or_path: str
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "sdpa"
    load_in_4bit: bool = False  # QLoRA path (requires bitsandbytes, Linux)
    trust_remote_code: bool = False

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


class DataConfig(_Base):
    """Dataset source and preprocessing."""

    train_path: str  # jsonl
    eval_path: Optional[str] = None
    template: str = "chatml"  # see onlyone.data.templates
    max_len: int = 2048
    # SFT-only: concatenate multiple samples into one max_len sequence.
    # Must stay False for preference/RL trainers (pairs cannot be packed).
    packing: bool = False
    num_workers: int = 4


class TrainConfig(_Base):
    """Optimization loop, shared by all trainers."""

    output_dir: str = "runs/sft"
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_steps: int = 1000
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42

    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: Optional[int] = None
    resume_from: Optional[str] = None

    log_with: Literal["none", "wandb", "tensorboard"] = "none"
    run_name: Optional[str] = None
    # "auto" = cuda if available else cpu; set explicitly in tests / debug.
    device: str = "auto"


class AlgoConfig(_Base):
    """Algorithm selection and hyperparameters.

    Flat fields with per-algo prefixes keep YAML simple; unused fields for the
    selected algorithm are ignored. Hyperparameter meanings:
    - orpo_lambda: weight of the odds-ratio preference term (paper default 0.1)
    - kto_beta: KL-to-reference temperature (paper default 0.1)
    - dpo_beta: KL regularization strength (paper default 0.1)
    - simpo_beta / simpo_gamma: logp-difference scale / target reward margin
      (paper defaults β=2.0, γ=0.5)
    """

    name: Literal["sft", "orpo", "kto", "dpo", "simpo", "grpo"] = "sft"
    orpo_lambda: float = 0.1
    kto_beta: float = 0.1
    kto_desirable_weight: float = 1.0
    kto_undesirable_weight: float = 1.0
    dpo_beta: float = 0.1
    simpo_beta: float = 2.0
    simpo_gamma: float = 0.5


class RaftConfig(_Base):
    """RAFT data-flywheel settings (rollout -> filter -> retrain loop)."""

    prompts_path: str           # jsonl: {"prompt": ..., "meta": {...}}
    group_size: int = 4         # candidates sampled per prompt
    rewards: list[str] = Field(default_factory=lambda: ["math_answer"])
    threshold: float = 0.5      # min summed reward for a sample to be kept
    rounds: int = 2
    max_new_tokens: int = 256
    temperature: float = 0.8    # >0: flywheel needs diverse candidates
    top_p: float = 0.95
    rollout_batch_size: int = 8
    output_dir: str = "runs/raft"
    train_sft: bool = True      # retrain on filtered data each round
    make_preference: bool = True  # also emit best/worst DPO pairs


class GRPOConfig(_Base):
    """GRPO online-RL settings (design doc §3.2b).

    Pipeline per iteration: rollout G candidates per prompt -> rule rewards ->
    group-relative advantage -> clipped surrogate + k3 KL against the frozen
    reference adapter. No Critic, no reward model — see design doc §2.2/§2.3.
    """

    prompts_path: str           # jsonl: {"prompt": ..., "meta": {...}}
    group_size: int = 4         # G candidates per prompt
    prompts_per_step: int = 8   # distinct prompts per iteration (batch = ×G)
    rewards: list[str] = Field(default_factory=lambda: ["math_answer"])

    clip_eps: float = 0.2       # PPO-style surrogate clip range
    kl_beta: float = 0.04       # weight of the k3 KL-to-reference penalty
    # Circuit breaker (design doc §5): abort if a step's mean KL exceeds this.
    # 0 disables. Guards against silent policy collapse away from reference.
    kl_circuit_breaker: float = 0.0

    max_new_tokens: int = 256
    temperature: float = 0.9    # >0: candidates within a group must differ
    top_p: float = 0.95
    rollout_batch_size: int = 8
    engine: Literal["hf", "vllm"] = "hf"  # vllm requires Linux + onlyone[vllm]
    vllm_gpu_mem_util: float = 0.35       # leave headroom for training peak


class TrainJobConfig(_Base):
    """Root config for any training job; `algo.name` selects the trainer."""

    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    algo: AlgoConfig = Field(default_factory=AlgoConfig)
    raft: Optional[RaftConfig] = None
    grpo: Optional[GRPOConfig] = None

    @model_validator(mode="after")
    def _check_combinations(self):
        if self.model.load_in_4bit and not self.model.use_lora:
            raise ValueError("4bit 量化只能配合 LoRA 训练（QLoRA）")
        if self.data.packing and self.algo.name != "sft":
            raise ValueError("packing 仅适用于 SFT；偏好/二元数据必须保持样本对齐")
        return self


# Backwards-compatible alias: SFT jobs are TrainJobConfig with algo.name="sft".
SFTConfig = TrainJobConfig


def load_config(path: str | Path) -> TrainJobConfig:
    """Load and validate a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrainJobConfig.model_validate(raw)
