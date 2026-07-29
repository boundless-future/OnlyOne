# OnlyOne

> 一张卡，完成大模型对齐全流程：SFT / ORPO / KTO / DPO / SimPO / RAFT / GRPO。

面向个人开发者的单卡（24G~48G）强化对齐框架。核心思想：**显存中永远只有一份主干权重**——
Reference 用 PEFT 双 adapter 槽位零拷贝切换，Critic 由 GRPO 组内相对优势取代，Reward 优先走规则函数。

设计文档：[docs/design.md](docs/design.md)

## 能力矩阵（24G 单卡基准）

| 模型 | 精度/方式 | SFT | ORPO/KTO | DPO | GRPO (G=4) |
|---|---|---|---|---|---|
| 1.5B | bf16 全量 | ✅ | ✅ | ✅ | ✅ 舒适 |
| 3B | bf16 + LoRA | ✅ | ✅ | ✅ | ✅ |
| 7~8B | 4bit QLoRA | ✅ | ✅ | ✅ | ⚠️ 小 G + 短序列 |
| 14B | 4bit QLoRA | ⚠️ 仅 SFT | 勉强 | ❌ | ❌ |

## 安装

```bash
pip install -e ".[dev]"          # 基础（SFT/ORPO/DPO 可跑）
pip install -e ".[qlora,flash]"  # Linux + 7B 模型
pip install -e ".[vllm,logging]" # GRPO 加速 + wandb
```

## 快速开始

```bash
onlyone train --config configs/sft.yaml
```

## 当前状态

- [x] M1：基建 + SFT（双 adapter UnifiedModel、packing、completion-only loss、数值单测）
- [ ] M2：ORPO + KTO + GSM8K 评估
- [ ] M3：DPO/SimPO + RAFT 数据飞轮
- [ ] M4：GRPO + vLLM colocate
