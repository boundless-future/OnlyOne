# OnlyOne

> 一张卡，完成大模型对齐全流程：SFT / ORPO / KTO / DPO / SimPO / RAFT / GRPO。

面向个人开发者的单卡（24G~48G）强化对齐框架。核心思想：**显存中永远只有一份主干权重**——
Reference 用 PEFT 双 adapter 槽位零拷贝切换，Critic 由 GRPO 组内相对优势取代，Reward 优先走规则函数。

已经有 TRL 了，为什么还要有 OnlyOne：

- **单卡跑全算法梯队**——从 SFT 到 GRPO，不需要第二张卡、不需要多卡并行框架
- **trainer 全部自写**——每个算法一份可通读的实现，数值有测试钉死，适合想搞懂 RLHF 内部的人
- **显存工程是一等公民**——所有省显存手段（双 adapter、vLLM sleep/wake、logits 分块）都是真实烧卡烧出来的，附排障记录

设计文档：[docs/design.md](docs/design.md)

## 核心设计

**单主干 + 双 adapter 槽位。** Reference 不是第二个模型，而是同一个主干上冻结的
PEFT adapter 槽位（`ref` 冻结 / `default` 训练），切换零拷贝；SimPO 这类
reference-free 算法则连 ref 前向都省掉。

**vLLM colocate：训练与推理在同一张卡上严格交替。** vLLM 常驻 bf16 基座并开启
sleep mode；每个训练步只保存 LoRA adapter（r=16 约 160MB），通过
`add_lora`/`remove_lora` 热插拔同步权重（秒级，对比 merge-and-reload 方案的
30~60s/步，300 步省下约 4 小时）：

```
┌─ 每个 GRPO step ───────────────────────────────────────────┐
│ 1. sync    存 policy adapter → wake_up() → 热插拔(秒级)   │
│ 2. rollout vLLM 批量生成 prompts × G 个候选                │
│ 3. sleep   vLLM 权重卸载到 CPU,释放 ~16G 显存给训练        │
│ 4. train   4bit QLoRA 前向/反向/优化器(此时 vLLM 零占用)   │
└────────────────────────────────────────────────────────────┘
```

**显存有界的全词表 logps。** 7B 模型 152k 词表的 fp32 logits 是 GRPO 显存大头
（与模型大小无关）：逐 chunk forward + `torch.utils.checkpoint` 重计算，
保留显存从 ~25G 压到接近零。

**数值正确性优先。** 93 个测试在 CPU 小模型上钉死每个算法的 loss/优势/KL 数值，
先证明对，再花 GPU 小时。

## 算法支持

| 算法 | 配置 | 状态 |
|---|---|---|
| SFT | [configs/sft.yaml](configs/sft.yaml) | ✅ |
| ORPO | [configs/orpo.yaml](configs/orpo.yaml) | ✅ |
| KTO | [configs/kto.yaml](configs/kto.yaml) | ✅ |
| DPO | [configs/dpo.yaml](configs/dpo.yaml) | ✅ |
| SimPO | `algo.name: simpo`（与 DPO 同族配置） | ✅ |
| RAFT 飞轮 | [configs/raft.yaml](configs/raft.yaml) | ✅ |
| GRPO | [configs/grpo.yaml](configs/grpo.yaml)(1.5B)/ [configs/grpo_7b_math.yaml](configs/grpo_7b_math.yaml)(7B) | ✅ 真机验证 |

## 能力矩阵（24G 单卡基准）

| 模型 | 精度/方式 | SFT | ORPO/KTO | DPO | GRPO (G=4) |
|---|---|---|---|---|---|
| 1.5B | bf16 全量 | ✅ | ✅ | ✅ | ✅ 舒适 |
| 3B | bf16 + LoRA | ✅ | ✅ | ✅ | ✅ |
| 7~8B | 4bit QLoRA | ✅ | ✅ | ✅ | ✅ 32G 实测（vLLM colocate) |
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
onlyone train-grpo --config configs/grpo.yaml   # 1.5B GRPO 冒烟
```

7B 实战（环境 → 数据配比 → 训练 → 监控 → 评估验收的完整手册）：
[docs/math-grpo-runbook.md](docs/math-grpo-runbook.md)

## 真机验收（RTX 5090 32G，2026-08)

Qwen2.5-7B-Instruct + QLoRA GRPO 数学推理（MATH+GSM8K 混合集）:

| 指标 | 起点 | 训练后 | Δ |
|---|---|---|---|
| MATH-500 pass@1 | 53.4% | 66.4% | **+13.0pt** |
| GSM8K test | 90.8% | 90.1% | −0.8pt |
| completion 长度 | 1117 chars | 696 chars | 0.62x（更短且更对） |

## 观测与工具

- 训练指标逐步落盘 `{output_dir}/metrics.jsonl`（与 wandb/tb 后端无关，零依赖）
- `scripts/check_training.py`：尾部指标表 + 六项健康判定 + ASCII/PNG 趋势图
- `scripts/eval_math.py`：固定题集验收（base vs LoRA 同 vLLM 实例对照，贪婪 pass@1)
- 断点续训 + 检查点轮转（`keep_last_n_checkpoints`)，宕机最多损失 `save_steps` 步

## 文档

- [docs/design.md](docs/design.md)：架构设计与算法选型决策
- [docs/math-grpo-runbook.md](docs/math-grpo-runbook.md)：7B 数学 GRPO 实战手册（含已知坑表）
- [docs/git-conventions.md](docs/git-conventions.md)：提交与 PR 规范

## 路线图

- [ ] M2/M3 真机验收（ORPO 偏好准确率、RAFT 飞轮 GSM8K 提升）
- [ ] 与 TRL 同配置训练曲线对照
- [ ] 14B QLoRA（48G 卡）

## License

[MIT](LICENSE)
