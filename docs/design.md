# OnlyOne — 单卡大模型强化对齐框架 · 项目方案 v1.0

> **OnlyOne**：一张卡，完成 SFT / 偏好对齐 / 强化学习的全流程。
> 本文档是项目的顶层设计，代码编写前请先评审本文档。

---

## 1. 项目定位

| 项 | 说明 |
|---|---|
| **目标用户** | 个人开发者 / 研究者，单张消费级显卡（24G~48G） |
| **核心命题** | 用「单模型多角色」架构消除多模型并发，让 RLHF 全流程在单卡上可跑、可复现、可观测 |
| **对标差异** | 相比 `Joyce94/LLM-RLHF-Tuning`（传统 SFT→RM→PPO 三段式），本项目：去 Critic（GRPO）、去独立 Ref/RM（Adapter 复用 + 规则奖励）、覆盖 ORPO/SimPO/KTO 等现代免 Ref 算法、内置数据飞轮与评估闭环 |
| **运行环境** | **Linux 服务器**（开发可在 Windows，训练链路面向 Linux；vLLM / flash-attn / bitsandbytes 均按 Linux 选型） |
| **Trainer 策略** | **核心 trainer 全部自写**；数据格式与配置对齐 TRL，用 TRL 同配置曲线做正确性回归对照 |

### 非目标（v1 不做）

- 多卡 / 分布式训练（FSDP/DeepSpeed 多节点）——架构上不封堵，但不投入
- 多模态模型
- 全量参数 PPO（四模型）——被 GRPO 路线取代
- Web UI（先用 CLI + wandb）

---

## 2. 核心设计原则

### 2.1 单模型多角色（Unified Model）

显存中**永远只有一份主干权重**。所有"多模型"需求通过机制复用解决：

| 角色 | 实现机制 | 额外显存 |
|---|---|---|
| Actor（训练策略） | base + LoRA adapter `"default"`，正常反传 | LoRA 权重 + 优化器状态（很小） |
| Reference（参考策略） | 同 base + LoRA adapter `"ref"`，`set_adapter()` 切换 + `torch.no_grad()` | ≈ 0（一份冻结 LoRA 副本） |
| Reward | 规则函数 / 外部 API / 可选小型 RM | 规则与 API 为 0 |

> **关键修正**（相对初版方案）：Ref 不是 `disable_adapters()` 得到的裸 base——那是"base 直接开 RL"的特例。标准流程是 SFT 之后开 RL，Ref 必须是 **SFT 后策略的快照**。因此采用 PEFT 双 adapter 槽位：训练开始前把初始 LoRA 权重复制到 `"ref"` 槽并冻结，`set_adapter("ref" / "default")` 零拷贝切换。

### 2.2 去 Critic 化

不实现传统 PPO。在线强化统一走 **GRPO**：对同一 prompt 采样 G 个输出，用组内相对 reward 标准化得到 advantage，省掉 Critic 模型及其训练开销。

### 2.3 规则奖励优先

单卡场景下 RM 模型是显存负担。奖励插件优先级：

1. **RuleReward**：数学答案校验、代码执行、JSON/格式正则、长度惩罚——零显存、确定性强
2. **ApiReward**：外部 judge（可选，适合主观质量）
3. **ModelReward**：小型 RM（可选，仅 48G 卡或主模型 ≤3B 时建议）

### 2.4 算法按显存友好度分层推进

```
SFT ──► ORPO / KTO ──► DPO / SimPO ──► RAFT 飞轮 ──► GRPO
(基建)   (免Ref,单阶段)  (免Ref或双adapter)  (复用rollout)  (在线,最难)
```

- **ORPO**：无 Ref、SFT 与偏好对齐单阶段、显存 ≈ 纯 SFT —— **首发对齐算法**
- **KTO**：只需 binary 好/坏标签，数据门槛最低
- **SimPO**：无 Ref，用长度归一化 logprob 做隐式 reward
- **DPO**：经典基线，用双 adapter 机制实现 Ref
- **RAFT（数据飞轮）**：rollout → 规则筛选 → 好样本回炉 SFT；同 prompt 高分/低分自动组成 DPO 偏好对 —— **不是混合 loss，而是迭代 pipeline**
- **GRPO**：在线强化，复用 RAFT 的 rollout + reward 基建

> **术语澄清**（相对初版方案）：「SFT loss + DPO loss 混合训练」不是独立算法，其思想已被 ORPO 覆盖，不单独实现。

### 2.5 一切可降级

每个加速/显存特性都有 fallback：

| 特性 | 首选 | 降级 |
|---|---|---|
| Rollout 引擎 | vLLM（sleep/wake 共存） | HF `generate()`（慢 3~10x，零依赖） |
| Attention | flash-attn 2 | SDPA（PyTorch 原生） |
| 量化 | 4bit QLoRA (bnb) | 8bit / bf16 |
| 长文本显存 | Gradient checkpointing | 缩短 max_len |

---

## 3. 架构设计

### 3.1 目录结构

```text
OnlyOne/
├── configs/                     # YAML 配置（pydantic 校验）
│   ├── sft.yaml
│   ├── orpo.yaml
│   ├── dpo.yaml
│   ├── grpo.yaml
│   └── raft.yaml
├── onlyone/
│   ├── models/
│   │   ├── unified.py           # UnifiedModel：单主干 + 双adapter槽位 + 角色切换
│   │   └── loading.py           # 量化/flash-attn/设备映射的加载策略
│   ├── trainers/
│   │   ├── base.py              # 公共：logps计算、KL估计器(k1/k2/k3)、梯度累积、ckpt
│   │   ├── sft.py               # 支持 packing
│   │   ├── orpo.py              # 首发对齐算法
│   │   ├── dpo.py               # 含 SimPO 变体（loss 开关）
│   │   ├── kto.py
│   │   └── grpo.py              # 组相对策略优化
│   ├── rollout/
│   │   ├── base.py              # RolloutEngine 抽象接口
│   │   ├── hf_engine.py         # HF generate 降级路径（先实现）
│   │   └── vllm_engine.py       # vLLM colocate（sleep/wake），后实现
│   ├── rewards/
│   │   ├── base.py              # RewardFn 协议：fn(prompt, response, meta) -> float
│   │   ├── rules.py             # 数学/代码/格式/长度
│   │   ├── api.py               # 外部 judge
│   │   └── registry.py          # 奖励插件注册表（配置里按名字组合）
│   ├── data/
│   │   ├── templates.py         # ChatML / Qwen / Llama 模版
│   │   ├── datasets.py          # SFT / Preference / PromptOnly / Binary(KTO) 四种格式
│   │   └── packing.py           # SFT 专用序列拼接（仅限 SFT！）
│   ├── flywheel/
│   │   └── raft.py              # 数据飞轮：rollout→筛选→产出SFT/DPO数据→再训练
│   ├── eval/
│   │   ├── benchmarks.py        # GSM8K / IFEval 轻量评测
│   │   └── report.py            # checkpoint 对比报告
│   ├── utils/
│   │   ├── memory.py            # 显存监控/峰值上报/offload
│   │   ├── config.py            # pydantic 配置模型 + YAML 加载
│   │   └── logging.py           # wandb / tensorboard 封装
│   └── cli.py                   # typer 入口：train / rollout / eval / flywheel
├── scripts/
│   ├── prepare_data.py
│   └── compare_with_trl.py      # 与 TRL 同配置对照脚本（正确性回归）
├── tests/
│   ├── test_logps.py            # logps 数值正确性（对齐 HF 手算）
│   ├── test_adapter_switch.py   # ref/default 切换一致性
│   ├── test_losses.py           # 各 loss 的极值/对称性单测
│   └── test_smoke.py            # 小模型端到端冒烟（CI 可跑，CPU/tiny model）
├── examples/
│   └── qwen15b_gsm8k_grpo/      # 端到端示例：Qwen2.5-1.5B + GSM8K 规则奖励
├── docs/
│   └── design.md                # 本文档
├── pyproject.toml
└── README.md                    # 含能力矩阵（见 §6）
```

### 3.2 关键机制设计

**（a）UnifiedModel 角色切换**

```python
model = UnifiedModel(base="Qwen/Qwen2.5-1.5B", lora_cfg=...)
model.snapshot_ref()            # 训练前：default → 复制到 "ref" 槽，冻结
with model.as_reference():      # 上下文管理器：set_adapter("ref") + no_grad + eval
    ref_logps = model.logps(batch)
# 退出后自动恢复 default 槽 + train 模式
```

约束：`snapshot_ref()` 必须在任何训练 step 之前调用，trainer 初始化时强制断言。

**（b）GRPO 单卡流水线（每个 iteration）**

```text
1. Rollout: Actor 对 batch 内每个 prompt 采样 G 个输出 (eval 模式, no_grad, KV cache)
2. Reward: 规则插件计算组内 reward → 标准化得 advantage
3. Train forward: 重算当前 policy 的 logps（grad 开启, gradient checkpointing）
4. Ref forward: as_reference() 重算 ref logps（no_grad）
5. Loss: clipped surrogate + β·KL(k3 估计器)，反传更新 LoRA
```

**（c）vLLM 共存（第二阶段实现）**

vLLM 按 `gpu_memory_utilization` 预占显存，与训练峰值冲突。方案：
- vLLM ≥0.9 的 sleep/wake：rollout 前 `wake_up()`，rollout 后 `sleep()` 释放 KV cache 再进训练
- 权重同步：每 iteration 把 LoRA merge 后热加载进 vLLM（`load_lora`）或走 `collective` 更新
- 实现前先用 HF engine 跑通全流程，vLLM 作为纯性能优化插入

**（d）LoRA 热合并**：GRPO 每轮 rollout 前需要最新 Actor 权重生成——LoRA 场景下直接 forward 即可（adapter 已挂载），vLLM 场景才需要同步。

---

## 4. 数据格式（对齐 TRL，保证可对照）

| 格式 | 字段 | 用于 |
|---|---|---|
| SFT | `{"prompt": ..., "completion": ...}` 或 messages | SFT / ORPO 的 SFT 部分 |
| Preference | `{"prompt", "chosen", "rejected"}` | DPO / SimPO / ORPO |
| Binary | `{"prompt", "completion", "label": bool}` | KTO |
| PromptOnly | `{"prompt", "meta"?}` | GRPO / RAFT rollout |

模版系统统一在 `templates.py`，tokenize 时记录 response 区间 mask（completion-only loss）。

---

## 5. 评估与可观测性（不可裁剪）

没有 eval 的对齐是盲飞（reward hacking 会让训练曲线骗人）：

- **训练指标**：loss、reward mean/std（组内）、KL(policy||ref)、per-token entropy、response 长度均值、显存峰值、tokens/s
- **熔断**：KL 超阈值 / 长度爆炸 / reward 方差塌缩 → 告警或早停
- **Held-out 评测**：每个 milestone 在 GSM8K（数学）/ IFEval（指令遵循）上跑 accuracy，产出 checkpoint 对比报告
- **对照验证**：`scripts/compare_with_trl.py` 同数据同超参跑 TRL，loss/reward 曲线误差应在噪声范围内

---

## 6. 能力矩阵（写进 README，24G 单卡基准）

| 模型 | 精度/方式 | SFT | ORPO/KTO | DPO | GRPO (G=4) |
|---|---|---|---|---|---|
| 1.5B | bf16 全量 | ✅ | ✅ | ✅ | ✅ 舒适 |
| 3B | bf16 + LoRA | ✅ | ✅ | ✅ | ✅ |
| 7~8B | 4bit QLoRA | ✅ | ✅ | ✅ | ⚠️ 小 G + 短序列 + vLLM sleep |
| 14B | 4bit QLoRA | ⚠️ 仅 SFT | 勉强 | ❌ | ❌ |

48G 卡整体上调一档（7~8B 全流程舒适，14B 可 GRPO）。

---

## 7. 里程碑

### M1 · 基建 + SFT（第 1 周）
- UnifiedModel（加载策略、双 adapter、as_reference）
- pydantic 配置系统 + typer CLI + wandb 日志
- SFT trainer（含 packing、completion-only loss）
- `test_logps.py` / `test_adapter_switch.py` 数值单测
- **验收**：Qwen2.5-1.5B 在小 SFT 集上 loss 正常下降，与 TRL SFTTrainer 曲线对齐

### M2 · ORPO + KTO（第 2 周）
- ORPO / KTO trainer + Preference/Binary 数据格式
- 评估模块 v1（GSM8K）
- **验收**：ORPO 在 1.5B 上偏好准确率 >60%，TRL 对照通过

### M3 · DPO/SimPO + RAFT 飞轮（第 3~4 周）
- DPO（双 adapter ref）+ SimPO 开关
- RolloutEngine 抽象 + HF engine
- 规则奖励插件（数学校验、格式、长度）
- RAFT pipeline：rollout → 筛选 → SFT/DPO 数据产出 → 再训练
- **验收**：飞轮跑 2 轮，GSM8K accuracy 可测提升

### M4 · GRPO（第 5~6 周）
- GRPO trainer（组采样、advantage、KL 熔断）
- vLLM colocate 引擎（sleep/wake）
- examples/qwen15b_gsm8k_grpo 端到端示例
- **验收**：1.5B GRPO 训练 GSM8K accuracy 持续提升且无长度爆炸；QLoRA 7B 能跑通（不追求指标）

---

## 8. 技术选型

| 层 | 选型 | 理由 |
|---|---|---|
| 模型/训练 | transformers + peft + accelerate | 主流兼容，TRL 同栈便于对照 |
| 量化 | bitsandbytes | QLoRA 事实标准 |
| 生成 | vLLM（可选）+ HF generate（兜底） | 见 §3.2(c) |
| 配置 | pydantic v2 + YAML | 自用框架不上 Hydra |
| CLI | typer | 轻量 |
| 日志 | wandb（可选 tensorboard） | 指标 + 显存曲线 |
| 测试 | pytest + 小型随机模型（hf-internal tiny） | CI 可在 CPU 跑冒烟 |
| Python | ≥3.10 | — |

---

## 9. 风险清单

| 风险 | 等级 | 缓解 |
|---|---|---|
| 自写 GRPO 数值错误（logps/KL/advantage） | 高 | TRL 同配置对照脚本 + 单测覆盖 logps 与 loss 极值 |
| vLLM 与训练显存打架 | 中 | HF engine 先行；sleep/wake 兜底；`gpu_memory_utilization` 可配 |
| 7B QLoRA + GRPO 显存仍爆 | 中 | 降 G、降 max_len、rollout/train 串行 offload；能力矩阵提前管理预期 |
| Reward hacking（长度爆炸/格式投机） | 中 | 长度惩罚项 + KL 熔断 + held-out eval 强制纳入里程碑验收 |
| 范围膨胀（自用框架过度工程化） | 中 | 严格按里程碑走；非目标清单写死在 §1 |

---

## 10. 下一步

本文档评审通过后，从 **M1** 开始编码：先 `pyproject.toml` + `onlyone/models/unified.py` + 配置系统，第一行训练代码之前先把 `test_logps.py` 写出来（TDD 对数值正确性类代码收益最大）。
