# OnlyOne 数学 GRPO 操作手册（环境准备 → 数据配比定稿）

> 适用对象：第一次接手本项目的使用者。
> 范围：从空服务器到完成阶段 1 难度探测、定稿 GRPO 训练数据配比。
> 后续阶段（GRPO 主训练、评估）见 `qwen25_7b_math_grpo_plan.md` §6~§7（待补充进本文档）。
> 目标硬件：单卡 32G（RTX 5090 实测）；参考实现分支：`feat/7b-math-grpo`。

---

## 0. 总览

```
环境准备 ──► 阶段0 数据准备 ──► 阶段1 难度探测 ──► 配比定稿
 (30 min)     (CPU, 10 min)      (GPU 推理 1~2h)    (1 条命令)
```

核心原则：**GRPO 的训练信号来自组内 reward 方差**。7B-Instruct 在 GSM8K 上太强
（pass@1 ≈ 96%），必须混入足够比例的 MATH L3~L5，否则大量组全对/全错、无梯度。
配比不靠拍脑袋，靠阶段 1 实测。

---

## 1. 环境准备（Linux GPU 服务器）

### 1.1 依赖安装

```bash
# conda 环境（Python 3.12）
conda activate py312

# torch 必须 cu128 构建（RTX 50 系 Blackwell 的硬性要求）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# vllm 钉死 0.10.2:与 torch 2.8.0 实测兼容(pip check 无冲突,推理验证通过)
# 不要直接 pip install vllm(会拉最新版并可能把 torch 升级掉)
pip install "vllm==0.10.2"

# 项目本体(含 math reward 依赖)
cd /root/work/OnlyOne
pip install -e ".[vllm,math,qlora,logging]"
```

> 参考实测环境（2026-08-06,RTX 5090):torch 2.8.0(cu128)+ vllm 0.10.2。
> 历史验证过的另一组合是 torch 2.7.1+cu128 + vllm 0.10.x，亦可使用。

验证：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import vllm, bitsandbytes, math_verify; print('deps ok')"
pip list | grep -E "^torch "   # 确认 torch 是预期版本,没被其他安装换掉
pip check                      # 应输出 No broken requirements found
```

### 1.2 模型下载

国内服务器建议走 modelscope：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /cloud/models/Qwen2.5-7B-Instruct
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir /cloud/models/Qwen2.5-1.5B-Instruct
```

或 HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com` 后用 `hf download`。

### 1.3 已知坑（免排查）

| 现象 | 原因 | 处理 |
|---|---|---|
| 训练中途 segfault | vLLM sleep 状态下 load_weights（已在 `a729403` 修复） | 拉最新分支即可 |
| `'LLMEngine' object has no attribute 'model_executor'` | vLLM ≥0.10 默认 V1 engine | 代码已强制 `VLLM_USE_V1=0`，无需操作 |
| 训练结束后退出码 139 | 上游 bug vllm#16993，退出时 double-free | 不影响训练结果，可忽略 |
| flash_attn 未安装 | sm_120 编译成本高 | 配置用 `attn_implementation: sdpa`，不影响正确性 |

---

## 2. 阶段 0：数据准备（CPU 即可）

仓库已带生成好的 `data/math/`，如需从头生成：

```bash
python scripts/prepare_math_data.py --output-dir data/math
```

产出 6 个文件：

| 文件 | 条数 | 用途 |
|---|---:|---|
| `gsm8k_train.jsonl` | 7,473 | GSM8K 训练 prompts |
| `gsm8k_test.jsonl` | 1,319 | GSM8K held-out 评测 |
| `math_train.jsonl` | 7,500 | MATH 训练 prompts（带 level） |
| `math500_test.jsonl` | 500 | MATH-500 held-out 评测 |
| `prompts_mix.jsonl` | 4,000 | GRPO 主训练 prompt 集 |
| `prompts_mix_probe.jsonl` | 600 | 阶段 1 探测集（每难度档 100 条） |

数据格式：`{"prompt": ..., "meta": {"gold": "42", "level": 0, "source": "gsm8k"}}`，
level 0 = GSM8K，1~5 = MATH 难度。

跑测试确认环境正确：

```bash
python -m pytest tests -q
# 预期: 83 passed
```

---

## 3. 阶段 1：难度探测（GPU 纯推理，约 1~2 小时）

目的：实测目标模型在各难度档的通过率，用数据决定配比。

### 3.1 探测配置

参考 `raft_probe.yaml`（按实际路径改 `name_or_path` / `prompts_path` / `output_dir`）：

```yaml
model:
  name_or_path: /cloud/models/Qwen2.5-7B-Instruct
  dtype: bf16
  attn_implementation: sdpa
  load_in_4bit: false
  use_lora: false             # 纯推理探测,不需要 adapter

data:
  train_path: data/sft_example.jsonl   # schema 必填,探测不读
  template: chatml
  max_len: 1280
  packing: false

train:
  output_dir: runs/raft_probe_7b/sft   # train_sft: false,不会用到
  max_steps: 1
  log_with: none
  run_name: raft-probe-7b

algo:
  name: sft

raft:
  prompts_path: data/math/prompts_mix_probe.jsonl
  group_size: 4               # pass@4 探测
  rewards: [math_verify]
  threshold: 0.5              # 任一候选答对即保留 → keep_rate = pass@4
  rounds: 1                   # 只探测不训练
  max_new_tokens: 512
  temperature: 0.8
  top_p: 0.95
  rollout_batch_size: 16
  output_dir: runs/raft_probe_7b
  train_sft: false
  make_preference: false
  engine: hf                  # 7B bf16 单副本;vllm 会与训练侧模型双副本超显存
```

### 3.2 执行与分析

```bash
nohup onlyone flywheel --config raft_probe.yaml > runs/raft_probe_7b.log 2>&1 &

# 跑完后分析
python scripts/analyze_probe.py runs/raft_probe_7b/round_0/scores.jsonl
```

### 3.3 判定表

| keep_rate | 判定 | 动作 |
|---|---|---|
| 20%~70% | ✅ 甜区 | 进入 GRPO prompt 集主力 |
| >80% | 太简单 | 降占比 |
| <5% | 太难 | 剔除 |

Qwen2.5-7B-Instruct 实测参考（2026-08-06，RTX 5090）：

| 难度桶 | pass@1 | pass@4 | 判定 |
|---|---:|---:|---|
| gsm8k (L0) | 96.0% | 97.0% | 太简单 |
| L1 | 90.8% | 93.0% | 太简单 |
| L2 | 83.5% | 90.0% | 太简单 |
| L3 | 62.3% | 79.0% | 边缘，作过渡 |
| L4 | 50.0% | 66.0% | ✅ 甜区 |
| L5 | 19.2% | 26.0% | ✅ 甜区 |

换模型时必须重跑探测，不要套用此表。

---

## 4. 数据配比定稿

根据探测结果确定各 level 权重，重新生成 `prompts_mix.jsonl`：

```bash
python scripts/prepare_math_data.py \
  --mix-level-weights '{"0":5,"1":5,"2":10,"3":25,"4":27.5,"5":27.5}'
```

权重说明（7B-Instruct 实测后的推荐值）：

- L0~L2 合计 20%：保留基础算术能力，防 GSM8K held-out 下跌
- L3 25%：边缘区过渡
- L4+L5 合计 55%：甜区主力，提供主要梯度信号

验证配比：

```bash
python -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('data/math/prompts_mix.jsonl', encoding='utf-8')]
print(Counter(r['meta']['level'] for r in rows))
"
# 预期: {0: 200, 1: 200, 2: 400, 3: 1000, 4: 1100, 5: 1100}
```

不传 `--mix-level-weights` 时保持旧的 1:1:2（GSM8K : L1~2 : L3~5）行为，仅作兼容，
不推荐用于 7B-Instruct。

---

## 5. 冒烟验证（可选但强烈建议）

正式 GRPO 前先用 1.5B 小跑 10 步，确认管线正常（vLLM 热加载、reward、checkpoint）：

```bash
onlyone train-grpo --config test_grpo.yaml \
  --override model.name_or_path=/cloud/models/Qwen2.5-1.5B-Instruct \
  --override train.max_steps=10
```

健康标志：每步有 `reward_mean` 输出、`checkpoint saved` 正常、`vram_peak_gb` 稳定。
若 `n_degenerate_groups` 持续过半，说明配比或 group_size 有问题（G 至少为 4）。

> 权重同步机制（vLLM 引擎）：训练侧只存 LoRA adapter（r=16 约 160MB）,
> vLLM 常驻 bf16 基座，每步通过 `add_lora`/`remove_lora` 热插拔（秒级）。
> 因此 vLLM 引擎要求 `use_lora: true`；全量训练请用 `engine: hf`。

---

## 下一步（本文档待续）

- 阶段 3：GRPO 主训练配置（`configs/grpo_7b_math.yaml`，QLoRA 4bit + vLLM）
- 阶段 4：MATH-500 / GSM8K 全量对比评估

详见 `qwen25_7b_math_grpo_plan.md` §6~§7。
