# OnlyOne 数学 GRPO 操作手册（环境准备 → GRPO 主训练 → 评估验收）

> 适用对象：第一次接手本项目的使用者。
> 范围：从空服务器到 GRPO 主训练、监控与阶段 4 评估验收（阶段 0~4）。
> 目标硬件：单卡 32G（RTX 5090 实测）；参考实现分支：`feat/7b-math-grpo`。

---

## 0. 总览

```
环境准备 ──► 阶段0 数据准备 ──► 阶段1 难度探测 ──► 配比定稿 ──► 阶段3 GRPO 主训练
 (30 min)     (CPU, 10 min)      (GPU 推理 1~2h)    (1 条命令)     (单卡 ~4.2h)
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
| 训练中途 segfault | vLLM sleep 状态下变更引擎（已在 `a729403` 修复：wake 先行） | 拉最新分支即可 |
| `'LLMEngine' object has no attribute 'model_executor'` | vLLM ≥0.10 默认 V1 engine | 代码已强制 `VLLM_USE_V1=0`，无需操作 |
| `wake_up` 时 cuMem OOM | PyTorch 缓存块挡住物理映射（已在 `f9dccf5` 修复） | 拉最新分支即可 |
| 训练 forward OOM(7B 全词表 logits) | fp32 logits 峰值 + autograd 保留（已在 `8b2ac5e`/`a3d4fa4` 修复） | 拉最新分支即可 |
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
# 预期: 87 passed
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
nohup onlyone flywheel --config raft_probe.yaml > runs/raft_probe_7b_$(date +%Y%m%d_%H%M%S).log 2>&1 &

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

> 指标观测：每个记录步都会追加一行 JSON 到 `{output_dir}/metrics.jsonl`
> （与 wandb/tb 后端无关，零依赖）。随时用
> `python scripts/check_training.py <output_dir>/metrics.jsonl`
> 看尾部指标表 + 健康判定（kl / clip_frac / 长度 / 显存 / reward 趋势 / 退化组）。

---

## 6. 阶段 3：GRPO 主训练（单卡 32G，约 4.2 小时）

### 6.1 配置：`configs/grpo_7b_math.yaml`

- 模型：Qwen2.5-7B-Instruct,QLoRA 4bit + LoRA r=16/α=32,`sdpa`
- 数据：`prompts_mix.jsonl`(4000 条，§4 配比）
- GRPO:G=4,prompts_per_step=8（每步 32 条）,rewards=[math_verify, length],
  kl_beta=0.04,KL 熔断 0.5
- 生成：max_new_tokens=512,temperature=0.9
- 训练：lr=5e-6,300 步，save_steps=20,logging_steps=5,`keep_last_n_checkpoints: 6`
  （轮转保留最近 6 个检查点，覆盖最近 120 步，宕机最多损失 20 步）
  （lr 历史：run#1 用 1e-6,200 步仅消耗 0.04% kl 预算、reward 爬升过慢，故 ×5)
- rollout:engine=vllm,`vllm_gpu_mem_util: 0.55`(7B bf16 基座 ~15G 必须落在
  vLLM 预算内：0.55×32G≈17G = 权重 15G + KV ~2G；训练期 vLLM sleep 不占显存）

### 6.2 先 3 步冒烟（必做）

```bash
onlyone train-grpo -c configs/grpo_7b_math.yaml \
  -O model.name_or_path=/cloud/models/Qwen2.5-7B-Instruct \
  -O train.max_steps=3 -O train.logging_steps=1 \
  -O train.output_dir=runs/grpo_7b_smoke
```

通过标准：3 步无错误完成、逐步指标打出、`vram_peak_gb` < 30、
`n_degenerate_groups` 为 0 或接近 0。实测（2026-08-08）约 50s/step。

### 6.3 正式开跑

```bash
# 日志落盘带时间戳:nohup 不重定向的话,输出丢了就只剩 metrics.jsonl 可看
# 注意:每一轮新训练必须用新的 output_dir —— 复用旧目录会把 metrics.jsonl
# 追加混叠,趋势判定就没法看了(旧目录保留,里面的检查点留作对比)
nohup onlyone train-grpo -c configs/grpo_7b_math.yaml \
  -O model.name_or_path=/cloud/models/Qwen2.5-7B-Instruct \
  -O train.output_dir=runs/grpo_7b_math_run2 \
  > runs/grpo_7b_math_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

checkpoint 每 50 步存到 `runs/grpo_7b_math/`(LoRA adapter，每个 ~160MB),
指标每 5 步追加到 `runs/grpo_7b_math/metrics.jsonl`。

### 6.4 监控（不用爬日志）

> 路径规则：`metrics.jsonl`、step 检查点都落在 **启动命令的 `output_dir`** 里。
> 下面用配置默认的 `runs/grpo_7b_math`;如果你启动时覆盖了
> `-O train.output_dir=...`,把路径换成你的 output_dir（续训的
> `resume_from` 同理）。

```bash
# 尾部指标表 + 六项健康判定,训练中随时可跑
python scripts/check_training.py runs/grpo_7b_math/metrics.jsonl

# 加 ASCII 趋势图(零依赖) / PNG 六宫格(需 pip install matplotlib)
python scripts/check_training.py runs/grpo_7b_math/metrics.jsonl --spark
# 裸 --plot 存到 metrics.jsonl 同目录(output_dir/trend.png);也可显式给路径
python scripts/check_training.py runs/grpo_7b_math/metrics.jsonl --plot
```

| 指标 | 健康形态 | 异常 → 预案 |
|---|---|---|
| `reward_mean` | 缓慢上升(0.3→0.5 量级) | 长期不动 → 难度错配,回阶段 1 |
| `reward_L45` | 缓慢上升(难度固定桶,最干净的学习信号) | 上升而 overall 不动 → 只是抽样噪声;两者都不动 → 真没学到 |
| `kl` | <0.1 缓涨 | 触发 0.5 熔断 → 见 6.5 |
| `completion_chars` | 平稳 | 持续上涨超 2x → 长度黑客,收紧 length penalty |
| `clip_frac` | 0.05~0.2 | >0.3 → lr 偏大 |
| `n_degenerate_groups` | <一半 | 持续过半 → 配比失衡 |
| `vram_peak_gb` | <30G | 逼近上限 → `vllm_gpu_mem_util` 降 0.05 重跑 |

### 6.5 异常处置与断点续训

- **KL 熔断**：自动保存 checkpoint 后报错退出（这是设计行为，不是 bug）。
  处置：降 lr（如 1e-6 → 5e-7）或升 kl_beta，然后从最近的 checkpoint 续训：
  ```bash
  onlyone train-grpo -c configs/grpo_7b_math.yaml \
    -O model.name_or_path=/cloud/models/Qwen2.5-7B-Instruct \
    -O train.lr=5e-7 \
    -O train.resume_from=runs/grpo_7b_math/step250
  ```
  续训会恢复 policy adapter 权重、optimizer/scheduler 状态、global_step 和
  prompt 采样流位置；ref 锚点保持为起跑时的初始策略（实现上先冻结 ref、
  再加载续训权重，顺序由 CLI 保证）。检查点轮转默认
  `keep_last_n_checkpoints: 6`——只留最近 6 个 step 目录，`final` 永远保留。
- **pod 宕机/被回收**：同上用最新保留的 step 目录续训，最多损失
  `save_steps`(20）步。
- **OOM**：先把 `vllm_gpu_mem_util` 降 0.05 重跑；仍 OOM 则
  `prompts_per_step` 8→4。
- **退出码 139**：上游 vllm#16993 退出时 double-free,checkpoint 已落盘，忽略。

### 6.6 单卡共存架构（排障背景知识）

训练侧 4bit QLoRA 模型与 vLLM(bf16 基座）共用一张卡，严格交替：

```
sync: 存 policy adapter(~160MB)→ wake_up() → add_lora(新id)/remove_lora(旧id)
rollout: generate(lora_request=policy) → sleep()(权重卸载到 CPU,释放 ~16G)
train: 前向/反向/优化器(此时 vLLM 不占显存)
```

两条铁律（都是 segfault/OOM 换来的）:**任何引擎变更必须先 `wake_up()`**;
**`wake_up()` 前必须 `torch.cuda.empty_cache()`** 把 PyTorch 缓存块还给驱动。

---

## 7. 阶段 4：固定题集评估（验收）

训练内 reward 混有 length 成分（completion 变短也挣分），只有固定题集 +
纯正确率是终裁。用 `scripts/eval_math.py`：同一个 vLLM 实例里 base 与
adapter 先后生成（解码条件严格一致），贪婪 pass@1，只跑 math_verify:

```bash
# MATH-500(base + adapter 各一遍,500 题约 15~25 分钟)
nohup python scripts/eval_math.py \
  --model /cloud/models/Qwen2.5-7B-Instruct \
  --adapter <output_dir>/step40 \
  --data data/math/math500_test.jsonl \
  --out <output_dir>/eval_step40_math500.json \
  > eval_math500_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# GSM8K held-out 回退检查(1319 题)
nohup python scripts/eval_math.py \
  --model /cloud/models/Qwen2.5-7B-Instruct \
  --adapter <output_dir>/step40 \
  --data data/math/gsm8k_test.jsonl \
  --out <output_dir>/eval_step40_gsm8k.json \
  > eval_gsm8k_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

输出 overall + per-level 的 base/adapter 正确率与平均长度对照表；
`--limit 20` 可先快速自检链路。验收判读：

- **MATH-500 Δ ≥ +5pt** 为主要验收线
- **GSM8K Δ ≥ −2pt** 为回退线；掉得接近线说明 completion 被压得过短
  （简单题跳步骤），换更早的检查点重评（早停点常优于最终点——实测
  step40 全面优于 step60)
- `adapter_chars` 显著低于 base 且正确率上升 = 高效推理（好）；正确率
  不涨只变短 = 长度塑形刷分（坏）

参考结果（2026-08-08,lr=1.6e-4 run,step40):MATH-500 +13.0pt、
GSM8K −0.8pt，验收通过。详见笔记 `qwen25_7b_math_grpo_experiments.md`。
