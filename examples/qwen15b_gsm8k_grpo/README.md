# 端到端示例:Qwen2.5-1.5B + GSM8K + GRPO

这是 OnlyOne 的"hello world"全链路示例:从 SFT 到 GRPO 在线强化,
在 24G 单卡上让 1.5B 模型的数学能力可测提升。

## 前置条件

- Linux 服务器,GPU ≥ 24G(RT 3090/4090/A5000)
- `pip install -e ".[qlora,logging]"`(vllm 可选)

## 数据准备

GSM8K 原始格式需要转换成本框架的 jsonl。`scripts/prepare_data.py`(后续补充)或手动:

```jsonl
// prompts 文件(GRPO 用):{"prompt": 问题, "meta": {"gold": 答案数字}}
{"prompt": "Natalia sold clips to 48 friends in April...", "meta": {"gold": "72"}}
```

示例数据已带 6 条:`data/prompts_example.jsonl`(冒烟用,真训练换完整 GSM8K train 集)。

## 第一步:SFT(可选但强烈推荐)

GRPO 的 KL 锚点是 SFT 后的策略。直接在 base 模型上 GRPO,候选质量太差,
reward 方差为 0(全错),学不到东西。

```bash
onlyone train --config configs/sft.yaml   # 用 GSM8K SFT 数据
```

## 第二步:GRPO

把 `configs/grpo.yaml` 的 `model.name_or_path` 指向上一步的 SFT 输出
(或 base + SFT adapter 合并后的目录),然后:

```bash
onlyone train-grpo --config configs/grpo.yaml
```

每步观察:
- `reward_mean` 应单调上升(数学答案准确率)
- `kl` 应缓升但远低于熔断线 0.5
- `completion_chars` 不应爆炸式增长(长度 hacking 预警)
- `clip_frac` 健康范围约 5%~20%;持续 >30% 说明 lr 太大

## 第三步:验收

```bash
onlyone eval --config configs/grpo.yaml \
  --adapter runs/grpo_qwen15b/final \
  --eval-path data/gsm8k_test.jsonl --limit 200
```

对照 SFT 模型的同口径评测,accuracy 应有可见提升(1.5B + GSM8K 典型 +3~8 点)。

## vLLM 加速(可选)

```yaml
grpo:
  engine: vllm
  vllm_gpu_mem_util: 0.35
```

rollout 提速 3~10 倍。权重每步热加载,显存与训练共存(sleep/wake)。
