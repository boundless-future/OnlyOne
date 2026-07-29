# OnlyOne 提交与 PR 规范

> 本规范约束所有成员(包括 AI 协作者)的 commit、分支与 Pull Request。
> 格式基线:[verl 官方规范](https://github.com/volcengine/verl/blob/main/CONTRIBUTING.md) 的 `[module] type: Title`。

---

## 1. Commit 与 PR 标题格式

```
[module] type: Title
```

- **module**(必填):影响范围,方括号小写,见 §2;多模块用逗号分隔:`[trainer, data] feat: ...`
- **type**(必填):只允许 4 个:`feat` / `fix` / `refactor` / `chore`(见 §3)
- **Title**(必填):祈使句、小写开头、**不超过 72 字符**、结尾不加句号
- **Breaking**:破坏 API 时前缀 `[BREAKING]`:`[BREAKING][trainer] feat: rename build_sft_trainer`

示例:

```
✅ [trainer] feat: add KTO trainer with dual-adapter reference
✅ [eval] fix: strip commas from extracted GSM8K numbers
✅ [model] refactor: extract token_logps from logps
✅ [doc] chore: add git conventions
✅ [trainer, data] feat: add Preference dataset and ORPO trainer

❌ update code                                  (无 module、无 type)
❌ [Trainer] feat: Add ORPO Trainer.            (module 大写、Title 大写开头、句号结尾)
❌ fix bug                                      (无信息量)
```

---

## 2. module 取值

与 `onlyone/` 包结构对应:

| module | 范围 | module | 范围 |
|---|---|---|---|
| `trainer` | 各算法训练器 | `cli` | 命令行入口 |
| `model` | UnifiedModel、加载策略 | `config` | 配置系统与 YAML |
| `data` | 数据集、模版、packing | `test` | tests/ |
| `eval` | 评测模块 | `ci` | CI、pre-commit |
| `rollout` | 生成引擎 | `doc` | docs/、README |
| `reward` | 奖励插件 | `build` | 依赖、打包 |
| `flywheel` | 数据飞轮 | `perf` | 性能专项 |
| `ckpt` | 存档/续训 | `misc` | 其他杂项 |

跨 3 个及以上模块:列最主要的 2 个,不堆砌。

---

## 3. type 取值(只允许 4 个)

| type | 用途 |
|---|---|
| `feat` | 新功能、新算法、新模块、新测试能力 |
| `fix` | 缺陷修复(含数值正确性修复) |
| `refactor` | 不改变外部行为的结构调整 |
| `chore` | 文档、依赖、CI、脚本、杂项 |

文档、测试、CI 一律归 `chore`,靠 module(`doc` / `test` / `ci`)区分,不设独立 type。

---

## 4. 分支命名

```
<type>/<short-description>
```

- `type` 与 §3 一致
- 短描述:kebab-case、英文、≤4 词
- `main` 为保护分支:**禁止直接推送**,一切改动走 PR

示例:`feat/orpo-kto`、`feat/grpo`、`chore/git-conventions`、`fix/ref-snapshot`

---

## 5. 功能分支上的 commit

- 主分支采用 **Squash merge**:PR 标题即主分支 commit 标题,必须符合 §1 格式
- 功能分支上的中间 commit 不强制 §1 格式,但必须:
  - 原子提交:一个 commit 只做一件事
  - subject 有信息量(禁止 `update`、`fix bug`;临时提交用 `wip: xxx` 并在 PR 前说明)
- commit body 解释"为什么"而非"是什么";关联 issue 用 `Closes #N`
- 统计信息(`35 tests`、`+1200 lines`)属于 PR 正文,不属于 commit 标题

---

## 6. Pull Request 规范

### 6.1 标题

**描述准确、言简意赅,一句话讲清楚这个 PR 做了什么。** 格式同 §1(`[module] type: Title`)。

### 6.2 正文模板(必填五项)

```markdown
## 概述
一段话说明这个 PR 是什么、为什么做。

## 设计简介
关键设计决策与取舍(为什么这么做而不是别的方案);无设计内容可写"无"。

## 关键变更
- 变更 1:简要说明
- 变更 2:简要说明

## 测试结果
跑了什么测试、结果如何;附关键输出或数据。

## 遗留问题(及潜在影响)
已知未解决的问题、待验收项、可能影响的面;无则写"无"。
```

### 6.3 合并规则

- **Squash and merge**;合并后 commit 标题 = PR 标题
- 合并后删除功能分支
- 需要 GPU 真机验证的事项放入"遗留问题",允许先行合并,但必须显式列出

---

## 7. 所有协作者(含 AI)的强制约定

1. **先同意,后提交**:任何 `git commit`、`git push`、创建/合并 PR 之前,必须先向项目所有者说明要做的操作并**获得明确同意**,不得私自执行。
2. AI 协作者不得 `--force` 推送已开 PR 的分支(除非项目所有者明确要求)。
3. **AI 协作者必须自我标识**(工具无关):
   - commit footer 带 `Co-Authored-By: <AI 工具名> <工具方公开联系地址>`
   - PR 正文末尾用一行注明本 PR 由何种 AI 工具辅助生成
   - 目的仅为溯源与 review 侧重,不作任何工具推荐或限制

---

## 参考

- [verl CONTRIBUTING.md](https://github.com/volcengine/verl/blob/main/CONTRIBUTING.md)
- [verl PR 实例](https://github.com/volcengine/verl/pulls)(`[sglang] fix: ...`、`[trainer] feat: ...`)
