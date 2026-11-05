# EcoWitness v9 测试包：河流生态证据审核与修订评测

这个包用来测一个多模态模型能不能**根据证据回答、说清依据、并在证据被修订后正确地改或不改答案**。
题目来自中国河流样点的卫星影像、eDNA 鱼类名录、站点记录和物种知识。全部为四选一，但修订后可能没有任何选项成立，此时正确答案是 `none`。

跑完把 `predictions.jsonl` 发回给出题人打分。测试集不含答案；`dev` 集含答案，用来自查流程。

## 1. 任务形式

每条记录是一段对话。证据条目都有编号（“证人”）：`IMG` 影像、`R1…` 站点记录字段、`SP-01…` 鱼类卡片、`A-01 / B-01` 两个样点的卡片、`ROW-01…` 序列匹配行。模型必须按固定格式回答：

```
answer: C
support: [SP-03, SP-07, IMG]
load_bearing: [SP-03]
```

- `support`：答案依赖的全部证人。
- `load_bearing`：**单独去掉就会改变答案**的证人。
- 收到修订通知后，再多输出第一行 `changed: yes` 或 `changed: no`。

三种记录（字段 `mode`）：

| mode | 内容 | 每条要生成几次 |
|---|---|---|
| `base` | 原题 | 1 |
| `revise` | 原题 + 模型的先前回答（给定）+ 一条修订通知（撤销、新增、改判、改字段、换影像） | 1 |
| `chain` | 原题 + 2 到 5 步连续修订。每一步都要把**模型自己上一步的回答**接到对话里再问下一步 | 1 + 步数 |

题族 27 个，按 `group` 分六组：A 影像判读、B DNA 名录推理、C 影像与名录融合、D 群落多样性与保护状态、E 站点记录推理、K 物种知识。`tier` 是难度（easy / medium / hard）。

## 2. 包内文件

| 文件 | 说明 |
|---|---|
| `test.public.en.jsonl`、`test.public.zh.jsonl` | 测试集，英文和中文各一份，**不含答案** |
| `dev.en.jsonl`、`dev.zh.jsonl` | 开发集（含答案、含 `edit_class` 标签），自查用 |
| `satellite_rgb/` | 影像，256×256 PNG，Sentinel-2 RGB，933 张 |
| `ecowitness/` | 运行与打分代码：`evaluate_api.py`（OpenAI 兼容 API）、`evaluate.py`（本地 HF 模型）、`score.py`（打分 / 格式检查）、`runner.py`、`protocol.py` |
| `requirements_api.txt`、`requirements_local.txt` | 两种跑法的依赖 |
| `manifest.json` | 文件清单与 sha256 |

规模（每种语言）：

| | base | revise | chain | chain 总步数 | 生成次数 |
|---|---|---|---|---|---|
| test | 4,263 | 12,694 | 4,263 | 14,954 | 36,174 |
| dev | 1,904 | 5,863 | 1,904 | 6,598 | 16,269 |

单条对话一般 1–2k token（不含影像），最长约 4.8k。走 API 的话，一种语言的完整测试约 5 千万输入 token。建议分两档：

- **最小方案**：`--langs en --modes base revise`，16,957 次生成。
- **完整方案**：`--langs en zh --modes base revise chain`。

## 3. 运行

### A. OpenAI 兼容 API（GPT、Gemini、Qwen、DashScope、vLLM 等）

```bash
pip install -r requirements_api.txt
export OPENAI_API_KEY=...            # 按服务商设置
export OPENAI_BASE_URL=...           # 例如 https://dashscope.aliyuncs.com/compatible-mode/v1
python -m ecowitness.evaluate_api --model <模型名> --data_dir . --workspace . \
    --split test --langs en zh --modes base revise chain --workers 4 --out runs/<模型名>
```

- 影像以 base64 PNG 发送。温度默认 0。
- 思考模型：要么 `--no_thinking`（vLLM、DashScope 支持），要么把 `--max_tokens` 提到 2048 以上，否则思考过程会把答案挤掉。打分时 `</think>` 之前的内容会被忽略。
- 回复按对话缓存在 `runs/<模型名>/cache.jsonl`，中断后重跑会接着做。

### B. 本地 HuggingFace 模型（Qwen3.5 系列，或任何 `AutoModelForImageTextToText` 能加载的模型）

```bash
pip install -r requirements_local.txt
python -m ecowitness.evaluate --model <模型路径> --data_dir . --workspace . \
    --split test --langs en zh --modes base revise chain --batch_size 8 --out runs/<模型名>
```
贪心解码，`--max_new_tokens 64` 够用（输出只有三四行）。LoRA 适配器加 `--adapter <路径>`。

### C. 自己的推理框架

只要产出同样格式的 `predictions.jsonl` 即可，每行一条：

```json
{"id": "Q9_..|en|base", "lang": "en", "mode": "base", "output": "answer: B\nsupport: [...]\nload_bearing: [...]"}
{"id": "Q9_..|en|e2",   "lang": "en", "mode": "revise", "output": "changed: yes\nanswer: none\n..."}
{"id": "Q9_..|en|chain", "lang": "en", "mode": "chain", "history": "own", "base_output": "...", "step_outputs": ["...", "..."]}
```

`messages` 字段就是要发给模型的对话：`role` 为 `system` / `user` / `assistant`；`user` 的 `content` 是列表，含 `{"type":"image","path":...}`（相对本目录）和 `{"type":"text","text":...}`。
链式记录的做法：先对 `messages` 生成，得到 `base_output`；然后依次对每个 `steps[k]`，把 `{"role":"assistant","content":<上一步你的回答>}` 和 `steps[k]["message"]` 追加到对话后再生成。**必须用模型自己的回答做历史**，不要改写。

## 4. 先在 dev 上验证流程

```bash
python -m ecowitness.evaluate_api --model <模型名> --data_dir . --workspace . --split dev --langs en --max_items 50 --out runs/dev_check
```
dev 有答案，跑完会直接生成 `metrics.json`。看三件事：`parse_rate` 接近 1（模型按格式输出了）、`invalid_citation_rate` 接近 0（引用的编号都存在）、`coverage` 里 predicted 等于 total。

## 5. 检查并回传

测试集没有答案，跑完生成的是 `check.json`（覆盖率、格式解析率、无效引用率、链步数是否对得上）。也可以单独检查：
```bash
python -m ecowitness.score --data test.public.en.jsonl test.public.zh.jsonl --predictions runs/<模型名>/predictions.jsonl
```
回传 `runs/<模型名>/predictions.jsonl`，并注明：模型名与版本、是否开启思考、温度、`max_tokens`、跑了哪些语言和模式。

## 6. 规则

1. base 题只能用 base 记录回答。revise 记录里 assistant 的先前回答是题目设定的一部分（是正确的原答案），不能拿去回答 base 题。
2. 不要在 test 上训练或调提示；dev 可以用来调试流程。
3. 不要改 system prompt 和证据文本；贪心解码或温度 0。
4. 链式题必须用模型自己的历史。

## 7. 指标

打分输出 `metrics.json`，分 `all`（中英合并）、`en`、`zh` 三个范围，每个范围下按 base / revise / chain 分块，最后有一张 `scorecard`。

**base（原题）**

| 指标 | 含义 |
|---|---|
| `accuracy` | 选项准确率，另按 group、family、tier 分列；`accuracy_ci95` 为自助法 95% 区间 |
| `parse_rate`、`full_format_rate` | 能解析出 answer 的比例；四行齐全的比例 |
| `citation.support` / `citation.load_bearing` | 引用的精确率、召回率、F1，按证人编号与真值集合比 |
| `load_bearing_exact` | 承重集合完全正确的比例 |
| `sufficiency` | 真正承重的证人**全部**出现在模型的 support 里的比例，衡量有没有漏看关键证据 |
| `invalid_citation_rate` | 引用了对话里不存在的编号的比例（编造引用） |
| `faithful_accuracy` | 答案对**且**承重集合完全对的比例，衡量“因为对的理由而答对” |
| `load_bearing_exact_given_correct` | 答对的题中承重集合也对的比例 |
| `abstention` | 回答 `none` 的精确率、召回率、F1；`false_none_rate` 是本应有答案却答 none 的比例 |

**revise（单步修订）**，除上表外：

| 指标 | 含义 |
|---|---|
| `by_edit.class:*` | 按修订类别分的准确率与改答率：`flip` 答案应变、`keep_cited` 改了被引用但冗余的证人、`keep_uncited` 改了无关证人 |
| `by_edit.type:*` | 按修订动作分：remove / add / reassign / set_field / swap_image |
| `revision.changed_f1` | `changed` 行的 F1；`change_detection_accuracy` 是 changed 行判对的比例 |
| `revision.contagion_rate` | 无关证人被改时模型跟着改答案的比例（越低越好） |
| `revision.redundancy_overreaction_rate` | 冗余证人被改时改答案的比例（越低越好） |
| `revision.inertia_rate` | 承重证人被改时**没**改答案的比例（越低越好） |
| `revision.flip_accuracy` / `keep_accuracy` | 应变题和应保持题各自的准确率 |
| `revision.stale_citation_rate` | 修订撤销了某证人后仍然引用它的比例 |
| `revision.self_contradiction_rate` | `changed` 行与实际答案是否变化相矛盾的比例 |
| `revision.accuracy_truth_none` | 修订后应答 none 的题的准确率 |

**chain（多步修订，模型自己的历史）**

| 指标 | 含义 |
|---|---|
| `accuracy_by_step` | 第 0 步（原题）到第 5 步各自的准确率 |
| `final_step_accuracy`（+CI、按链长） | 最后一步的准确率 |
| `whole_chain_accuracy` | 整条链每一步都对的比例 |
| `drift` | 第 0 步准确率减最后一步准确率，衡量多轮后的退化 |
| `recovery_rate` / `slip_rate` | 上一步错、这一步对的比例；上一步对、这一步错的比例 |
| `step_accuracy_by_edit_class` | 各步按修订类别分的准确率与改答率 |

**cross_lingual**：同一道题中英文答案一致的比例、两种语言都答对的比例。

## 8. 参考数字（测试集，中英合并）

| 预测方式 | base 准确率 | faithful | 承重 F1 | 修订后准确率 | changed F1 | 传染率 | 惯性率 | 陈旧引用率 | 链末步准确率 |
|---|---|---|---|---|---|---|---|---|---|
| 随机选项 | 0.253 | 0.087 | 0.28 | 0.208 | 0.61 | 1.00 | 0.00 | 0.37 | 0.190 |
| 永远不改（照抄先前回答） | 0.244 | 0.087 | 0.28 | **0.557** | 0.00 | 0.00 | 1.00 | 0.73 | 0.190 |
| 修订后一律答 none | 0.245 | 0.085 | 0.28 | 0.164 | 0.61 | 1.00 | 0.00 | 0.00 | 0.183 |
| 标准答案 | 1.000 | 1.000 | 1.00 | 1.000 | 1.00 | 0.00 | 0.00 | 0.00 | 1.000 |

两点提醒：修订题里 55.7% 的正确做法是保持原答案，所以“修订后准确率”要和惯性率、`flip_accuracy` 一起看；`none` 题占修订题的 16.4%，一律答 none 的 F1 只有 0.28。

## 9. 输出协议原文（system prompt，英文版）

```
You are an ecological evidence auditor. You receive a question about a river site, a satellite image when one is
supplied, and numbered evidence items (witnesses). Answer the multiple-choice question and cite the witnesses by ID.
Reply with exactly these lines and nothing else:
answer: <A, B, C, D, or none>
support: [IDs of every witness your answer relies on]
load_bearing: [IDs whose removal alone would change the answer]
Use "none" only when no option is consistent with the evidence. When a revision of the evidence is announced,
re-evaluate the same question and put a first line "changed: yes" or "changed: no" before the three lines above.
```

---

## For English readers

Each record is a chat conversation about a Chinese river site (satellite image, eDNA fish list, site record, species knowledge). The model answers a 4-option question and must cite the numbered evidence items it relied on (`support`) and the ones whose removal alone would change the answer (`load_bearing`). `revise` records add one evidence revision after the model's (given) previous reply; the model must add `changed: yes|no` and may answer `none` when no option fits any more. `chain` records are 2–5 successive revisions that must be run with the model's own previous replies as history.

Run with `python -m ecowitness.evaluate_api --model <name> --data_dir . --workspace . --split test --langs en zh --modes base revise chain --out runs/<name>` (OpenAI-compatible API) or `python -m ecowitness.evaluate` (local HF model), check on `--split dev` first (it has answers), and send back `runs/<name>/predictions.jsonl` with the model name, thinking on/off, temperature and max tokens. Do not use the previous reply shown in `revise` records to answer `base` items; do not train on the test set.
