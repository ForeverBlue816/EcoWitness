# EcoWitness：反事实等变训练与证据审核评测（Qwen3.5 + v9 河流生态基准）

把 v9 基准（v8 的出题规则 + 在实验室原始注释上重建的 DNA 证据，1,334 个 DNA 样点，另加 D6–D9 四个基于科和 reads 占比的多样性题族，共 31 个题族）的每道题改写成“证人协议”。
数据只用英文（`--langs en`，配置里 `langs: [en]`）；中文版本仍可由生成器产出，但不再训练和发布：每条证据有编号（IMG、R1、SP-01、ROW-01…），模型回答时必须给出
`answer / support / load_bearing` 三行；再由生成器对证据做反事实编辑，精确算出编辑后的真值。

**方法：反事实等变（counterfactual equivariance）。** 规则知道每一次编辑对真值的精确作用，训练就要求模型的答案分布按同样的方式变换：
真值不变的编辑 → 分布不变（对称 KL）；真值变成 y' 的编辑 → 原答案的概率必须下降（铰链）；选项置换 → 分布跟着置换（置换后的 KL）。
编辑沿生物学轴采样：**taxonomy**（同科近缘种替换、性状相同的异科替换）、**trait**（功能性状）、**abundance**（reads 占比跨/不跨阈值）、
**status**（受威胁等级、保护级别、原生/外来、检出频率、科），外加通用轴 witness（撤销/新增）、image、record、presentation（选项置换）。
这些编辑以**静默**方式给模型：编辑后的证据作为一道新题提问，没有修订通知，模型必须自己从证据里读出变化。
原来的隔离损失（base 与 keep_uncited 修订的对称 KL）是其中“不变编辑”的一个特例，仍然保留。
评测新增归因忠实度：模型为原题声称的 `load_bearing`，与静默删除该证人后答案是否真的改变做对比。

```
ecowitness/
  ecowitness/
    protocol.py        证人编号、中英文提示渲染、输出解析
    rules.py           31 个题族的结论规则重算、留一法承重分析、反事实编辑（含生态轴编辑、选项置换）
    build_revisions.py 生成 data/{split}.{lang}.jsonl（base / revise / chain / silent 四种记录）
    data.py            Dataset（base + 反事实伴随记录分组、token 预算）+ Collator（Qwen 模板、标签掩码、答案位置、置换索引）
    train.py           LoRA 微调 + 等变损失（HF Trainer）
    runner.py score.py 生成循环与打分（含 silent 块：不变率、翻转率、置换等变率、归因忠实度）
    evaluate.py        本地模型评测
    evaluate_api.py    同一协议走 OpenAI 兼容 API（GPT / Gemini / Qwen-Max / DeepSeek 基线）
  configs/             equivariance（主方法）equivariance_ce_only  equivariance_generic  base（旧方法：SFT + 隔离损失）no_isolation  answer_only  base_questions_only
  scripts/             setup_env.sh  download_model.sh  train.slurm  eval.slurm  run_all.sh  smoke_test.sh  pack_for_cluster.ps1
  data/                由 build_revisions 生成（本地生成后随包上传）
```

## 一、本地生成数据（Windows，本目录）

```
cd C:\Users\yanlong.chen\Desktop\数据集\ecowitness
%USERPROFILE%\.local\bin\python3.14.exe -X utf8 -m ecowitness.build_revisions --splits train val test --out data
```
依赖同级目录的 `qualitative_benchmark_20261105_v9/`（含 tools、reports/answer_evidence.jsonl 与 reports/dna_evidence.json）
和 `Allsample_features_outputs/`。约 10 分钟。设 `EW_BENCH=qualitative_benchmark_20261026_v8` 可改回 v8（v8 用 v5 的 DNA 证据）。
生成器先用规则把基准三个 split 的全部答案重算一遍并断言一致，再做修订；统计写入 `data/stats.json`。

记录格式（每行一条）：`mode` 为 `base`（原题，目标含 support / load_bearing）、`revise`（原题 + 模型原答案 + 一条修订通知，
目标多一行 `changed`）、`chain`（val/test，2–5 步连续修订，`steps` 内逐步给出通知与真值）、`silent`（编辑后的证据作为新题，无通知）。
`edit_class` 把单步修订分成 `flip`（答案应变）、`keep_cited`（改动了被引用的冗余证人，答案不变）、`keep_uncited`（改动无关证人）。
silent 记录另带 `axis`（taxonomy / trait / abundance / status / witness / image / record / presentation）、`effect`（invariant / flip / equivariant）、
`base_answer`，置换记录带 `perm`（旧字母 → 新字母）；taxonomy 的 `edit.relation` 标明 congener_same_profile（同科同生态画像）、
congener_other_profile、other_family_same_profile、random。每题默认 2 条 silent（1 不变 + 1 翻转，生态轴优先）；train 一半的题再加一条置换，
val/test 每题都加，并另加 2 条删证人探针（1 条承重、1 条非承重），用于归因忠实度。base / revise / chain 记录与加入 silent 之前逐字节相同（只多了 `axis` 标签）。
默认 train 不含 C1（组合泛化留作测试）。

## 二、Leonardo 上的目录约定（与 PhysioWave-T/docs/cineca_setup.md 一致）

| 位置 | 内容 |
|---|---|
| `$HOME/ecowitness` | 代码（有备份，50 GB 配额） |
| `$HOME/ecowitness_env` | 虚拟环境，建在 cineca-ai 模块之上 |
| `$SCRATCH/ecowitness/data` | `{train,val,test}.{en,zh}.jsonl` |
| `$SCRATCH/ecowitness/satellite_rgb` | 影像 |
| `$SCRATCH/ecowitness/models` | Qwen3.5-9B 权重、tiny 调试模型 |
| `$SCRATCH/ecowitness/runs` | 训练输出与评测结果 |

`$SCRATCH` 的文件创建 40 天后会被清理，模型和数据都可以重建，但 adapter 和 metrics 要及时拷回 `$HOME` 或 `$FAST`。
所有路径集中在 `scripts/cineca_env.sh`，账号默认 `iscrb_wearusfm`，分区 `boost_usr_prod`，可用环境变量覆盖。

代码走 git，数据走 tar 包。

数据打包（本机，约 500 MB）：
```
powershell -ExecutionPolicy Bypass -File scripts\pack_for_cluster.ps1
scp ..\ecowitness_data.tar.gz ychen003@login.leonardo.cineca.it:/leonardo_scratch/large/userexternal/ychen003/
```
集群上：
```
cd ~ && git clone https://github.com/ForeverBlue816/EcoWitness.git ecowitness   # 以后更新只需 cd ~/ecowitness && git pull
mkdir -p $SCRATCH/ecowitness && tar -xzf $SCRATCH/ecowitness_data.tar.gz -C $SCRATCH/ecowitness
ls $SCRATCH/ecowitness        # 应看到 data/ 和 satellite_rgb/
```

## 三、环境与模型（登录节点，有网络）

```
cd ~/ecowitness
bash scripts/setup_env.sh        # 先 module load profile/deeplrn cineca-ai/4.3.0，再在其上建 venv，最后装缺的包
bash scripts/download_model.sh   # Qwen/Qwen3.5-9B -> $SCRATCH/ecowitness/models/Qwen3.5-9B（约 19 GB），另下 8 MB 的 tiny 调试模型
bash scripts/smoke_test.sh       # tiny 模型在登录节点 CPU 上跑通数据、LoRA、3 步训练和评测，约两分钟
```
`setup_env.sh` 的顺序和原则：模块先加载，再 `python -m venv $HOME/ecowitness_env --system-site-packages`。
cineca-ai 模块会把自己的 site-packages 写进 `PYTHONPATH`，优先级高于 venv，所以脚本（以及 `cineca_env.sh`）激活后把 venv 的
site-packages 放到 `PYTHONPATH` 最前面，并且只用 `python -m pip`、不升级 pip。
transformers 5（Qwen3.5 需要 ≥ 5.3）要求 torch ≥ 2.6；cineca-ai/4.3.0 的 torch 是 2.2，因此脚本会往 venv 装
`torch==2.7.1 torchvision==0.22.1`（cu126 轮子，约 3 GB，可用 `EW_TORCH_SPEC` / `EW_TORCH_INDEX` 覆盖）；模块 torch 够新时则沿用模块的。
transformers 连同它自己的依赖一起装（升级 regex、tokenizers 等），`peft`、`accelerate` 用 `--no-deps`。
装完断言：torch ≥ 2.6、transformers 来自 venv 且未禁用 PyTorch、认识 `qwen3_5`。
Qwen3.5 要点：架构 `Qwen3_5ForConditionalGeneration`（Gated DeltaNet + 注意力混合、原生多模态），`AutoModelForImageTextToText` 加载；
默认思考模式，训练与推理都用 `enable_thinking=False` 的模板；`flash-linear-attention` 与 `causal-conv1d` 可选，装不上就走慢一些的 torch 实现；
没有 torchvision 时代码自动退回慢速 PIL 图像处理器（256×256 影像无影响）。

## 四、一键启动

```
cd ~/ecowitness
bash scripts/run_all.sh            # 主实验（configs/equivariance.yaml）：训练(4×A100, 24h 上限) -> 测试集评测，外加零样本基线评测
bash scripts/run_all.sh ablations  # configs/ 下每个配置各提交一对 train+eval
RUN=base CONFIG=configs/base.yaml bash scripts/run_all.sh      # 旧方法基线：SFT + 隔离损失
TRAIN_EXTRA="--max_items 2000 --epochs 1" RUN=trial bash scripts/run_all.sh   # 先做一次小规模试跑
```
其他变量：`EW_ACCOUNT`、`MODEL_DIR`、`DATA_DIR`、`RUNS_DIR`、`NGPU`、`EVAL_EXTRA`。
作业脚本用 `python -m torch.distributed.run` 启动（不用 torchrun 可执行文件），让各进程继承 venv 的包。
结果：`$RUNS_DIR/<run>/adapter/`（LoRA）与 `$RUNS_DIR/<run>/eval_test/metrics.json`、`predictions.jsonl`。

## 五、评测：生成与打分分开（runner.py / score.py）

`evaluate.py`（本地模型）和 `evaluate_api.py`（OpenAI 兼容 API）共用 `runner.py` 的生成循环，只写 `predictions.jsonl`；
记录带真值时随后自动打分出 `metrics.json`，不带真值（公开测试包）时出 `check.json`。也可以单独打分：
```
python -m ecowitness.score --data data/test.en.jsonl data/test.zh.jsonl --predictions runs/base/eval_test/predictions.jsonl
```
指标分六族，定义见 `score.py` 文件头，完整表见 `docs/testpack_README.md` 第 7 节：
- **silent**：静默反事实。按轴 / 效果的准确率；`invariance_rate`（不该变时没变）、`flip_rate` / `flip_accuracy`（该变时变了、变对了）、
  `permutation_equivariance`（字母跟着选项走）、`by_relation`（近缘种替换）、`attribution_faithfulness`（声称的承重 vs 删除后实测的反应）、
  `notice_dependence`（有通知的翻转准确率 − 静默翻转准确率，越大说明越依赖通知）。silent 的行为指标需要同一次运行里有 base 预测。
- **accuracy**：按组、题族、难度；自助法 95% 区间。
- **abstention**：`none` 的 P/R/F1、`false_none_rate`。
- **citation**：support / load_bearing 的 P/R/F1、`load_bearing_exact`、`sufficiency`（承重证人是否全在 support 里）、
  `invalid_citation_rate`（编造编号）、`faithful_accuracy`（答对且承重集合全对）。
- **revision**：按修订类别 / 动作的准确率，`changed_f1`，`contagion_rate`、`redundancy_overreaction_rate`、`inertia_rate`，
  `flip_accuracy` / `keep_accuracy`，`stale_citation_rate`（引用已撤销的证人），`self_contradiction_rate`。
- **chain**：按步准确率、末步准确率（+CI、按链长）、整链全对率、`drift`、`recovery_rate` / `slip_rate`。
- **cross_lingual**：同题中英文答案一致率。

生成选项：`--teacher_forced`（链用真值历史，只有带 target 的数据能用）、`--own_history`（修订题的先前回答换成模型自己的 base 回答，
此时 `changed` 以模型自己的先前答案为准，传染率 / 惯性率不再定义）。

合成基线（测试集，中英合并）：随机 base 准确率 0.253；“永远不改”在修订题上准确率 0.557 但惯性率 1.0；标准答案全部 1.0。

API 基线：
```
export OPENAI_API_KEY=sk-...; export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
python -m ecowitness.evaluate_api --model qwen3-vl-plus --split test --langs en --modes base revise --out runs/api_qwen --no_thinking
```

**公开测试包**：`python scripts/build_test_pack.py` 生成 `../ecowitness_testpack_v9/`（不含答案的 test、含答案的 dev、影像、运行与打分代码、
README）和 tar.gz。外部评测者回传 `predictions.jsonl`，在本目录用 `score.py` 对 `data/test.*.jsonl` 打分。

## 六、训练细节

- 底座冻结，LoRA（r=32）挂在语言模型全部投影层（q/k/v/o、Gated DeltaNet 的 in_proj*/out_proj、MLP）；`--lora_merger true` 可把视觉融合层也挂上。
- 损失 = 助手行的交叉熵 + `lambda_iso` × 对称 KL（base 与每条不变伴随记录：silent invariant、keep_uncited 修订；置换伴随记录则先把 base 的分布按 `perm` 重排再比）
  + `lambda_flip` × 铰链 relu(log p_edit(y_old) − log p_base(y_old) + `flip_margin`)（silent flip：原答案在编辑后的证据下必须至少掉 `flip_margin` 个 nat）。
- 分组：Dataset 把一道 base 题和它的伴随记录放进同一组（`companions` 优先级：silent:invariant → silent:flip → silent:equivariant → revise:keep_uncited，
  `group_size` 默认 3），组的填充代价（成员数 × 最长成员的近似 token 数）超过 `group_budget`（8000）时伴随记录退为单独样本（只算交叉熵）。
- 图像 256×256，Qwen3.5 下只占 64 个视觉 token；单条对话一般 1–2k token；v9 最长的训练对话约 4.8k token（两个样点名录对比的 D3 题），`max_length` 设为 6144。
- 训练集：base 20,154 + revise 65,739 + silent（每题 2 条 + 一半的题 1 条置换）= 136,237 条对话（silent 50,344）；`equivariance.yaml` 跑 2 轮，计算量与 `base.yaml` 的 3 × 86k 相当。
  4×A100 64 GB、每卡 1 组、梯度累积 8，作业时限 24 小时；先用 `TRAIN_EXTRA="--max_items 2000 --epochs 1"` 试跑量一下真实速度。
- 消融：`equivariance_ce_only`（同样的数据，只有交叉熵）、`equivariance_generic`（去掉四个生态轴）、`base`（旧方法：只有 keep_uncited 隔离损失，无 silent）、
  `no_isolation`、`answer_only`、`base_questions_only`。
