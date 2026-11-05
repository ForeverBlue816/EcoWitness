# EcoWitness：证据修订训练与评测（Qwen3.5 + v9 河流生态基准）

把 v9 基准（v8 的出题规则 + 在实验室原始注释上重建的 DNA 证据，1,334 个 DNA 样点）的每道题改写成“证人协议”：每条证据有编号（IMG、R1、SP-01、ROW-01…），模型回答时必须给出
`answer / support / load_bearing` 三行；再由生成器对证据做反事实修订（撤销、新增、改判、改字段、换影像），
精确算出修订后的真值，训练并评测模型“分得清哪条证据承重、不被无关修订传染”。

```
ecowitness/
  ecowitness/
    protocol.py        证人编号、中英文提示渲染、输出解析
    rules.py           27 个题族的结论规则重算、留一法承重分析、反事实编辑
    build_revisions.py 生成 data/{split}.{lang}.jsonl（base / revise / chain 三种记录）
    data.py            Dataset（配对分组）+ Collator（Qwen 模板、标签掩码、答案位置）
    train.py           LoRA 微调 + 隔离损失（HF Trainer）
    evaluate.py        本地模型评测（修订 F1、传染率、引用精度、多步漂移）
    evaluate_api.py    同一协议走 OpenAI 兼容 API（GPT / Gemini / Qwen-Max 基线）
  configs/             base / no_isolation / answer_only / base_questions_only
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
目标多一行 `changed`）、`chain`（val/test，2–5 步连续修订，`steps` 内逐步给出通知与真值）。
`edit_class` 把单步修订分成 `flip`（答案应变）、`keep_cited`（改动了被引用的冗余证人，答案不变）、`keep_uncited`（改动无关证人）。
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
`setup_env.sh` 的顺序和原则：模块先加载，`python -m venv $HOME/ecowitness_env --system-site-packages` 复用模块里的 CUDA torch；
纯 Python 依赖正常安装；`transformers`、`peft`、`accelerate` 用 `--no-deps` 安装，避免 pip 拉一个自己的 torch 把模块版本盖掉；
装完会断言 `torch.__file__` 仍指向 cineca-ai，并检查这版 transformers 认识 `qwen3_5`。
Qwen3.5 要点：架构 `Qwen3_5ForConditionalGeneration`（Gated DeltaNet + 注意力混合、原生多模态），`AutoModelForImageTextToText` 加载；
默认思考模式，训练与推理都用 `enable_thinking=False` 的模板；`flash-linear-attention` 与 `causal-conv1d` 可选，装不上就走慢一些的 torch 实现；
没有 torchvision 时代码自动退回慢速 PIL 图像处理器（256×256 影像无影响）。

## 四、一键启动

```
cd ~/ecowitness
bash scripts/run_all.sh            # 主实验：训练(4×A100, 24h 上限) -> 测试集评测，外加零样本基线评测
bash scripts/run_all.sh ablations  # configs/ 下每个配置各提交一对 train+eval
RUN=no_iso CONFIG=configs/no_isolation.yaml bash scripts/run_all.sh
TRAIN_EXTRA="--max_items 2000 --epochs 1" RUN=trial bash scripts/run_all.sh   # 先做一次小规模试跑
```
其他变量：`EW_ACCOUNT`、`MODEL_DIR`、`DATA_DIR`、`RUNS_DIR`、`NGPU`、`EVAL_EXTRA`。
作业脚本用 `python -m torch.distributed.run` 启动（不用 torchrun 可执行文件），让各进程继承 venv 的包。
结果：`$RUNS_DIR/<run>/adapter/`（LoRA）与 `$RUNS_DIR/<run>/eval_test/metrics.json`、`predictions.jsonl`。

## 五、评测指标（evaluate.py）

- base：选项准确率（按组、题族、难度）、support / load_bearing 的精确率与召回率、承重集合完全匹配率。
- revise：修订 F1（changed 的 P/R/F1）、`contagion_rate`（keep_uncited 中被错误改答的比例）、
  `redundancy_overreaction_rate`（keep_cited 中错误改答）、`inertia_rate`（flip 中没改）、`none_accuracy`（应答 none 的题）。
- chain：按步准确率（默认用模型自己的历史，`--teacher_forced` 用真值历史）、最后一步准确率。

API 基线：
```
export OPENAI_API_KEY=sk-...; export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
python -m ecowitness.evaluate_api --model qwen3-vl-plus --split test --langs en --modes base revise --out runs/api_qwen --no_thinking
```

## 六、训练细节

- 底座冻结，LoRA（r=32）挂在语言模型全部投影层（q/k/v/o、Gated DeltaNet 的 in_proj*/out_proj、MLP）；`--lora_merger true` 可把视觉融合层也挂上。
- 损失 = 助手行的交叉熵 + `lambda_iso` × 对称 KL（base 与其 keep_uncited 修订在答案字母上的分布）。配对样本由 Dataset 分组、Collator 同批展开。
- 图像 256×256，Qwen3.5 下只占 64 个视觉 token；单条对话一般 1–2k token；v9 最长的训练对话约 4.8k token（两个样点名录对比的 D3 题），`max_length` 设为 6144。
- v9 训练集每种语言 72,498 条（v8 为 54,741）。4×A100 64 GB、每卡 1 组、梯度累积 8，一轮估计 3–4 小时（无融合核时更慢），作业时限 24 小时；先用 `TRAIN_EXTRA="--max_items 2000 --epochs 1"` 试跑量一下真实速度。
