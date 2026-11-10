# EcoWitness v10 test pack: evidence auditing and revision in river ecology

This pack measures whether a multimodal model can **answer from evidence, say which evidence it relied on, and update
(or refuse to update) its answer when the evidence is revised**. Questions come from Chinese river sites: Sentinel-2
satellite chips, eDNA fish lists with read shares (species-level detections from the laboratory read table), site records and species knowledge. Every question has four options;
after a revision no option may fit any more, and the correct answer is then `none`.

The test set ships without answers. Run your model, then return `predictions.jsonl` to the pack owner for scoring. The
`dev` set (= validation split) ships with answers so you can check your pipeline end to end.

## 1. Task

Each record is one conversation. Evidence items are numbered witnesses: `IMG` satellite image, `R1…` site-record
fields, `SP-01…` fish cards, `A-01 / B-01` cards of two sites, `ROW-01…` sequence-match rows. The reply has a fixed form:

```
answer: C
support: [SP-03, SP-07, IMG]
load_bearing: [SP-03]
```

- `support`: every witness the answer relies on.
- `load_bearing`: witnesses whose removal **alone** would change the answer.
- After a revision notice, a first line `changed: yes` or `changed: no` is added.

Four record kinds (`mode`):

| mode | content | generations per record |
|---|---|---|
| `base` | the question | 1 |
| `silent` | the same question with the evidence **silently edited** (a species reassigned to a relative, a trait, read share or conservation status changed, a witness removed or added, the image or a record field replaced, or only the options reordered); asked as a fresh question, no notice, no previous reply | 1 |
| `revise` | question + the model's previous reply (given) + one revision notice (withdraw, add, reassign, correct a field, replace the image) | 1 |
| `chain` | question + 2–5 successive revisions; each step is asked with **the model's own previous replies** in the history | 1 + number of steps |

`silent` records are scored against the model's own `base` reply to the same question (did the answer stay when it
should, move when it must, follow the options when they were reordered), so run `base` whenever you run `silent`.

31 question families in six groups (`group`): A habitat from imagery, B community trait reasoning from DNA cards,
C scene–community fusion and image timing, D diversity and conservation (threatened records, detection plausibility,
site comparison, rare species, shared species, family richness, read dominance, rare detections, dominance class),
E river network and connectivity, K species knowledge. `tier` is the difficulty (easy / medium / hard).

## 2. Files

| file | content |
|---|---|
| `test.public.en.jsonl` | test set, **no answers** |
| `dev.en.jsonl` | development set with answers and `edit_class` labels |
| `satellite_rgb/` | 256×256 Sentinel-2 RGB PNGs referenced by test and dev |
| `ecowitness/` | runner and scorer: `evaluate_api.py` (OpenAI-compatible API), `evaluate.py` (local HF model), `score.py` (metrics / format check), `runner.py`, `protocol.py` |
| `requirements_api.txt`, `requirements_local.txt` | dependencies for the two ways of running |
| `manifest.json` | file list with sha256 |

Sizes:

| | base | silent | revise | chain | chain steps | generations |
|---|---|---|---|---|---|---|
| test | 7,523 | 32,461 | 25,738 | 7,523 | 26,341 | 99,586 |
| dev | 3,449 | 14,917 | 12,039 | 3,449 | 12,200 | 46,054 |

Per question the test set has about four silent records: two edited-evidence versions (one that should not change the
answer, one that must), one with reordered options, and up to two with a witness removed.

A conversation is usually 1–2k tokens of text (the longest about 5k) plus one or two images. Two tracks:

- **minimal**: `--modes base silent revise` (no chains).
- **full**: `--modes base silent revise chain`.

## 3. Running

### A. OpenAI-compatible API (GPT, Gemini, Qwen, DashScope, vLLM, …)

```bash
pip install -r requirements_api.txt
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...          # e.g. https://dashscope.aliyuncs.com/compatible-mode/v1
python -m ecowitness.evaluate_api --model <model> --data_dir . --workspace . \
    --split test --langs en --modes base silent revise chain --workers 4 --out runs/<model>
```

- Images are sent as base64 PNG. Temperature defaults to 0.
- Thinking models: either `--no_thinking` (vLLM / DashScope honour it) or raise `--max_tokens` to 4096 or more, otherwise
  the reasoning uses up the budget before the answer. Anything before `</think>` is ignored when scoring.
- Replies are cached per conversation in `runs/<model>/cache.jsonl`; an interrupted run resumes.

### B. Local HuggingFace model (Qwen3.5 family, or anything `AutoModelForImageTextToText` loads)

```bash
pip install -r requirements_local.txt
python -m ecowitness.evaluate --model <path> --data_dir . --workspace . \
    --split test --langs en --modes base silent revise chain --batch_size 8 --out runs/<model>
```
Greedy decoding. Keep the default `--max_new_tokens 640`: the reply is three or four lines, but the citation lists can be long (a quarter of the
reference replies exceed 64 tokens, 2 % exceed 256, the longest is about 800). A reply cut off before its `load_bearing` line loses its citation
score. Add `--adapter <path>` for a LoRA.

### C. Your own inference stack

Produce `predictions.jsonl` in this format, one record per line:

```json
{"id": "Q10_..|en|base",  "lang": "en", "mode": "base",   "output": "answer: B\nsupport: [...]\nload_bearing: [...]"}
{"id": "Q10_..|en|s1",    "lang": "en", "mode": "silent", "output": "answer: B\nsupport: [...]\nload_bearing: [...]"}
{"id": "Q10_..|en|e2",    "lang": "en", "mode": "revise", "output": "changed: yes\nanswer: none\n..."}
{"id": "Q10_..|en|chain", "lang": "en", "mode": "chain",  "history": "own", "base_output": "...", "step_outputs": ["...", "..."]}
```

`messages` is the conversation to send: `role` is `system` / `user` / `assistant`; a `user` content is a list of
`{"type":"image","path":...}` (relative to this folder) and `{"type":"text","text":...}` items. For chains: generate on
`messages` to get `base_output`; then for each `steps[k]`, append `{"role":"assistant","content":<your previous reply>}`
and `steps[k]["message"]` to the conversation and generate again. **Use the model's own replies as history**, unedited.

## 4. Check the pipeline on dev first

```bash
python -m ecowitness.evaluate_api --model <model> --data_dir . --workspace . --split dev --langs en --max_items 50 --out runs/dev_check
```
dev has answers, so this writes `metrics.json`. Look at `parse_rate` (close to 1: the model follows the format),
`invalid_citation_rate` (close to 0: cited IDs exist) and `coverage` (predicted equals total).

## 5. Check and return

On the test set the run writes `check.json` (coverage, parse rate, invalid-citation rate, chain step counts). To check a
prediction file on its own:
```bash
python -m ecowitness.score --data test.public.en.jsonl --predictions runs/<model>/predictions.jsonl
```
Return `runs/<model>/predictions.jsonl` together with: model name and version, thinking on or off, temperature,
max tokens, and which modes you ran.

## 6. Rules

1. Answer `base` items from `base` records only. The previous assistant reply inside a `revise` record is part of the task
   (it is the correct original answer) and must not be used to answer `base` items.
2. Do not train or tune prompts on the test set; dev is for pipeline debugging.
3. Keep the system prompt and the evidence text unchanged; greedy decoding or temperature 0.
4. Chains must use the model's own history.

## 7. Metrics

`metrics.json` has one block per mode (base / revise / chain), an `ecology` block and a `scorecard` of headline numbers.

**base**

| metric | meaning |
|---|---|
| `accuracy` | option accuracy, also by group, family and tier; `accuracy_ci95` is a bootstrap 95 % interval |
| `parse_rate`, `full_format_rate` | share of replies with a parsable answer; share with all required lines |
| `citation.support` / `citation.load_bearing` | precision, recall, F1 of cited witness IDs against the truth sets |
| `load_bearing_exact` | load-bearing set exactly right |
| `sufficiency` | every truly load-bearing witness appears in the cited support (no key evidence overlooked) |
| `invalid_citation_rate` | cites an ID that does not exist in the conversation (fabricated citation) |
| `faithful_accuracy` | answer right **and** load-bearing set exactly right (right for the right reason) |
| `load_bearing_exact_given_correct` | among correct answers, share with the load-bearing set right |
| `abstention` | precision, recall, F1 of answering `none`; `false_none_rate` = answered `none` although an option was correct |

**revise** (single revision), in addition:

| metric | meaning |
|---|---|
| `by_edit.class:*` | accuracy and change rate per edit class: `flip` the answer should change, `keep_cited` a cited but redundant witness changed, `keep_uncited` an unrelated witness changed |
| `by_edit.type:*` | per edit action: remove / add / reassign / set_field / swap_image |
| `revision.changed_f1` | F1 of the `changed` line; `change_detection_accuracy` = `changed` line right |
| `revision.contagion_rate` | changed the answer although an unrelated witness changed (lower is better) |
| `revision.redundancy_overreaction_rate` | changed the answer although only a redundant witness changed (lower is better) |
| `revision.inertia_rate` | kept the answer although a load-bearing witness changed (lower is better) |
| `revision.flip_accuracy` / `keep_accuracy` | accuracy on should-change and should-keep items |
| `revision.stale_citation_rate` | still cites a witness the revision withdrew |
| `revision.self_contradiction_rate` | `changed` line disagrees with whether the answer actually changed |
| `revision.accuracy_truth_none` | accuracy on revisions whose correct answer is `none` |

**silent** (edited evidence asked afresh; compared with the model's own `base` reply)

| metric | meaning |
|---|---|
| `accuracy` | overall and by edit axis (`axis:taxonomy`, `axis:trait`, `axis:abundance`, `axis:status`, `axis:witness`, `axis:image`, `axis:record`), by effect and by family |
| `invariance_rate` | edits that should not matter (a congener with the same ecological profile, a trait of an unrelated species, a read share that stays on the same side of the threshold, an uncited witness …): the answer stayed the same |
| `flip_rate` / `flip_accuracy` | edits that must change the answer: the answer moved; the answer moved to the right option (or `none`) |
| `permutation_equivariance` | the options were reordered and the chosen letter followed its option |
| `by_axis` / `by_relation` | the three numbers above per axis, and per reassignment relation (`congener_same_profile`, `congener_other_profile`, `other_family_same_profile`, `random`) |
| `attribution_faithfulness` | for silent witness removals: agreement between what the model **stated** as load-bearing in its `base` reply and how its answer **actually reacted** when that witness was removed; precision / recall / F1 of the stated set as a predictor of the measured reaction |
| `notice_dependence` (scorecard) | flip accuracy with a revision notice minus flip accuracy on silent flips; large values mean the model follows notices rather than evidence |

**chain** (own history)

| metric | meaning |
|---|---|
| `accuracy_by_step` | accuracy at step 0 (the question) through step 5 |
| `final_step_accuracy` (+ CI, by chain length) | accuracy at the last step |
| `whole_chain_accuracy` | every step of the chain right |
| `drift` | step-0 accuracy minus last-step accuracy |
| `recovery_rate` / `slip_rate` | right after a wrong step; wrong after a right step |
| `step_accuracy_by_edit_class` | per-step accuracy and change rate by edit class |

**ecology** (base and revise items read through an ecological lens)

| metric | meaning |
|---|---|
| `competency_profile` | accuracy, faithful accuracy and load-bearing F1 per competency: habitat from imagery, community trait reasoning, scene–community fusion, diversity and conservation, river network and connectivity, species knowledge |
| `conservation_items` | accuracy on questions about threatened / protected / non-native / marine-stray / sea-route / native status |
| `read_abundance_items` | accuracy on questions that need read shares (dominance, rare detections, dominance class) |
| `species_counts` | for count questions: mean absolute error in species or families, bias, over- and under-count rates |
| `habitat_bins` | for ordered image classes (channel width, water fraction, sinuosity, riparian vegetation, built-up bank): exact, adjacent-bin and far-error rates |
| `modality_grounding` | image questions that cite the image; fusion questions that cite both sources |
| `revision_by_edited_witness` | accuracy, contagion and inertia split by what was revised: image, record field, species card, match row |

## 8. Reference numbers (test set, synthetic predictors)

| predictor | base acc | faithful | LB F1 | post-revision acc | changed F1 | contagion | inertia | stale cite | chain final |
|---|---|---|---|---|---|---|---|---|---|
| random option | 0.254 | 0.053 | 0.20 | 0.206 | 0.60 | 1.00 | 0.00 | 0.26 | 0.176 |
| always keep the previous reply | 0.244 | 0.052 | 0.20 | 0.574 | 0.00 | 0.00 | 1.00 | 0.73 | 0.173 |
| always `none` after a revision | 0.249 | 0.052 | 0.20 | 0.182 | 0.60 | 1.00 | 0.00 | 0.00 | 0.164 |
| gold answers | 1.000 | 1.000 | 1.00 | 1.000 | 1.00 | 0.00 | 0.00 | 0.00 | 1.000 |

Two reminders: 57.4 % of revision items should keep the previous answer, so post-revision accuracy must be read
together with inertia and `flip_accuracy`; `none` is the correct answer for 18.2 % of revision items.

## 9. The protocol (system prompt)

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
