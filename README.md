# MedHEval-Tree-V

Reliability-calibrated evidence-tree routing for medical visual question answering.

This repository contains the experiment code and aggregate results for **MedHEval-Tree-V**, a no-neural-fine-tuning framework for endoscopic medical VQA. The central idea is simple: structured visual evidence should not be blindly injected into every question. Instead, the system extracts a direct answer and a structured evidence tree from an MLLM, retrieves class-aware answer templates, scores evidence-template compatibility, and routes each sample through a calibrated direct or evidence-aware answer path.

![Main idea](paper_figures/Figure1.png)

## Project Summary

Medical VQA models often produce visually plausible but verbose free-form answers. Template normalization can improve benchmark-style answer matching, but uniform evidence fusion can hurt simple recognition questions. MedHEval-Tree-V studies this failure mode on Kvasir-VQA-x1 and implements a reliability-calibrated routing pipeline:

1. **Direct answer branch**: a frozen MLLM generates a concise direct answer.
2. **Structured evidence branch**: the same MLLM returns parseable evidence fields such as lesion presence, type, count, location, instrument visibility, abnormality, and uncertainty.
3. **Class-aware template retrieval**: training answers are grouped by question class; TF-IDF retrieves top-K candidate answer templates.
4. **Evidence-tree compatibility**: candidate answers are scored against field-level evidence compatibility.
5. **Reliability-calibrated routing**: a lightweight route estimator chooses direct-template retrieval or evidence-aware reranking per sample.

The code is organized around the real experiment trajectory used in the paper: lightweight baselines, full Qwen inference, TF-IDF answer retrieval, cache-only robustness studies, RC-ETR routing, strong prompt baselines, and an external SLAKE stress test.

## What Is Included

This repository includes:

- Experiment scripts for Kvasir-VQA-x1 and SLAKE.
- Server-side Qwen inference scripts for direct answer and structured evidence extraction.
- Cache-only analysis scripts for random splits, bootstrap significance, metadata ablation, shuffle controls, sensitivity tests, calibrated routing, and RC-ETR.
- Aggregate CSV result tables from the real experiment runs.
- Paper figures used to explain the method and summarize results.
- Documentation for expected data layout, label usage, and reproduction commands.

This repository intentionally does **not** include:

- Raw medical images or dataset parquet files.
- Model weights.
- SSH keys, passwords, machine-specific paths, or server credentials.
- Full per-sample prediction dumps or Qwen JSONL outputs, because they can contain benchmark question text, gold answers, image identifiers, and model outputs.
- Synthetic what-if manuscript variants or synthetic result tables.

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   ├── data_schema.md
│   ├── open_source_scope.md
│   ├── reproduction.md
│   └── reports/
├── paper_figures/
│   ├── Figure1.png
│   ├── Figure2.png
│   ├── Figure3.png
│   ├── main_results.png
│   ├── tradeoff_scatter.png
│   ├── complexity_effect.png
│   └── complexity_routing_lines.png
├── step3_dataset_migration/raw/kvasir_vqa_x1/
│   └── README.md
├── data/slake/
│   └── README.md
├── examples/
│   ├── sample_qwen_cache.jsonl
│   └── sample_template_predictions.csv
└── step5_experiments/
    ├── scripts/
    ├── server_scripts/
    └── results/
```

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Qwen/MLLM inference, install a PyTorch build compatible with your CUDA runtime and set the model path:

```bash
export QWEN_MODEL_PATH=/path/to/Qwen3.5-9B
export CUDA_VISIBLE_DEVICES=0,1
```

The cache-only experiments do not require GPU inference once `qwen_full_test_outputs.jsonl` and `qwen_full_tfidf_fast_predictions.csv` have been generated.

## Data Preparation

### Kvasir-VQA-x1

Place the Kvasir-VQA-x1 parquet files in:

```text
step3_dataset_migration/raw/kvasir_vqa_x1/
├── train.parquet
└── test.parquet
```

The scripts expect at least these columns:

- `img_id`
- `question`
- `answer`
- `complexity`
- `question_class`
- an image URL or image path column, depending on the downloader/bundle script

See `docs/data_schema.md` for details.

### SLAKE

Place SLAKE files under:

```text
data/slake/
```

The SLAKE script supports `inspect`, `qwen`, and `eval` subcommands and can be adapted to local SLAKE file naming if needed.

## Main Components

### 1. Lightweight Kvasir Baselines

Runs non-MLLM baselines and early template experiments:

```bash
python step5_experiments/scripts/run_kvasir_vqa_lite_experiments.py
```

This produces template/majority baselines and helps verify that the Kvasir parquet files are readable.

### 2. Build a Qwen Server Bundle

Build a small sanity bundle:

```bash
python step5_experiments/scripts/build_server_sanity_bundle.py
```

Build the full test bundle:

```bash
python step5_experiments/scripts/build_full_server_bundle.py
```

If the bundle uses image URLs, download images on the GPU server:

```bash
python step5_experiments/server_scripts/download_kvasir_images.py \
  --bundle step5_experiments/server_bundle_full
```

### 3. Qwen Direct Answer and Structured Evidence Cache

Sanity run:

```bash
python step5_experiments/server_scripts/qwen_image_smoke.py
python step5_experiments/server_scripts/run_qwen_sanity.py \
  --bundle step5_experiments/server_bundle \
  --limit 20 \
  --out step5_experiments/results/qwen_sanity_outputs.jsonl
```

Full run:

```bash
python step5_experiments/server_scripts/run_qwen_sanity.py \
  --bundle step5_experiments/server_bundle_full \
  --limit 0 \
  --out step5_experiments/results/qwen_full_test_outputs.jsonl \
  --resume
```

Each JSONL row contains the benchmark metadata, a direct answer, a structured evidence object, parse status, and runtime metadata.

### 4. TF-IDF Template Retrieval

Evaluate class-aware direct retrieval and evidence-aware reranking:

```bash
python step5_experiments/scripts/evaluate_full_tfidf_fast.py \
  --input step5_experiments/results/qwen_full_test_outputs.jsonl \
  --metrics step5_experiments/results/qwen_full_tfidf_fast_metrics.csv \
  --report docs/reports/qwen_full_tfidf_fast_report.md \
  --predictions step5_experiments/results/qwen_full_tfidf_fast_predictions.csv
```

The full prediction CSV is not included in this repository, but aggregate metrics are included.

### 5. Reliability Gate and Cache-Only Strengthening

Analyze bucket gates:

```bash
python step5_experiments/scripts/analyze_reliability_gates.py \
  --predictions step5_experiments/results/qwen_full_tfidf_fast_predictions.csv \
  --qwen-jsonl step5_experiments/results/qwen_full_test_outputs.jsonl \
  --out step5_experiments/results/reliability_gate_metrics.csv \
  --report docs/reports/reliability_gate_analysis.md
```

Run cache-only robustness experiments:

```bash
python step5_experiments/server_scripts/cache_only_strengthening.py
```

This script covers repeated random splits, bootstrap significance, metadata ablation, predicted metadata, shuffle controls, min-count sensitivity, and calibrated gate summaries.

### 6. RC-ETR: Reliability-Calibrated Evidence-Tree Routing

Run the upgraded routing framework:

```bash
python step5_experiments/server_scripts/reliability_calibrated_tree_v2.py
```

This produces:

- `rcetr_v2_main.csv`
- `rcetr_v2_random_raw.csv`
- `rcetr_v2_random_summary.csv`

The released aggregate main table reports the held-out evaluation split and ablations for direct template, uniform evidence reranking, bucket gate, no-tree routing, no-bucket routing, and RC-ETR.

### 7. Strong Qwen Prompt Baselines

Run prompt-only baselines:

```bash
python step5_experiments/server_scripts/run_qwen_strong_baselines.py \
  --bundle step5_experiments/server_bundle_full \
  --out step5_experiments/results/qwen_strong_baselines.jsonl \
  --modes constrained,class,evidence \
  --resume
```

Evaluate them:

```bash
python step5_experiments/server_scripts/evaluate_qwen_strong_baselines.py
```

The full JSONL is not included, but aggregate metrics by method and complexity are included.

### 8. SLAKE External Stress Test

Inspect SLAKE:

```bash
python step5_experiments/server_scripts/slake_external_validation.py inspect \
  --data-dir data/slake
```

Run Qwen on SLAKE:

```bash
python step5_experiments/server_scripts/slake_external_validation.py qwen \
  --data-dir data/slake \
  --split test \
  --lang en \
  --out outputs/slake/results/slake_qwen_test_en.jsonl
```

Evaluate routing on SLAKE:

```bash
python step5_experiments/server_scripts/slake_external_validation.py eval \
  --data-dir data/slake \
  --out-dir outputs/slake/results \
  --lang en \
  --qwen-jsonl outputs/slake/results/slake_qwen_test_en.jsonl
```

SLAKE is used as an external stress test of the routing principle, not as a full endoscopic RC-ETR validation.

## Label and Field Usage

### `complexity`

The Kvasir-VQA-x1 complexity label is used as a routing and analysis signal. The scripts treat it as a discrete metadata value and use it to form development buckets and complexity-level ablations.

### `question_class`

`question_class` is expected to be a list or list-like string. The first class is used as the primary class in bucket policies. Class labels also restrict the answer-template bank:

```text
A(C) = answer templates whose training question class matches C
```

If no class-specific answer templates are available, the retrieval code falls back to global high-frequency templates.

### Structured Evidence Fields

The structured evidence branch expects a compact JSON object with fields such as:

- `lesion_presence`
- `lesion_type`
- `lesion_count`
- `location`
- `instrument_presence`
- `text_overlay_presence`
- `abnormality_presence`
- `uncertainty`
- `evidence_sentence`

The evidence-tree compatibility code maps these fields into interpretable compatibility signals for candidate answers.

### Route Labels

Route labels are not manual labels. They are derived on a development prefix: if evidence-aware retrieval beats direct-template retrieval under token F1 for a development bucket, that bucket is considered evidence-positive. The RC-ETR router then learns sample-level route reliability features and applies a calibrated threshold.

### Lexical Drift

Some legacy scripts and reports use the field name `hallucination_proxy`. In the paper narrative this was renamed to **lexical drift proxy** to avoid overclaiming clinical hallucination. The old column name is retained in some code paths for backward compatibility.

## Released Aggregate Results

The folder `step5_experiments/results/` contains aggregate CSVs only. Important files include:

- `qwen_full_tfidf_fast_metrics.csv`
- `reliability_gate_metrics.csv`
- `rcetr_v2_main.csv`
- `rcetr_v2_random_summary.csv`
- `cache_only_random_split_summary.csv`
- `cache_only_bootstrap.csv`
- `cache_only_metadata_ablation.csv`
- `cache_only_shuffle_summary.csv`
- `cache_only_calibrated_gate_summary.csv`
- `qwen_strong_baseline_metrics.csv`
- `slake_external_summary_en.csv`

These files are safe to publish as aggregate summaries. They are not a replacement for regenerating the full per-sample predictions when reproducing the paper.

## Figures

The `paper_figures/` folder contains the manuscript figures:

![Pipeline](paper_figures/Figure2.png)

![Routing mechanism](paper_figures/Figure3.png)

Additional result figures are included for convenience:

- `main_results.png`
- `tradeoff_scatter.png`
- `complexity_effect.png`
- `complexity_routing_lines.png`

## Reproducibility Notes

- The main Kvasir evaluation uses a 529-example development prefix and a held-out evaluation split.
- Full per-sample caches are not released. Re-run Qwen inference to regenerate them.
- The method does not fine-tune Qwen or train a new visual encoder.
- The route estimator is lightweight and operates on cached direct-answer, retrieval, metadata, and evidence-tree compatibility features.
- Exact numbers can vary slightly with model checkpoint revisions, preprocessing, image availability, and library versions.

## Responsible Use

This code is intended for research reproduction and analysis of medical VQA routing behavior. It is not a clinical decision support system. Structured evidence parseability does not imply clinically verified factual correctness.

## Citation

If you use this repository, please cite the corresponding MedHEval-Tree-V paper when available.
