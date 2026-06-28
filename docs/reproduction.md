# Reproduction Guide

This guide gives a practical reproduction order. The expensive MLLM steps can be run on a GPU server; the cache-only steps can run on CPU after full caches are generated.

## 1. Prepare Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export QWEN_MODEL_PATH=/path/to/Qwen3.5-9B
```

## 2. Prepare Kvasir Data

Put `train.parquet` and `test.parquet` under:

```text
step3_dataset_migration/raw/kvasir_vqa_x1/
```

Run:

```bash
python step5_experiments/scripts/run_kvasir_vqa_lite_experiments.py
```

to verify data loading and generate lightweight baselines.

## 3. Build and Run the Qwen Cache

Build a full bundle:

```bash
python step5_experiments/scripts/build_full_server_bundle.py
```

Download images if necessary:

```bash
python step5_experiments/server_scripts/download_kvasir_images.py \
  --bundle step5_experiments/server_bundle_full
```

Run Qwen:

```bash
python step5_experiments/server_scripts/run_qwen_sanity.py \
  --bundle step5_experiments/server_bundle_full \
  --out step5_experiments/results/qwen_full_test_outputs.jsonl \
  --limit 0 \
  --resume
```

## 4. Evaluate Template Retrieval

```bash
python step5_experiments/scripts/evaluate_full_tfidf_fast.py \
  --input step5_experiments/results/qwen_full_test_outputs.jsonl \
  --metrics step5_experiments/results/qwen_full_tfidf_fast_metrics.csv \
  --report docs/reports/qwen_full_tfidf_fast_report.md \
  --predictions step5_experiments/results/qwen_full_tfidf_fast_predictions.csv
```

## 5. Run Gate and RC-ETR Experiments

```bash
python step5_experiments/scripts/analyze_reliability_gates.py \
  --predictions step5_experiments/results/qwen_full_tfidf_fast_predictions.csv \
  --qwen-jsonl step5_experiments/results/qwen_full_test_outputs.jsonl \
  --out step5_experiments/results/reliability_gate_metrics.csv \
  --report docs/reports/reliability_gate_analysis.md
python step5_experiments/server_scripts/cache_only_strengthening.py
python step5_experiments/server_scripts/reliability_calibrated_tree_v2.py
```

## 6. Run Strong Prompt Baselines

```bash
python step5_experiments/server_scripts/run_qwen_strong_baselines.py \
  --bundle step5_experiments/server_bundle_full \
  --out step5_experiments/results/qwen_strong_baselines.jsonl \
  --modes constrained,class,evidence \
  --resume

python step5_experiments/server_scripts/evaluate_qwen_strong_baselines.py
```

## 7. Run SLAKE Stress Test

```bash
python step5_experiments/server_scripts/slake_external_validation.py inspect \
  --data-dir data/slake

python step5_experiments/server_scripts/slake_external_validation.py qwen \
  --data-dir data/slake \
  --split test \
  --lang en \
  --out outputs/slake/results/slake_qwen_test_en.jsonl

python step5_experiments/server_scripts/slake_external_validation.py eval \
  --data-dir data/slake \
  --out-dir outputs/slake/results \
  --lang en \
  --qwen-jsonl outputs/slake/results/slake_qwen_test_en.jsonl
```

## Expected Aggregate Files

After the full pipeline, the key aggregate files are:

```text
step5_experiments/results/qwen_full_tfidf_fast_metrics.csv
step5_experiments/results/reliability_gate_metrics.csv
step5_experiments/results/cache_only_random_split_summary.csv
step5_experiments/results/cache_only_bootstrap.csv
step5_experiments/results/cache_only_metadata_ablation.csv
step5_experiments/results/cache_only_shuffle_summary.csv
step5_experiments/results/cache_only_calibrated_gate_summary.csv
step5_experiments/results/rcetr_v2_main.csv
step5_experiments/results/rcetr_v2_random_summary.csv
step5_experiments/results/qwen_strong_baseline_metrics.csv
step5_experiments/results/slake_external_summary_en.csv
```
