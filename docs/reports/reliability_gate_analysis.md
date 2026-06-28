# Reliability Gate Analysis

- Predictions: `step5_experiments/results/qwen_full_tfidf_fast_predictions.csv` (not included; regenerate from Qwen outputs)
- Qwen JSONL: `step5_experiments/results/qwen_full_test_outputs.jsonl` (not included; regenerate with the Qwen inference scripts)
- Development rows: first 529
- Evaluation rows: 15426
- Dev-positive evidence classes: `["abnormality_color", "abnormality_location", "abnormality_presence", "instrument_count", "instrument_location", "instrument_presence", "landmark_location", "polyp_type", "text_presence"]`
- Bucket policy entries: 38

## Evaluation Split Ranking

| Method | N | Exact | Token F1 | Hallucination Proxy |
|---|---:|---:|---:|---:|
| bucket_gate_dev_policy | 15426 | 0.0172 | 0.4268 | 0.4592 |
| complexity_or_class_gate | 15426 | 0.0167 | 0.4263 | 0.4649 |
| complexity_gate_23 | 15426 | 0.0175 | 0.4248 | 0.4708 |
| complexity_gate_3 | 15426 | 0.0183 | 0.4240 | 0.4526 |
| complexity_gate_3_low_uncertainty | 15426 | 0.0183 | 0.4235 | 0.4519 |
| class_gate_dev_positive | 15426 | 0.0167 | 0.4234 | 0.4568 |
| complexity_gate_3_no_contradiction | 15426 | 0.0182 | 0.4210 | 0.4468 |
| tfidf_direct_template | 15426 | 0.0180 | 0.4160 | 0.4429 |
| tfidf_evidence_rerank | 15426 | 0.0071 | 0.4127 | 0.5112 |
| tfidf_multiobjective | 15426 | 0.0098 | 0.4103 | 0.4562 |

## Interpretation

- The split uses early rows only for gate selection and reports the remaining rows as the evaluation split.
- Complexity-only gates are more robust than heavily class-fitted bucket gates if their evaluation gains are comparable.
- The final paper method should prefer the simplest gate that improves Token F1 without a large hallucination-proxy penalty.
