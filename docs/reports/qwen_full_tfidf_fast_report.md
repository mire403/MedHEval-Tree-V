# Full TF-IDF Fast Evaluation

- Input: `step5_experiments/results/qwen_full_test_outputs.jsonl` (not included; regenerate with the Qwen inference scripts)
- Rows: 15955
- Candidate K per class: 200
- Retrieval K: 40

## Metrics

| Method | N | Exact Match | Token F1 | Hallucination Proxy |
|---|---:|---:|---:|---:|
| class_majority | 15955 | 0.0743 | 0.3710 | 0.4741 |
| qwen_direct | 15955 | 0.0003 | 0.1990 | 0.6452 |
| tfidf_direct_template | 15955 | 0.0181 | 0.4164 | 0.4427 |
| tfidf_evidence_rerank | 15955 | 0.0070 | 0.4129 | 0.5111 |
| tfidf_multiobjective | 15955 | 0.0100 | 0.4107 | 0.4560 |
| medheval_tree_v_complexity_gate_23 | 15955 | 0.0177 | 0.4252 | 0.4707 |
| medheval_tree_v_complexity_gate_3 | 15955 | 0.0185 | 0.4243 | 0.4525 |

## Evidence JSON

- Parseable evidence: 15955/15955 (100.00%)
- Direct/evidence contradiction proxy: 3817
- Uncertainty distribution: `{"low": 14777, "moderate": 138, "no": 992, "yes": 15, "unknown": 15, "0.0": 6, "high": 12}`
- Unknown fields: `{"location": 26, "lesion_type": 19, "uncertainty": 15, "lesion_count": 1}`
