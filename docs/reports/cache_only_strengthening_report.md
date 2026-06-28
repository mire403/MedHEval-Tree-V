# Cache-Only Experiment Strengthening Report

This report uses the frozen full-test prediction cache only. No MLLM generation or template-bank rebuilding is performed. Token F1 and exact match for aligned direct/evidence routes use the stored per-row metrics from the original full run. Lexical drift values are recomputed here as a diagnostic and should be treated as local relative diagnostics rather than replacements for the paper tables.

## Fixed First-529 Metadata Ablation

| method | n | exact | token_f1 | lexical_drift | evidence_route_rate | policy_entries |
| --- | --- | --- | --- | --- | --- | --- |
| direct_template | 15426 | 0.0180 | 0.4160 | 0.4830 | 0.0000 | 0 |
| uniform_evidence | 15426 | 0.0071 | 0.4127 | 0.5688 | 1.0000 | 0 |
| global_policy | 15426 | 0.0180 | 0.4160 | 0.4830 | 0.0000 | 1 |
| complexity_policy | 15426 | 0.0175 | 0.4248 | 0.5123 | 0.6556 | 3 |
| class_policy | 15426 | 0.0167 | 0.4234 | 0.5133 | 0.5394 | 18 |
| bucket_policy | 15426 | 0.0172 | 0.4268 | 0.5054 | 0.4420 | 50 |

## Random Split Stability

| dev_size | method | runs | token_f1_mean | token_f1_std | exact_mean | lexical_drift_mean | evidence_route_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 529 | bucket_policy | 10 | 0.4247 | 0.0011 | 0.0180 | 0.4987 | 0.3691 |
| 529 | class_policy | 10 | 0.4224 | 0.0018 | 0.0163 | 0.5106 | 0.4312 |
| 529 | complexity_policy | 10 | 0.4243 | 0.0024 | 0.0178 | 0.5075 | 0.5570 |
| 1000 | bucket_policy | 10 | 0.4283 | 0.0009 | 0.0179 | 0.5008 | 0.4932 |
| 1000 | class_policy | 10 | 0.4226 | 0.0016 | 0.0164 | 0.5123 | 0.4468 |
| 1000 | complexity_policy | 10 | 0.4248 | 0.0005 | 0.0179 | 0.5060 | 0.5568 |

## Paired Bootstrap

| comparison | eval_n | observed_token_f1_gain | bootstrap_ci95_low | bootstrap_ci95_high | bootstrap_two_sided_p | bootstrap_samples |
| --- | --- | --- | --- | --- | --- | --- |
| bucket_gate_vs_direct_template | 15426 | 0.0108 | 0.0090 | 0.0125 | 0.0000 | 2000 |

## Evidence Shuffle Control

| method | runs | token_f1_mean | token_f1_std | exact_mean | lexical_drift_mean | evidence_route_rate_mean |
| --- | --- | --- | --- | --- | --- | --- |
| direct_template | 1 | 0.4160 | 0.0000 | 0.0180 | 0.4830 | 0.0000 |
| bucket_gate_aligned | 1 | 0.4268 | 0.0000 | 0.0172 | 0.5054 | 0.4420 |
| bucket_gate_shuffled_evidence | 5 | 0.3163 | 0.0007 | 0.0143 | 0.6820 | 0.4420 |

## Sample-Adaptive Calibrated Gate

| method | split | n | exact | token_f1 | lexical_drift | evidence_route_rate | threshold | dev_token_f1 | feature_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_calibrated_gate | first529_rest | 15426 | 0.0181 | 0.4264 | 0.4969 | 0.2049 | 0.3000 | 0.4399 | 30 |

| dev_size | runs | token_f1_mean | token_f1_std | exact_mean | lexical_drift_mean | evidence_route_rate_mean | threshold_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 529 | 10 | 0.4265 | 0.0013 | 0.0180 | 0.5003 | 0.2433 | 0.2900 |
| 1000 | 10 | 0.4260 | 0.0013 | 0.0178 | 0.5004 | 0.2213 | 0.2950 |

## Predicted Metadata Gate

| method | n | exact | token_f1 | lexical_drift | evidence_route_rate | complexity_acc | primary_class_acc | policy_entries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predicted_complexity_policy | 15426 | 0.0124 | 0.4184 | 0.5202 | 0.6595 | 0.6766 | 0.6570 | 3 |
| predicted_class_policy | 15426 | 0.0159 | 0.4199 | 0.5139 | 0.4475 | 0.6766 | 0.6570 | 18 |
| predicted_bucket_policy | 15426 | 0.0170 | 0.4252 | 0.5072 | 0.4904 | 0.6766 | 0.6570 | 50 |

## Minimum Bucket Count Sensitivity

| method | min_count | n | exact | token_f1 | lexical_drift | evidence_route_rate | policy_entries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| class_policy | 1 | 15426 | 0.0167 | 0.4234 | 0.5133 | 0.5394 | 18 |
| bucket_policy | 1 | 15426 | 0.0178 | 0.4289 | 0.5089 | 0.5332 | 50 |
| class_policy | 3 | 15426 | 0.0167 | 0.4234 | 0.5133 | 0.5394 | 18 |
| bucket_policy | 3 | 15426 | 0.0178 | 0.4289 | 0.5089 | 0.5332 | 50 |
| class_policy | 5 | 15426 | 0.0167 | 0.4234 | 0.5133 | 0.5394 | 18 |
| bucket_policy | 5 | 15426 | 0.0176 | 0.4286 | 0.5076 | 0.5042 | 50 |
| class_policy | 8 | 15426 | 0.0167 | 0.4234 | 0.5133 | 0.5394 | 18 |
| bucket_policy | 8 | 15426 | 0.0172 | 0.4268 | 0.5054 | 0.4420 | 50 |
| class_policy | 12 | 15426 | 0.0167 | 0.4234 | 0.5133 | 0.5394 | 18 |
| bucket_policy | 12 | 15426 | 0.0170 | 0.4241 | 0.4897 | 0.2795 | 50 |
| class_policy | 20 | 15426 | 0.0169 | 0.4236 | 0.5058 | 0.4368 | 18 |
| bucket_policy | 20 | 15426 | 0.0180 | 0.4160 | 0.4830 | 0.0000 | 50 |
