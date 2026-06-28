# Reliability-Calibrated Evidence Tree Routing V2

This cache-only experiment upgrades the bucket lookup gate into a sample-adaptive reliability estimator. The feature set combines a coarse bucket prior with fine evidence-tree compatibility signals computed from lesion presence, lesion type, count, location, instrument visibility, text overlay, abnormality, and uncertainty fields. Gold answers are used only to define route-supervision labels and threshold selection on the development prefix; evaluation features use only question metadata, predictions, and structured evidence.

## First-529 Development Prefix

| method | n | exact | token_f1 | lexical_drift | evidence_route_rate | threshold | dev_token_f1 | feature_count | mean_route_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_template | 15426 | 0.0180 | 0.4160 | 0.4231 | 0.0000 |  |  |  |  |
| uniform_evidence_rerank | 15426 | 0.0071 | 0.4127 | 0.4809 | 1.0000 |  |  |  |  |
| bucket_gate | 15426 | 0.0172 | 0.4268 | 0.4336 | 0.4420 |  |  |  |  |
| rcetr_no_tree | 15426 | 0.0176 | 0.4296 | 0.4378 | 0.3429 | 0.2500 | 0.4500 | 40 | 0.2213 |
| rcetr_no_bucket | 15426 | 0.0182 | 0.4283 | 0.4408 | 0.3249 | 0.2500 | 0.4446 | 40 | 0.2214 |
| rcetr_v2 | 15426 | 0.0177 | 0.4293 | 0.4327 | 0.2338 | 0.3000 | 0.4501 | 40 | 0.2229 |

## Random Development Prefixes

| dev_size | runs | token_f1_mean | token_f1_std | exact_mean | lexical_drift_mean | evidence_route_rate_mean | threshold_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 529 | 10 | 0.4298 | 0.0016 | 0.0179 | 0.4344 | 0.2915 | 0.2650 |
| 1000 | 10 | 0.4304 | 0.0015 | 0.0181 | 0.4354 | 0.3157 | 0.2550 |

## Algorithmic Interpretation

- The original bucket gate acts as a coarse reliability prior.
- Evidence-tree compatibility supplies a fine per-sample signal over structured evidence fields.
- A calibrated logistic router selects between direct-template and evidence-reranked answers.
- This mirrors a coarse-to-fine routing pattern: do not activate evidence merely because it exists; activate it when the current sample's evidence tree is compatible and the bucket prior supports evidence use.
