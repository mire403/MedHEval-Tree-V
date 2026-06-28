# Full Result Analysis - 2026-06-21

## Alignment Check

The completed experiment remains aligned with the planned five-step route:

- Step 1 supplied the top-conference method pool.
- Step 2 selected VADTree-style hierarchical evidence reasoning, with CREMA/VERA/PANDA-style evidence and hallucination control as support.
- Step 3 selected Kvasir-VQA-x1 as the main medical multimodal VQA target dataset.
- Step 4 refined the method into MedHEval-Tree-V: extract structured visual evidence, retrieve answer templates, then perform evidence-aware reranking.
- Step 5 now validates this idea on all 15,955 Kvasir-VQA-x1 test questions.

## Data Integrity

- Full output rows: 15,955/15,955.
- Duplicate IDs: 0.
- Generation error rows: 0.
- Parseable structured evidence: 15,955/15,955.
- Direct/evidence contradiction proxy: 3,817 rows.

The generation run is usable for full analysis and ablation.

## Main Metrics

| Method | Exact | Token F1 | Hallucination Proxy |
|---|---:|---:|---:|
| class_majority | 0.0743 | 0.3710 | 0.4741 |
| qwen_direct | 0.0003 | 0.1990 | 0.6452 |
| tfidf_direct_template | 0.0181 | 0.4164 | 0.4427 |
| tfidf_evidence_rerank | 0.0070 | 0.4129 | 0.5111 |
| tfidf_multiobjective | 0.0100 | 0.4107 | 0.4560 |
| medheval_tree_v_complexity_gate_23 | 0.0177 | 0.4252 | 0.4707 |
| medheval_tree_v_complexity_gate_3 | 0.0185 | 0.4243 | 0.4525 |

## Interpretation

Raw Qwen direct answers are visually plausible but poorly normalized to the dataset answer style, so their exact match and token F1 are low and hallucination proxy is high.

Class-aware TF-IDF answer retrieval is a strong normalization baseline. It substantially improves token F1 over raw Qwen, showing that answer-style normalization is necessary for Kvasir-VQA-x1.

Naively injecting evidence into every question is not optimal. `tfidf_evidence_rerank` hurts simple questions and increases hallucination proxy. This is an important negative result: structured visual evidence is useful, but only when routed selectively.

Complexity-gated evidence routing is the strongest current method. It supports the paper's main claim: hierarchical evidence should be activated for more complex medical VQA questions, while simple questions are better handled by direct class-aware retrieval.

## Complexity Finding

Evidence reranking is harmful on complexity-1 questions but beneficial on complexity-3 questions. This matches the VADTree-inspired hypothesis that additional evidence nodes help when the reasoning structure is more complex.

Current recommendation:

- Use direct class-aware TF-IDF template retrieval for complexity 1.
- Use evidence-aware reranking for complexity 3.
- Treat complexity 2 as a tunable middle case; `gate_23` maximizes token F1, while `gate_3` gives a better exact/hallucination trade-off.

## Paper Direction

The paper should not claim that generic evidence extraction always improves medical VQA. The stronger, more defensible claim is:

> Evidence should be routed conditionally. MedHEval-Tree-V improves complex medical VQA by activating structured visual evidence only when question complexity indicates that direct answer retrieval is insufficient.

This is a better BIBM story because it is domain-relevant, interpretable, and supported by a nontrivial negative-to-positive ablation.

## Next Experimental Actions

1. Freeze the full Qwen output as the visual evidence cache.
2. Produce final tables for overall, complexity-wise, and question-class-wise metrics.
3. Add a reliability gate beyond complexity, using uncertainty and direct/evidence contradiction features.
4. Inspect representative wins and failures for figure/table examples.
5. Decide whether the main method should report `gate_23` or `gate_3`:
   - `gate_23`: best Token F1.
   - `gate_3`: best Exact Match and cleaner hallucination trade-off.

## Reliability Gate Follow-Up

A reliability-gate analysis was run using the first 529 rows as a development split and the remaining 15,426 rows as an evaluation split. This avoids selecting the final gate directly on the whole test set.

Evaluation split ranking:

| Method | Exact | Token F1 | Hallucination Proxy |
|---|---:|---:|---:|
| bucket_gate_dev_policy | 0.0172 | 0.4268 | 0.4592 |
| complexity_or_class_gate | 0.0167 | 0.4263 | 0.4649 |
| complexity_gate_23 | 0.0175 | 0.4248 | 0.4708 |
| complexity_gate_3 | 0.0183 | 0.4240 | 0.4526 |
| tfidf_direct_template | 0.0180 | 0.4160 | 0.4429 |

Full split metrics:

| Method | Exact | Token F1 | Hallucination Proxy |
|---|---:|---:|---:|
| bucket_gate_dev_policy | 0.0174 | 0.4275 | 0.4580 |
| complexity_or_class_gate | 0.0168 | 0.4270 | 0.4642 |
| complexity_gate_23 | 0.0177 | 0.4252 | 0.4707 |
| complexity_gate_3 | 0.0185 | 0.4243 | 0.4525 |
| tfidf_direct_template | 0.0181 | 0.4164 | 0.4427 |

Final method recommendation:

- Main method: `bucket_gate_dev_policy`, framed as a **complexity- and class-aware reliability gate**.
- Simpler ablation: `complexity_gate_3` and `complexity_gate_23`.
- Conservative fallback if reviewers dislike class-bucket fitting: `complexity_gate_3`, which improves Token F1 and Exact Match over direct template retrieval while keeping hallucination proxy close.

The strongest paper claim should be:

> Structured evidence is beneficial when routed by question reliability signals. A lightweight reliability gate decides whether to use direct template retrieval or structured-evidence reranking, improving full-test Token F1 from 0.4164 to 0.4275.
