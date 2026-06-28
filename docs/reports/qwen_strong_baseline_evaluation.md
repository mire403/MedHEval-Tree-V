# Strong Qwen Prompt Baseline Evaluation

Rows evaluated: 15955.

## Overall

| method | n | errors | empty | exact | token_f1 | lexical_drift | avg_len |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bucket_gate_dev_policy_full | 15955 | 0 | 0 | 0.0174 | 0.4275 | 0.4580 |  |
| bucket_gate_dev_policy_eval | 15426 | 0 | 0 | 0.0172 | 0.4268 | 0.4592 |  |
| tfidf_direct_template | 15955 | 0 | 0 | 0.0181 | 0.4164 | 0.4237 | 8.3831 |
| tfidf_evidence_rerank | 15955 | 0 | 0 | 0.0070 | 0.4129 | 0.4819 | 9.7530 |
| qwen_evidence_then_answer | 15955 | 0 | 0 | 0.0005 | 0.2648 | 0.6338 | 7.7180 |
| qwen_direct_original | 15955 | 0 | 0 | 0.0003 | 0.1990 | 0.6424 | 6.3322 |
| qwen_constrained_answer | 15955 | 0 | 0 | 0.0000 | 0.1602 | 0.6371 | 3.3081 |
| qwen_class_constrained_answer | 15955 | 0 | 0 | 0.0000 | 0.1049 | 0.7026 | 2.3320 |

## Strong Prompt Baselines by Complexity

| method | complexity | n | exact | token_f1 | lexical_drift | avg_len |
| --- | --- | --- | --- | --- | --- | --- |
| qwen_constrained_answer | 1 | 5496 | 0.0000 | 0.1563 | 0.6135 | 1.3410 |
| qwen_constrained_answer | 2 | 5251 | 0.0000 | 0.1521 | 0.6444 | 3.5033 |
| qwen_constrained_answer | 3 | 5208 | 0.0000 | 0.1726 | 0.6546 | 5.1870 |
| qwen_class_constrained_answer | 1 | 5496 | 0.0000 | 0.1152 | 0.6958 | 1.1301 |
| qwen_class_constrained_answer | 2 | 5251 | 0.0000 | 0.1016 | 0.6492 | 2.4572 |
| qwen_class_constrained_answer | 3 | 5208 | 0.0000 | 0.0973 | 0.7636 | 3.4741 |
| qwen_evidence_then_answer | 1 | 5496 | 0.0000 | 0.1495 | 0.6496 | 1.7949 |
| qwen_evidence_then_answer | 2 | 5251 | 0.0002 | 0.2905 | 0.6785 | 9.4344 |
| qwen_evidence_then_answer | 3 | 5208 | 0.0013 | 0.3606 | 0.5722 | 12.2379 |
