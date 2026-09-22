# Phase 0B2：纯文本效用预测结果

文本模型不接收未来记录，也不接收模拟器中的结构化真值字段。

| 测试集 | Random | FIFO | Lexical | Text | 结构化基线 | Oracle | Text 恢复比例 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| id | 0.092 | 0.100 | 0.483 | 0.992 | 0.958 | 1.000 | 99.1% | 通过 |
| ood_length_paraphrase | 0.042 | 0.033 | 0.458 | 0.333 | 0.908 | 1.000 | 31.0% | 失败 |
| paraphrase | 0.108 | 0.125 | 0.475 | 0.333 | 0.933 | 1.000 | 23.8% | 失败 |

## 分任务成功率

| 测试集/任务 | FIFO | Text | 结构化基线 | Oracle |
|---|---:|---:|---:|---:|
| id/delayed_query | 0.050 | 1.000 | 0.800 | 1.000 |
| id/state_overwrite | 0.400 | 0.950 | 1.000 | 1.000 |
| id/long_instruction | 0.000 | 1.000 | 1.000 | 1.000 |
| id/completed_intermediate | 0.050 | 1.000 | 1.000 | 1.000 |
| id/unresolved_subgoal | 0.000 | 1.000 | 1.000 | 1.000 |
| id/relation_chain | 0.100 | 1.000 | 0.950 | 1.000 |
| ood_length_paraphrase/delayed_query | 0.000 | 0.100 | 0.500 | 1.000 |
| ood_length_paraphrase/state_overwrite | 0.200 | 0.600 | 1.000 | 1.000 |
| ood_length_paraphrase/long_instruction | 0.000 | 0.350 | 1.000 | 1.000 |
| ood_length_paraphrase/completed_intermediate | 0.000 | 0.600 | 1.000 | 1.000 |
| ood_length_paraphrase/unresolved_subgoal | 0.000 | 0.050 | 1.000 | 1.000 |
| ood_length_paraphrase/relation_chain | 0.000 | 0.300 | 0.950 | 1.000 |
| paraphrase/delayed_query | 0.050 | 0.000 | 0.650 | 1.000 |
| paraphrase/state_overwrite | 0.500 | 0.650 | 1.000 | 1.000 |
| paraphrase/long_instruction | 0.000 | 0.300 | 1.000 | 1.000 |
| paraphrase/completed_intermediate | 0.000 | 0.550 | 1.000 | 1.000 |
| paraphrase/unresolved_subgoal | 0.000 | 0.100 | 1.000 | 1.000 |
| paraphrase/relation_chain | 0.200 | 0.400 | 0.950 | 1.000 |
