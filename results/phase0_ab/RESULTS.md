# Phase 0A/0B symbolic validation

## Decision summary

| Split | FIFO | Oracle | Predicted | Oracle−FIFO | Gap recovered | A | B |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| id | 0.105 | 1.000 | 0.943 | 0.895±0.012 | 93.6%±1.8% | PASS | PASS |
| ood_length | 0.052 | 1.000 | 0.927 | 0.948±0.014 | 92.3%±1.9% | PASS | PASS |

A requires Oracle−FIFO >= 0.10 and a >= 0.05 gap on at least four task families. B requires the predictor to recover >= 50% of the aggregate gap.
The oracle alone sees future counterfactual labels; predicted utility receives only decision-time features.

## Success by task family

| Split/task | FIFO | Oracle | Predicted | Oracle−FIFO |
|---|---:|---:|---:|---:|
| id/delayed_query | 0.072 | 1.000 | 0.660 | 0.927 |
| id/state_overwrite | 0.443 | 1.000 | 0.998 | 0.557 |
| id/long_instruction | 0.000 | 1.000 | 1.000 | 1.000 |
| id/completed_intermediate | 0.058 | 1.000 | 1.000 | 0.943 |
| id/unresolved_subgoal | 0.010 | 1.000 | 1.000 | 0.990 |
| id/relation_chain | 0.045 | 1.000 | 1.000 | 0.955 |
| ood_length/delayed_query | 0.020 | 1.000 | 0.560 | 0.980 |
| ood_length/state_overwrite | 0.247 | 1.000 | 1.000 | 0.752 |
| ood_length/long_instruction | 0.000 | 1.000 | 1.000 | 1.000 |
| ood_length/completed_intermediate | 0.035 | 1.000 | 1.000 | 0.965 |
| ood_length/unresolved_subgoal | 0.003 | 1.000 | 1.000 | 0.998 |
| ood_length/relation_chain | 0.007 | 1.000 | 1.000 | 0.993 |
