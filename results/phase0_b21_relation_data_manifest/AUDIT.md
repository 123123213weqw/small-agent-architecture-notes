# Phase 0B2.1 数据审计

- preset：`stage1_relation_aug`
- seed：`0`
- schema：`1`
- episode 与动态实体已经按 split 隔离；
- 每个决策组已经执行因果边界、候选数量、标签和输入字段检查。

| split | episode | 决策组 | 有信息决策 | 正效用记录 | 模板 | 查询可见性 | SHA-256 |
|---|---:|---:|---:|---:|---|---|---|
| train | 3000 | 36000 | 36.3% | 5.1% | A:833, B:834, C:833, R0:63, R1:63, R2:63, R3:63, R4:62, R5:62, R6:62, R7:62 | announced_query:2764, hidden_query:236 | `c5a6dc412ef6` |
| validation | 400 | 4800 | 35.9% | 5.1% | A:112, B:111, C:111, R0:9, R1:9, R2:8, R3:8, R4:8, R5:8, R6:8, R7:8 | announced_query:368, hidden_query:32 | `e3fac370efd7` |
| test_id | 600 | 7200 | 33.7% | 4.8% | A:200, B:200, C:200 | announced_query:549, hidden_query:51 | `2959f5e219e2` |
| test_composition | 600 | 7200 | 32.5% | 4.6% | D:600 | announced_query:548, hidden_query:52 | `6575595dc597` |
| test_length | 600 | 7200 | 26.2% | 3.8% | D:600 | announced_query:550, hidden_query:50 | `e388c2b3d2d0` |
| test_semantic_stress | 600 | 7200 | 33.3% | 4.7% | E:600 | announced_query:549, hidden_query:51 | `e48eea935064` |
