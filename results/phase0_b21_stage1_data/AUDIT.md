# Phase 0B2.1 数据审计

- preset：`stage1`
- seed：`0`
- schema：`1`
- episode 与动态实体已经按 split 隔离；
- 每个决策组已经执行因果边界、候选数量、标签和输入字段检查。

| split | episode | 决策组 | 有信息决策 | 正效用记录 | 模板 | 查询可见性 | SHA-256 |
|---|---:|---:|---:|---:|---|---|---|
| train | 3000 | 36000 | 36.3% | 5.1% | A:1000, B:1000, C:1000 | announced_query:2764, hidden_query:236 | `68dc77b978d0` |
| validation | 400 | 4800 | 35.9% | 5.1% | A:134, B:133, C:133 | announced_query:368, hidden_query:32 | `ae9596aee2b6` |
| test_id | 600 | 7200 | 33.7% | 4.8% | A:200, B:200, C:200 | announced_query:549, hidden_query:51 | `bd41bd8f3bef` |
| test_composition | 600 | 7200 | 32.5% | 4.6% | D:600 | announced_query:548, hidden_query:52 | `113d46e63068` |
| test_length | 600 | 7200 | 26.2% | 3.8% | D:600 | announced_query:550, hidden_query:50 | `ecb6204a1fe7` |
| test_semantic_stress | 600 | 7200 | 33.3% | 4.7% | E:600 | announced_query:549, hidden_query:51 | `5ff61dd50eea` |
