# Phase 0B2.1 数据审计

- preset：`audit`
- seed：`0`
- schema：`1`
- episode 与动态实体已经按 split 隔离；
- 每个决策组已经执行因果边界、候选数量、标签和输入字段检查。

| split | episode | 决策组 | 有信息决策 | 正效用记录 | 模板 | 查询可见性 | SHA-256 |
|---|---:|---:|---:|---:|---|---|---|
| train | 100 | 1200 | 36.2% | 5.2% | A:33, B:34, C:33 | announced_query:95, hidden_query:5 | `cd98ef1d18b1` |
| validation | 24 | 288 | 37.8% | 6.2% | A:8, B:8, C:8 | announced_query:21, hidden_query:3 | `d85a66867ae5` |
| test_id | 24 | 288 | 37.5% | 5.4% | A:8, B:8, C:8 | announced_query:23, hidden_query:1 | `936288fa7067` |
| test_composition | 24 | 288 | 33.3% | 5.1% | D:24 | announced_query:22, hidden_query:2 | `478e5e23f58b` |
| test_length | 24 | 288 | 25.0% | 3.7% | D:24 | announced_query:22, hidden_query:2 | `66b622eae81a` |
| test_semantic_stress | 24 | 288 | 34.4% | 4.7% | E:24 | announced_query:21, hidden_query:3 | `ccee9546653a` |
