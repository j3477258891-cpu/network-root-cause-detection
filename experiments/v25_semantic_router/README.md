# V25 Semantic Router

## Status

| 组件 | 状态 | 位置 |
|------|------|------|
| Qwen2.5-7B NPU 冒烟测试 | ✅ 通过 | 云端 Ascend 910B |
| DAPT (无监督领域适配) | ✅ 完成 | 云端 |
| 语义专家 fold=0 训练 | 🔄 进行中 | 云端 |
| Oracle 审计 | ✅ 完成 | 本地 |
| 单折评估脚本 | ✅ 就绪 | 本地 `eval/eval_fold0.py` |
| 动作路由器 | ✅ 就绪 | 本地 `router/action_router.py` |
| 近邻专家 | ⏳ 待语义专家产出 | — |
| 完整五折训练 | ⏳ 等 fold=0 gate 通过 | 云端 |

## Oracle 审计结论

```
Baseline: 2826 TP,  215 FN
Budget=2: 2991 TP (+165) ← gate FAILED (-5)
Budget=3: 3010 TP (+184) ← gate PASSED
Budget=4: 3025 TP (+199)
Budget=5: 3031 TP (+205)
```

- 所有 215 个 V11 FN 在理论上可达（在 top-8 候选内）
- 建议动作预算从 2 提到 3（已在路由器中实现）
- 语义专家的核心任务：把 215 个 FN 中的至少 125 个推到排序顶端

## 纯统计方案 (探索性，不推荐提交)

- `pure_statistical.py` — 零 ML 模型: 模板匹配 + 标题先验 + 共现 + 图距离
- 5 折 OOF: -89 TP vs V11 (0/5 折改善)
- 结论: 纯统计远不如 V11 ExtraTrees，ML 模型对捕捉非线性交互是必需的
- CSV: `submissions/result_record_pure_statistical_p1059.csv` (仅作概念参考)

## 文件夹结构

```
v25_semantic_router/
├── oracle/
│   └── oracle_audit.py      # 理论上限计算
├── eval/
│   ├── eval_fold0.py        # 单折探针评估
│   └── eval_pure_5fold.py   # 纯统计方案 5 折 OOF
├── router/
│   └── action_router.py     # ExtraTrees+HGB + DP 解码
├── outputs/
│   └── v25_oracle_report.json
├── submissions/
│   └── result_record_pure_statistical_p1059.csv
├── pure_statistical.py      # 纯统计方案主脚本
└── cloud_dataset/           # 云端数据集验证
```

## 下一步

1. 等云端 fold=0 产出 → 用 `eval_fold0.py` 评估
2. 若 fold=0 delta ≥ 0 且 FN fix rate ≥ 15% → 继续完整五折
3. 若 fold=0 为负 → 审计语义专家提示模板和模型输出质量
4. 五折通过 → 训练路由器 → 生成提交
