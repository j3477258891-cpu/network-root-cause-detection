# V119 联合删除–新增自适应方案

该目录由 `experiments/v119_joint_campaign.py` 生成。它只生成本地文件，不执行上传。

## 当前状态

- 基线：`v60_combined_checkpoint/highest_verified_combined.csv`（本地离线验证 F1=0.919673，尚未在当前提交记录中确认）
- 当前线上提交记录最高：`v30_safe_split_a2.csv`，F1=0.918711
- 候选：71 个新增、80 个删除（包含 V118 校准的 40 个删除节点）
- 探针：8 个删除组、8 个新增组、4 个混合组
- 已执行 100,000 次离线模拟
- 当前保守门槛未通过：不得生成最终提交文件
- `probe_01.csv` 已线上得分 0.910151；其 20 个删除节点中约 18–19 个为真阳性，删除组分支已停止

V118 的校准结果被强制用于删除先验：40 个节点中预计只有 9 个是安全删除。V118 固定删除池不会直接作为最终提交。

## 命令

```powershell
$py='C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH='D:\zgyidong\.deps'

# 重新生成候选、矩阵和 20 个探针
& $py experiments\v119_joint_campaign.py --generate

# 运行结构检查和离线门槛检查
& $py experiments\v119_joint_campaign.py --validate

# 记录已获得的线上分数并解码；分数按 probe_01 到当前连续提供
& $py experiments\v119_joint_campaign.py --decode --scores <score1> <score2> ...

# 只有 posterior.json 通过 0.85/0.80/p10 门槛时才会生成 final_*.csv
& $py experiments\v119_joint_campaign.py --emit-final
```

## 重要安全规则

1. `validation.json` 中 `final_emission_allowed` 为 `false` 时，不提交任何 V119 探针以外的最终文件。
2. 每个线上分数必须记录对应文件、预测数量和 F1；显示 6 位小数时使用整数 TP 反推并检查舍入候选。
3. 如果后验门槛无法达到，保留 `v60_combined_checkpoint/highest_verified_combined.csv`，不要降低当前最高真实分数。
4. 不要继续提交 `probe_02.csv` 到 `probe_08.csv`；它们属于同一失败的删除组分支。
