# V155 分类型纠错实验

## 固定目标和边界

已评分冠军是 V153 p03：P=1049、TP=971、F1=0.927855。目标是实际线上 F1≥0.937855。
本轮剩余 6 次，最多 4 次探针、1 次精确合并、1 次异常或风险保留；每天最多 2 次。
本工具没有任何比赛上传功能，也不会把生成文件计为已经提交。

本次训练从 2026-09-09 20:28:59（北京时间）起计 24 小时，原期限保存在 training_window.json。
重复执行不延长期限。新动作排序是模型先验，不是线上标签。

## 本次正式训练入口

使用已验证的本地 Python 环境。正式四组比较全部运行在同一机器和依赖版本中。
`worker.py` 是本次唯一正式比较实现；`prepare.py` 已冻结五折交叉拟合数据。
旧的 `training.py` 不用于本轮正式结论。`smoke_local/` 只是少量树的接口测试，不能用于筛选候选。

在 D:\zgyidong 中执行：

```powershell
& experiments/v152_error_repair_campaign/.venv/Scripts/python.exe experiments/v155_typed_repair_campaign/campaign.py audit
& experiments/v152_error_repair_campaign/.venv/Scripts/python.exe experiments/v155_typed_repair_campaign/campaign.py train
& experiments/v152_error_repair_campaign/.venv/Scripts/python.exe experiments/v155_typed_repair_campaign/campaign.py build
& experiments/v152_error_repair_campaign/.venv/Scripts/python.exe experiments/v155_typed_repair_campaign/campaign.py recommend
```

训练正在运行时不要再启动第二份同目录任务。完成的检查点只有数据、代码、软件版本签名完全匹配时才能复用。
云端依赖隔离检查和 smoke test 曾由用户确认通过；本次页面交互超时，未成功启动 V155 云端正式训练。
不会把云端安装失败的旧状态改成虚假的成功记录，也不混合两端不同版本的验证结果。

## 发文件前的两道门槛

1. 完成五折，以 32 动作/546订单预算比较。分类型模型总 U 必须正且严格超过两个对照；至少三折正；最差折不差于主对照。
2. 历史方程/风险池先筛选，再按校准期望 U 排序。候选池的整数乐观上限必须≥目标。

U=2×ΔTP−0.937855×ΔP；从冠军到目标需要 U≥20.930515。
门槛不通过就写出 final_gate.json 并停止生成 CSV。不会通过换模型名、放宽门槛或复制旧 CSV 来伪装执行成功。

候选最多48个独立订单，各动作类型优先16个，不足再补。每组最多8个，均独立从p03冠军生成。
完整保护 p03 新增节点和方程已确认真节点，排除固定假新增、既定风险池与 dd294。

## CSV 与后续反馈

`emit-spec` 只输出可审查的 JSON 生成规格，不写 CSV。通过门槛后由随项目保留的 ArtifactTool 生成器导出文件，
再用 `register --spec <规格路径>` 复核并登记。检查包括546订单、原顺序、1–8根因、精确差集、元数据、P和SHA-256。
单纯模型概率不允许生成“最终已提分版”。

```powershell
& experiments/v152_error_repair_campaign/.venv/Scripts/python.exe experiments/v155_typed_repair_campaign/campaign.py emit-spec
& experiments/v152_error_repair_campaign/.venv/Scripts/python.exe experiments/v155_typed_repair_campaign/campaign.py record --probe-id p01 --score 六位小数 --submitted-at 带时区的实际时间 --evidence 真实反馈说明
```

`record` 只更新证据和建议，不自动生成下一份文件。失败和异常尝试占用预算；异常时暂停，不继续探索。
已测组与拆分子组的 TP 计数守恒，枚举不相交组的并集计算精确合并分数。
两步查询优先条件达标概率，再比较保留冠军后的最佳F1期望；这些都是条件工作模型计算，非真实成功概率。
第6次默认保留，任何风险版需另外审查，不自动用完额度。

## 验证

```powershell
& experiments/v152_error_repair_campaign/.venv/Scripts/python.exe -m unittest discover -s experiments/v155_typed_repair_campaign -p "test_*.py"
```

测试覆盖：收益三值/二值映射、权重、目标门槛、分组独立、拆分守恒、整数上限穷举对照、两步查询决策树穷举对照。
历史审计结果见 history_audit.json。正式进度和结果见 training_local/progress.json，局部折结果不能充当线上增益。
