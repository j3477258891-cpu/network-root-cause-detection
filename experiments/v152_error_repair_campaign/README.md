# V152 错误修复与自适应提交

真实冠军固定为 V149，P=1045、TP=969、F1=0.927717。旧目录和冠军文件只读。
本工具不登录平台、不上传、不自动消耗提交次数。生成CSV不等于线上提分。

## 运行

在 PowerShell 中使用隔离环境（不修改旧 `.deps` 或捆绑运行时）：

```powershell
$v152Py = 'D:\zgyidong\experiments\v152_error_repair_campaign\.venv\Scripts\python.exe'
$v152Cli = 'D:\zgyidong\experiments\v152_error_repair_campaign\campaign.py'
& $v152Py -B $v152Cli audit --refresh
& $v152Py -B $v152Cli train --max-hours 48
& $v152Py -B $v152Cli build
& $v152Py -B $v152Cli recommend
```

`train --smoke` 只检查流水线，输出到独立目录，永远不能通过正式生成门槛。
正式五折可从已完成的折恢复；输入或代码改变时拒绝复用旧折，不覆盖已冻结的 campaign。
`training_window.json`保留首次训练的48小时截止时间，续跑不重新计时。
此前使用验证折真实F1筛选动作的部分结果已归档为无效协议，不能通过生成门槛。
训练优于对照、至少三折正收益之后，`build` 才筛选最多64个独立订单。
组的最优理论上限也无法达到目标时，不生成冲目标探针。

`recommend` 只读。每次拿到分数先 `record`，读取继续价值评估后，才执行
`recommend --emit` 生成下一份。一个未反馈文件存在时不会再生成其他探针。

```powershell
# SCORE 必须替换为真实六位榜单分数；时间为实际提交时间。
& $v152Py -B $v152Cli record --probe-id p01 --attempt-id platform-attempt-001 --score SCORE --submitted-at '2026-09-07T10:00:00+08:00' --evidence '实际榜单截图或提交记录标识'
& $v152Py -B $v152Cli recommend --emit
& $v152Py -B $v152Cli emit-final
```

失败上传使用 `record ... --failed`，不提供分数，也计入十次额度。
相同 attempt-id 的完全相同录入幂等；重复实际上传必须用新的 attempt-id，并计费一次。
异常记录不覆盖；新实际复测通过后可加 `--resolves-attempt 原异常ID`，两次都计数。
时间必须含时区。工具限制本campaign每天两次；提交前仍须检查平台当天其他campaign的额度。

## 证据和模型

- 历史方程包括真实评分文件和 V150 的20/80计数，假设真实正例总数1044及确定性micro-F1。
- 训练不使用这些测试标签；它们只用于筛选、条件可行性及线上解码。
- V16原始图/时间特征加固定语义哈希，不使用无法证明无泄漏的旧监督得分。
- 五折使用共享站点/相同模板连通分组；内层调参、温度校准和基线交叉拟合互相隔离。
- 内层先平衡独立组数量，防止大连通组导致训练侧仅剩一组而无法继续交叉拟合。
- 动作排序与正收益筛选只使用固定V149参考值，不读取验证折标签或真实F1；标签仅在选定动作后用于评价。
- V30对照为重训ET/HGB节点模型，V38对照为重训ET/HGB动作模型。不是把历史分数当成同协议复现。
- OOF基线只匹配预测密度，不声称复现经过线上修正的V149训练集标签。
- 主验收预算为每546订单64个动作，16/32作为辅助报告。原始指标、最差折和全部失败结果保留。
- TabPFNv2须提供已核对的赛规材料 `--tabpfn-rules-evidence 文件路径` 且安装可用的本地CUDA依赖。
  未获得材料或无法运行时明确标记未执行，不称为已验证失败。禁用遥测，不使用远程推理API。

## 提交策略

前两次选择预期收益最高的不重叠组，第3–8次从新组和既有组的前/后缀拆分中自适应选择。
计数条件概率仅用于排序；采用订单独立工作模型、父计数条件及整数可行性过滤，
不是所有历史方程下精确的贝叶斯后验，也不代表实际达到0.94的概率。
至多八个已知组枚举256种并集。第九次优先兑现精确组收益；第十次仅生成有预期提升的高风险精修或留给复核。
最佳精确组合若已实际提交，则跳过重复合并，可直接做一次高风险精修，省下的额度保留复核。
达到目标提前停。已知组合能超过目标则提前合并。亏损路线停止扩展，但允许继续有信息价值的拆分。

关键输出：`project_audit.json`、`model_comparison.json`、`candidate_catalog.json`、
`equation_system.json`、`online_scores.json`、`submission_manifest.json`、`final_gate.json`。
`final_*.csv`只在推荐门槛通过时生成；失败时保留冠军，不输出伪装成最终版的模型预测。

## 测试

```powershell
& $v152Py -B -m unittest discover -s 'D:\zgyidong\experiments\v152_error_repair_campaign' -p 'test_*.py' -v
& $v152Py -B 'D:\zgyidong\experiments\v152_error_repair_campaign\verify.py'
& $v152Py -B $v152Cli audit
```

CSV由Artifact Tool导入、修改、往返验证后写出，并另存可读变更预览。
原始提交格式 `order_id,output` 不添加说明列、样式或额外元数据。
`verify.py`输出`verification_report.json`，同时确认测试没有修改真实冠军及分数账本。
旧协议部分训练和冒烟结果留在`prior_protocol_*`目录，仅供审计，不参与正式门槛。
