# V150：六次额度的全整数自适应新增计划

基线是线上已确认的 V149 最终版：P=1045、TP=969、F1=0.927717。
本轮六次机会在这次确认之后计算，平台剩余额度仍需用户自行核对。
不自动上传，不删除基线节点，不使用6600替换，不改动历史提交文件。

## 先提交什么

构建和验证通过后，第一份文件是 `v150_probe_01_all80.csv`，P=1125。
它是候选质量诊断，可能降分。请只提交这一份，将实际六位分数发回。
80节点中若 T>=51，文件本身超过0.94；T<=24则停止这个池的冲目标探针；
其他结果由程序重新选择下一次整数拆分，不按固定文件编号连续提交。

## 命令

```powershell
$taskPython = 'C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$taskCampaign = 'D:\zgyidong\experiments\v150_adaptive_addition_campaign.py'
& $taskPython -B $taskCampaign audit
& $taskPython -B $taskCampaign recommend
```

收到第一份真实成绩后，填写实际六位数和截图时间，例如：

```powershell
# 必须把两个占位值替换成真实结果；不要使用假分数运行 record。
& $taskPython -B $taskCampaign record --probe-id probe_01_all80 --score <实际六位分数> --submitted-at <带时区的实际提交时间>
& $taskPython -B $taskCampaign recommend
& $taskPython -B $taskCampaign emit-next
```

`recommend` 和 `audit` 只读。`emit-next` 只生成当前所需的一份拆分文件，
达到门槛时转入 `emit-best`。后者只生成确定计数的最佳已知组组合，
不把组的正确数当成每个节点的标签。实际达标只认最终线上评分。

同一 probe-id 的重复录分默认幂等；真正重复提交必须使用不同 `--attempt-id`，
每次实际提交都占预算。平台失败用 `record --failed --reason ...` 记录，
保守计入一次并暂停。分数异常、哈希变化或方程冲突暂停后先复核；
不能通过删除异常记录、重建目录或编造分数继续。更正需要保留原始证据。

## 数学假设与限制

新文件从固定 V149 基线独立生成，保持546个订单、顺序不变、根因数1–8。
所有文件都有 SHA-256 和精确新增差集。历史整数方程只使用实际已评分CSV，
不把OOF、模拟、推断标签写成线上观测。源模型票数不是独立证据。

策略条件概率假设给定组内正确数后，各标签排列等概率，不代表真实候选池
成功率；历史方程额外用于硬一致性检查，不伪装成已校准后验。
T=50场景中，任意整数拆分的条件达标率99.9661%不适用于T未知的当前池。
下一步必须先测总量。一个低分探针可以有信息价值，但不会替代真实冠军。

## 验证

```powershell
& $taskPython -B 'D:\zgyidong\experiments\test_v150_adaptive_campaign.py'
& $taskPython -B 'D:\zgyidong\experiments\test_v150_adaptive_campaign.py' --benchmark
```

测试使用内存中的假设场景，绝不写入本轮 `online_scores.json`。
保留1次最终确认；每天最多2次的提醒包括已记录的V149确认，但不掌握
其他任务/页面上的提交，上传前必须再看平台当天额度。
