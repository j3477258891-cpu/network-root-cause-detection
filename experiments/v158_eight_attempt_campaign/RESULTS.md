<!-- confirmed-safe-final-0930589 -->
# Latest final score confirmation

User reported online F1 **0.930589** for the previously recommended `v158_safe_final_utf8.csv` (file attribution inferred from conversation). Integer TP=972, P=1045; matches the unique decoded final. New actual champion updated in the V158 ledger.

Two of the eight authorized attempts are now recorded; **six remain**. Decision: **stop_research_gate_failed**. The new pool failed its statistical and 0.94 optimistic-bound gates. Preserve the remaining six attempts; target 0.94 is not achieved. Platform submission time is unavailable and recorded time is explicitly a proxy. All older dated sections below are historical snapshots.

# V158 实施与研究结果（2026-09-12）

## 最新反馈：2026-09-13

用户反馈 safe_probe（V157 adaptive p03）为 **0.929119**，P=1044、TP=970。
文件归属根据上一条推荐推定；未提供平台截图、上传字节或准确提交时间，账本明确标记为记录时间代理。
剩余状态从4种减到1种，确定安全删除1/2/3；原已确认新增8/11/15继续保留。

**下一次提交 `v158_safe_final_utf8.csv`**：P=1045、TP=972、方程预期F1=0.930589，
相对实际冠军0.929254预期增加约0.001335。当前剩7次；提交最终版消耗1次后剩6次。
后六次研究仍因门槛失败而停止，不为消耗次数继续探索。

最终版SHA-256：`ca38eca42a56ee36bab244639bb8c7d1f61dfebc3fa56d1ef8dfc51587b6fc4e`。
546订单、精确差集/元数据、根因数1–8、无重复和UTF-8无BOM均通过检查；
完整历史整数方程独立确认最终增量TP相对V149为3..3。详见`safe_final_verification.json`。
工具没有上传。0.930589仍待实际评分，当前实际冠军保持0.929254。

以下为9月12日实施快照，其中“尚未生成最终版/剩8次”等状态已被上述反馈更新。

**工具已实现；本轮研究完成且未通过门槛。执行前两次安全解码，停止后六次新池探索。**

当前最高实报仍为 V156 final：P=1048、TP=972、F1=0.929254。
本轮没有比赛上传，没有录入新的实际成绩，8次额度尚未消耗。0.94未达到。

## 研究结果

复用原始完整五折预测，没有重训大模型、没有在外折挑融合权重或预算。
固定等权融合两个动作模型，采用目标0.94、每546订单至多48个动作，与同预算两个原始对照配对比较。

| 方法 | 合并OOF F1增益 |
|---|---:|
| CatBoost原始对照 | +0.003152 |
| V38原始对照 | +0.002081 |
| 固定融合重排 | +0.003956 |

新排序在4/5折正收益。820个站点簇、2000次配对bootstrap：

- 绝对增益95%下界：+0.000371。
- 相对CatBoost增益95%下界：**-0.000981**。
- 相对V38增益95%下界：**-0.001752**。

相对对照的可靠性门槛未通过。上述是约0.857代理基线上的OOF增益，不能加到实际线上冠军分数上。

三个完整首次新池的历史计数重放完成；只使用该次得分之前的方程，不把旧拆分当作独立试验。
新排序的平均计数绝对误差为0.914890；CatBoost为0.802767，V38为1.315695。
V150的80个动作未被该冻结模型完整覆盖，因此没有用部分预测冒充整池计数评估；其20/80真实方程仍完整保留。

## 新池上限

2515个原始动作经合法性、历史排除、上下文一致性与整数收益筛选后，剩7个模型条件期望效用为正的动作，
按每工单最多一项选出5个不同工单。这里没有放宽失败门槛，没有把原top-80或已解码收益重复计入。

在安全合并P=1045、TP=972成立的条件下，该5动作池的整数乐观上限为：

**P=1042、TP=974、F1=0.9338446788111218，低于0.94。**

这个上限仅适用于本轮筛出的池，不代表整个候选空间的全球上限；它也不是预期成绩或已经实现的分数。
正式新候选池和新研究提交CSV均未生成。依据用户选择，停止后六次探索，保留次数。

## 当前唯一待提交文件

`D:\zgyidong\experiments\v157_joint_correction_campaign\v157_probe_adaptive_p03_utf8.csv`

- V158登记编号：`safe_probe`。
- P=1044、546个订单；精确节点差集、元数据、根因数、无重复、UTF-8无BOM检查通过。
- SHA-256：`c4096c31d9868b4f7304196070838e7e1922b4c1459bd81dab5ff780940985c8`。
- 成本：1次；作用：区分剩余4种状态。探针本身可能降分。
- 四个合法反馈：0.927203 / 0.928161 / 0.929119 / 0.930077。
- 收到真实反馈后，`prepare`仅生成该分支的安全最终版；再用1次验证预期0.930589。
- 今天账本代理时间已记2次；本工具没有实时平台额度，下一可用日上传前核对真实当日额度。

尚未生成安全最终版，因为首次真实反馈仍缺失；不会用测试分数代替。模拟测试仅发生在临时目录。
新反馈使用本目录的`campaign.py record`，不要同时修改已冻结的旧V157账本。

## 验证与恢复

24项不同回归测试通过，覆盖四个反馈分支、实际评分前阶段锁、重复尝试与幂等录入、CSV差集和编码、
异常计费暂停、研究门槛、防止未来信息、站点配对、分组守恒、最终次数保留、整数可行性与超时拒绝。
对最终5动作池另行枚举全部32个子集，以整数上界独立核对整体优化结果。
180个冻结历史来源保持哈希不变，73份历史评分CSV已逐份校验。

第一次研究结果保存遇到NumPy整数序列化错误，已修复并测试；失败记录保留在
`research/failed_run_01_serialization.json`，代码更新记录在`implementation_updates.json`。
这次修复没有更换算法、放宽门槛、重置预算或延长截止时间。

研究于北京时间2026-09-12 16:02:52完成，早于原截止时间2026-09-14 15:52:13。
48小时是允许的上限；已完成且失败的冻结方案不继续空跑或更名重试。

详细数值见`research/results.json`；核验清单见`verification_report.json`；操作命令见`README.md`。
