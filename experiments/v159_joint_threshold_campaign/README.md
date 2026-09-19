# V159：超过 0.938154 的联合池证据审计

**当前决定：不提交，保留六次。** 已执行接受方案中的“先检查验证数据是否足够，不足就停止”分支。详见 [RESULTS.md](D:/zgyidong/experiments/v159_joint_threshold_campaign/RESULTS.md)。

当前最高实报是 V158 安全最终版，P=1045、TP=972、0.930589。75 个 V150 剩余新增中恰好 17 真；辅助池为 3 删除、2 替换。联合乐观上限 0.942463 只表示存在某个理想组合。

指定训练数据只有 1,634 个有标签工单；忽略站点隔离最多容纳两个不重叠的 546 工单任务。80% 的双侧精确 95% 下界至少需要 17 个全部成功的独立任务。现有五折并不是五次完整独立竞赛；缓存、同池历史探针、bootstrap 均不能替代所缺的证据。

## 运行

在 `D:\zgyidong` 的 PowerShell 中：

```powershell
$v159Python = 'D:\zgyidong\experiments\v152_error_repair_campaign\.venv\Scripts\python.exe'
$v159Campaign = 'D:\zgyidong\experiments\v159_joint_threshold_campaign\campaign.py'
& $v159Python -B $v159Campaign audit
& $v159Python -B $v159Campaign evaluate
& $v159Python -B $v159Campaign recommend
& $v159Python -B $v159Campaign prepare
```

- `init`：首次冻结输入；重复运行只审计，不重置额度。不应重新建立目录规避历史账本。
- `audit`：只读地验证来源哈希、当前冠军、继承余额、截止时间和停止状态。
- `evaluate`：封存数学审计与样本量结论；重复运行返回同一结果。
- `recommend`：保存当前停止决定。
- `prepare`：目前返回 `emitted=false`，无正式 CSV；没有强制绕过选项。
- `record`：只接受登记文件的真实反馈，参数为 `--probe-id`、`--attempt-id`、`--score`、`--submitted-at`、`--evidence`；平台失败用 `--failed` 且不填分数。当前没有登记文件，因此无法把无对应文件的数字当成 V159 提交。未知平台时间必须明确标注为录入时间代理。真正重复提交用新 attempt ID，并消耗次数。

`--out` 是全局参数，放在子命令之前。默认输出仅写 V159。测试使用临时目录；假设反馈不进入线上账本。所有命令均不上传。

## 本次交付边界

已实现来源冻结、整数上限、全部辅助组反馈核对、独立验证容量审计、统计门槛、停止报告、余额继承和反馈记录保护。

路线 A 与 B 的完整自适应策略搜索、完整留出重放和 2,000 次 bootstrap **因前置证据门槛失败而未启动**。因此没有真实达标率、策略树或可提交新文件；也没有把旧模型的条件概率称作真实胜率。通过分支的搜索/生成器不属于此次提前停止后的可用功能，不能通过手改 `passed` 开启。

如未来增加有来源的独立验证数据，需要另行审计可比性和新的研究协议。本次截止仍是北京时间 2026-09-14 15:52，不能自动延长；V150–V158 历史输入全部只读。

## 验证

```powershell
& $v159Python -B 'D:\zgyidong\experiments\v159_joint_threshold_campaign\test_campaign.py'
```

成功标准仅为真实线上六位得分 ≥0.938155；离线报告完成不代表达标。
