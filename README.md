# “智枢杯”自智网络高校AI挑战赛：无线故障根因定位数据集


---

## 数据集概览

简要描述数据集的核心内容和价值：

- **目的**：用于“智枢杯”自智网络高校AI挑战赛：无线故障根因定位模型训练与推理
- **领域**：通信


---

## 数据集结构

### 文件目录
/train/
故障工单ID/
故障工单ID.log.topo.json
故障工单ID.rootcause.json
test/
故障工单ID/
故障工单ID.log.topo.json
---

## 数据字段说明

1）故障工单ID.log.topo.json文件
它是网络故障相关设备告警知识图谱数据，该文件中包括 3类字段，分别是time（时间），nodes（节点）和edges（边）。time是整型字段，代表故障上报的时间。nodes字段对应一个节点列表，edges字段对应一个边的列表。

节点列表中的每个节点以字典表示，节点分为表示网络设备信息的设备节点和表示网络设备发生告警信息的告警节点，告警节点可能包括字段的含义及说明如下：
|字段|类型|说明|
|-|-|-|
@rid|string|rid是知识图谱中每个节点都具备的唯一标识
@class|string |节点的类型
title/name|string  |告警标题
device|string|上报告警设备的名称，经过脱敏处理
location|string  |上报告警设备的位置地点，经过脱敏处理
time |int|告警发生时间
reason |string|能引发当前告警的原因或原因类别编码
timeLists |int_list|故障事件产生前30分钟内告警是否存在，列表发生30分钟前到25分钟前告警不存在第一个元素为0代表告警
addInfo |string   |告警补充信息，经过脱敏处理
vendor |string   |发生告警的设备厂家，经过脱敏处理
room|string |机房名称，经过脱敏处理
label  |string |告警节点的标签，是否为要分析的目标告警（TargetAlarm）
fault1/fault2|string |告警类型
ID|int|告警编号

设备节点可能包括字段的含义及说明如下：
字段|类型|说明
|-|-|-|
@rid|string  |rid是知识图谱中每个节点都具备的唯一标识
@class|string |节点的类型
ldn|string  |网络的相对位置标识，经过脱敏
zh_label|string|网络设备的中文名称，经过脱敏
related_respoint_zh_labe|string  |相关资源点的中文名称，经过脱敏
vendor_name|string  |网络设备厂家，经过脱敏
device_type|string|网络设备类型
in_dependon |string_list|当前设备节点连接的设备节点rid列表
out_dependon|string_list|当前设备节点连接的设备节点rid列表
positon |string   |设备位置，经过脱敏处理
service_level |string|网络设备层级

边可能包括字段的含义及说明如下：
字段|类型|说明
|-|-|-|
@rid|string  |rid是知识图谱中每条边都具备的唯一标识
@class|string |边的类型
in|string  |边起始的设备或告警节点rid
out|string|边终止的设备或告警节点rid
2）故障工单ID.rootcause.json文件
它是根因结果数据，包括4个字段，字段含义及说明如下：
字段|类型|说明
|-|-|-|
@rid|string  |rid是知识图谱中每条边都具备的唯一标识
title|string |告警标题
location|string  |上报告警设备的位置地点，经过脱敏处理
reason|string|可能引发当前告警的原因或原因类别编码