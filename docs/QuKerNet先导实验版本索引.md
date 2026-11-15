# QuKerNet先导实验版本索引

基础课程协议与进阶研究协议分别管理。每次运行必须绑定一个机器可读协议、数据版本、代码哈希、预算窗口和结果目录；不能同时使用不同版本的默认参数。

| 版本 | 用途与范围 | 协议与证据 |
|---|---|---|
| 基础课程A/B | 20股、精确核年度项目与固定池有限测量教学设计 | [基础数据协议](数据与实验协议.md) |
| cloud_v2 | 历史三季度研究与预检，保留当时状态 | [v2入口](../experiments/qukernet_finance/cloud_v2/README.md) |
| cloud_v3原截止版 | 原始50股三小时先导，历史记录 | [原协议](../experiments/qukernet_finance/cloud_v3/protocol.json) |
| 截至2026-03-25连续历史版 | 47条特征有效、3条正式停牌历史不符合连续121日门槛；未启动云实验 | [旧截止协议](../experiments/qukernet_finance/cloud_v3/protocol_cutoff_20260325.json)、[记录](../reports/qukernet_cloud_v3/cutoff_20260325/status.json) |
| 截至2026-03-25 Qlib missing v1 | 新缺失处理版本；50股、8比特、16/16/50样本、两臂；原默认3小时，本次经授权续作固定配额 | [新协议](../experiments/qukernet_finance/cloud_v3/protocol_cutoff_20260325_qlib_missing_v1.json)、[授权修订](QuKerNet固定配额续跑说明_20261115.md)、[版本报告目录](../reports/qukernet_cloud_v3/cutoff_20260325_qlib_missing_v1/) |

新版的实际启动、消耗、完成或停止，以其带时间戳的`status.json`及私有账本为准。教学文件更新不启动实验，也不把准备完成写成效果验证完成。

## 本轮教学路线

1. C02/C03：[停牌与缺失教案](停牌与缺失处理_qlib_missing_v1.md)和[解析练习](../notebooks/qukernet_cloud_v3/03_停牌与缺失处理.ipynb)。
2. C08进阶/C11：[完整链条解析演示](../notebooks/qukernet_cloud_v3/04_完整链条解析演示.ipynb)，区分神经评分器与SVC。
3. C09/C10/C11：[真实结果与预算](../notebooks/qukernet_cloud_v3/05_真实结果与预算.ipynb)，只读汇总重建图表。

本轮仅完善上述进阶案例，84课时安排不变，不宣称11章讲义全部完成。完整链条解析演示不是股票结果，真实汇总不替代受限数据的端到端重跑。没有数据再分发许可时，公开解析代码、数据接口与获准汇总，真实输入由有权访问者自行提供。
