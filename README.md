# 量化投资与量子核

从股票价格、收益和风险出发，学习因子选股、择时、组合、回测与机器学习。经典量化必修完整收录 Quant-for-Beginners 与 WhaleQuant 两个固定版本的教学内容。量子核与有限测量作为进阶选修，通过真实研究案例讨论一个模型何时值得采用。

**[打开课程目录](课程目录.md)**　·　[开始第一章](classic/chapters/01.md)　·　[进入量子研究案例](notebooks/qukernet_cloud_v3/README.md)

## 学习与运行

八章必修共用一条学习路径。每章的合讲连接两套教材；原实验保留自己的数据源、时期、标签和作业。相同概念下的不同实现可以直接比较。网页交互可在浏览器打开，Notebook 的运行条件和平台限制见对应章的无影运行记录。

本地阅读可从讲义或交互 HTML 开始。运行 Notebook 时为两门上游分别创建 Python 环境，选择对应的 Jupyter 内核，然后按章从头执行。[入门课原依赖](classic/sources/quant-for-beginners/requirements.txt)与[WhaleQuant 原依赖](classic/sources/whale-quant/requirements.txt)保留在各自目录；WhaleQuant 的统计、行情、回测和机器学习示例还用到章内列出的包或平台服务。

运行记录分别说明原课程数据和实验室适配的实际结果。动态下载记录本次获取状态与版本，构造数据与真实行情使用各自的评价口径。实验室私有行情、逐样本输出和原始日志存放于本地忽略目录。

2026 年 11 月 19 日的无影运行中，Quant-for-Beginners 的 31 项原数据实验有 18 项完成、1 项部分完成、12 项遇到数据源阻塞，7 个交互网页均已验证。WhaleQuant 的 32 个含代码工作流都有逐单元记录。[均线与仓位](classic/chapters/05.md)、[上证 50 调仓](classic/chapters/06.md)和[机器学习策略](classic/chapters/08.md)可分别查看原数据结果及实验室适配。

## 量子进阶

50 股、8 比特 QuKerNet 先导使用天衍全振幅模拟器，在 10 个评价日期上得到 50 条记录。微调前后准确率均为 52%，随机搜索和 RBF 为 54%，多数类与同样本线性基线为 56%。可从[研究型课程说明](研究课程说明.md)、[固定版本报告](reports/qukernet_cloud_v3/cutoff_20260325_qlib_missing_v1/README.md)和[离线教学 Notebook](notebooks/qukernet_cloud_v3/README.md)核查方法、预算与结果。

## 来源与共建

[上游路线图](classic/上游路线图.md)列出上游规划与已收录教材的关系。[第三方来源与许可](THIRD_PARTY_NOTICES.md)说明原作者、固定版本和改编范围。[贡献说明](docs/课程建设与贡献说明.md)提供章节、作业与复现实验的提交方式。
