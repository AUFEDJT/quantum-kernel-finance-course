# WhaleQuant 原实验与复现入口

本目录针对 [WhaleQuant](https://github.com/datawhalechina/whale-quant/tree/828009d910db5eb2c06fa14f4439421b235bef44) 的固定教学版本，保留原目标、数据源和算法语义，并补充可检查的运行结果。上游教学内容按其 README 声明的 **CC BY-NC-SA 4.0** 署名和传递；详见 [NOTICE.md](NOTICE.md)。

2026-11-19 在隔离的 Linux / Python 3.10.12 环境完成原版运行：32 个含代码工作流、457 个单元，覆盖 Markdown 独有示例、Notebook 代码及 markdown 内代码、`code/ch03` 全部脚本。多个导出版本的重复代码保留在覆盖计数中，独立代码哈希为 275 个。无可执行代码的导航和概念页继续作为教材保留。

| 原版单元结果 | 数目 | 含义 |
| --- | ---: | --- |
| executed | 235 | 单元执行结束；其中 11 个仅导入/定义、3 个空模块，221 个执行了其他语句 |
| failed | 173 | 实际异常，含上游缺输入之后的级联异常；不代表 173 个独立方法失败 |
| expected_demo_error | 20 | 依据源码注释识别的错误演示，保留具体异常供检查 |
| blocked_credential | 22 | 需要授权凭据；未使用上游原有凭据或用户本地配置 |
| blocked_platform | 6 | 平台运行时/订单接口需专用环境；原代码保留 |
| blocked_dynamic | 1 | 文内动态安装命令未执行；依赖改为隔离环境显式安装 |

完整可定位结果见 [results.json](results.json)：`entries` 是逐单元原版终态，`workflows` 汇总各源文件，`validations` 记录产物验证与修复，`script_statement_results` 保留脚本逐语句续跑结果。原始价格、取得时点、快照哈希、完整日志、图像和环境证据只保存在项目忽略的 `private/classic/whale-quant`。

## 已验证的产物与剩余条件

| 章节/条目 | 本次检查到的产物 | 边界与剩余条件 |
| --- | --- | --- |
| 第 2 章统计 | 原 Notebook 32 单元执行，11 张图；修复 Linux 中文字体映射后无缺字警告，抽看 2 图 | 计算未改动；不等同逐公式正确性证明，原随机示例并未全部固定种子 |
| 第 3 章 BaoStock | 原接口 8 个示例均返回非空表；日线 125 行、分钟线 1008 行、指数 117 行、成分 50 行、4 类财务表各 1 行 | 仅将 Windows 输出位置改为隔离目录；当前成分结果随接口时点变化 |
| 第 3 章 pandas 脚本 | 13 个非空脚本逐语句运行：289 次完成、11 次异常；Qantas 周均值约 6.982，文件输出已检查 | 补 `toolkit_config.DATADIR`、解决 `code` 与标准库同名冲突；`tsla_prc.csv` 上游缺失，对应日期练习仍待原文件 |
| 第 5 章择时 | MA/MACD 指标、滞后 1 期信号收益流水生成，1215 行 | 明确使用 BaoStock 替代数据；原 Tushare/JoinQuant 凭据链路仍待授权环境 |
| 第 6 章原数据调仓 | 按 Markdown 的 `000016` 定义，官方成分接口返回 50 股；以官方日期前一年至运行日，逐股请求原 `hfq` 数据 | 50/50 行情请求返回 `RemoteDisconnected / ConnectionError`，未缩减股票池，原滚动优化未开始；已从原 Notebook 恢复函数并去掉代码围栏混入正文 |
| 第 6 章补充函数练习 | 原 `sharp/rp/var` 三个优化目标实际求解；4 股、241 个收益样本，权重有限、非负、总和约 1 | 明示的替代股票池仅用于函数验证，不计作原 50 股滚动回测完成 |
| 第 7 章 Backtrader | 原双均线策略、仓位及费率设置运行 1215 根 K 线，完成 66 笔交易，资金结果有限 | 同标的和时间段使用 BaoStock 原始 OHLCV，不能与原 Tushare 数据结果混称 |
| 第 7 章 StupidHead | 从 Markdown 恢复原框架，用原 T888 日线样本运行 846 行；独立持仓流水误差 0；完成原 100 次参数优化 | 修复 pandas 赋值与 TA-Lib 整数周期；原 Sharpe/无风险利率口径和样本内优化仍需教学讨论 |
| 第 8.1 节小波 + SVR | 原 db8、30 阶滞后、原切分点和 LinearSVR 流程生成 1674 个训练样本、361 个测试预测，均为有限值 | BaoStock 替代价格；原流程先全样本缩放/小波再切分，存在时间泄漏，运行完成不能证明预测有效 |
| 第 8.3 节时间序列 | 修复 Matplotlib 旧样式名后，真实请求原 Yahoo 指数接口 | 本次返回 `YFRateLimitError` 和空数据，分解实验未完成 |
| 平台回调 | 原 JoinQuant/BigQuant 回调在内存模拟中通过 7 项断言：买卖分支、训练集分位裁剪、调仓周期、序列形状及排序等 | 没有提交真实订单；本环境缺 jqdata/jqdatasdk/biglearning/bigdatasource/bigtrader，完整平台训练与回测仍待支持环境 |
| 第 4 章及第 8.2 节 | 保留逐单元的实际异常与上游教学目标 | 第 4 章缺股票截面数据且标准化后 ndarray 被当 DataFrame；8.2 全市场 Tushare 数据未随上游交付，LightGBM 完整训练不计成功 |

11 项恢复/辅助检查中，9 项达到各自明示目标，2 项原数据链路分别因 Yahoo 限流、AkShare 后复权行情远端断开而阻塞；额外的统计条目只核验执行和图像。第 3 章旧 Tushare `get_hist_data` 与 `get_stock_basics` 也已实际请求，返回 DNS/网络错误，其结果保留在逐单元记录。这里的目标包括数据准备和模拟回调，不能理解成“9 套完整策略都已验证”。

## 在 Linux 复现

以下命令从课程项目根目录执行。执行器使用 Linux 的 `SIGALRM`；Windows 可使用独立 Linux/WSL 环境。完整依赖版本见 [requirements-tested.txt](requirements-tested.txt)，其中 TA-Lib 若无匹配 wheel 需要构建条件。本次依赖安装和 20 个主要包导入已在隔离环境验证。

```bash
python3.10 -m venv private/whale-replay-venv
private/whale-replay-venv/bin/python -m pip install -r experiments/classic/whale-quant/requirements-tested.txt

# 原文逐单元续跑；允许实际公开行情请求，源文凭据和平台订单调用受拦截。
private/whale-replay-venv/bin/python experiments/classic/whale-quant/run_workflows.py \
  --source classic/sources/whale-quant \
  --output private/classic/whale-quant/replay-original --network

# 明示的路径/兼容修复、替代源检查与内存模拟。完整运行会先准备后续所需数据。
private/whale-replay-venv/bin/python experiments/classic/whale-quant/recover_experiments.py \
  --source classic/sources/whale-quant \
  --output private/classic/whale-quant/replay-recovery
```

原版执行器可用 `--only notebook/ch02` 等来源子串重跑单一工作流。Linux 已有中文字体时可设置 `WHALE_CJK_FONT=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`，仅为原 `SimHei` 请求提供运行时别名。恢复程序的 `--only` 用于已有依赖数据的单项排查，每次应采用独立输出目录；完整复现建议运行全套命令。

这些工具只面向已审查的固定教学源。路径限制和调用拦截是本次运行的辅助保护，不是通用不可信 Python 沙箱。`StupidHead.py` 是从原 Markdown 恢复的教学框架入口；市场输入文件不随公开结果发布。
