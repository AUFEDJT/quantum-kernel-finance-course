# Quant-for-Beginners 原实验复现

固定来源为 Datawhale `quant-for-beginners` 提交 `8fbf8942afc0e80d01538ee545e72174e099e81a`。原教材保存在 `classic/sources/quant-for-beginners`，保留其 MIT 许可和作者信息。本目录提供独立复跑工具、兼容修复与不含原始行情的结果索引。

2026-11-19 在隔离的 Linux / Python 3.11.15 环境执行了全部 8 本 Notebook 的 61 个非空代码单元，并核验 7 个 HTML 演示。按独立教学目标归并的 31 项实验，兼容修复后为 **18 项完成、1 项部分完成、12 项受原数据源阻塞**；另外 18 项数值挑战/独立核验为 10 项完成、8 项受阻。7 个 HTML 的 78 个控件操作中观察到 65 个画布或文本变化；页面无 JavaScript 错误，40 步后的截图逐页可读。

每章、每个来源路径、原单元编号、独立实验 ID、挑战与实验室终态均在 [results.json](results.json)。状态以有效数据和教学目标判断：原始执行中 42 个单元没有 Python 异常，但空表及 NaN 结果不计为成功。

| 章 | 原目标有效终态（兼容修复后） | 主要情况 |
|---|---|---|
| 1 | 4 完成、1 受阻 | 模拟可运行；Yahoo AAPL 无数据 |
| 2 | 1 完成、3 受阻 | 手算收益率完成；行情分析受阻 |
| 3 | 2 完成、1 受阻 | 模拟与原文 A 股备选分支完成；Yahoo 分支受阻 |
| 4 | 3 受阻 | AAPL/SPY 回测无有效原行情 |
| 5 | 2 完成、1 部分、4 受阻 | 构造路径完成；含行情的统计部分受阻 |
| 6 | 3 完成 | 新浪原标的 qfq：Sharpe、Beta、统计 |
| 7 | 2 完成 | 新浪原标的 qfq：回撤、仓位 |
| 8 | 4 完成 | 新浪原标的 qfq：对齐、相关性、组合、基准 |

修复保持原目标和原数据源：按 AkShare 官方更名把 `stock_us_daily` 改为 `stock_us_hist_sina`；把 pandas 已删除的 `last('6M')` 等写法转换为原来的月末偏移语义；选用主机已有中文字体。变更只生成运行副本，逐单元差异及源文件哈希记录在 `compatibility_manifest.json`。原始失败记录保留。Yahoo 实测限流/空表且公开图表接口返回 403，未用别的源冒充 Yahoo 成功。

## 复跑

先进入包含 `classic`、`experiments` 和 `private` 的项目根目录，再创建独立环境。原实跑使用 Python 3.11；完整依赖版本记录在 `environment.lock.txt`。Linux 需有 Python 的 `venv` 组件。下面使用显式解释器路径，无需激活环境。

Linux / Bash：

```bash
python3.11 -m venv private/qfb_env
private/qfb_env/bin/python -m pip install -r experiments/classic/quant-for-beginners/environment.lock.txt
private/qfb_env/bin/python -m playwright install chromium
private/qfb_env/bin/python experiments/classic/quant-for-beginners/reproduce.py --source-root classic/sources/quant-for-beginners --output private/qfb_new_run
```

Windows / PowerShell（本机验证为 Python 3.12.14；以下均为单行命令）：

```powershell
py -3.12 -m venv private/qfb_env
& ./private/qfb_env/Scripts/python.exe -m pip install -r experiments/classic/quant-for-beginners/environment.windows.lock.txt
& ./private/qfb_env/Scripts/python.exe -m playwright install chromium
& ./private/qfb_env/Scripts/python.exe experiments/classic/quant-for-beginners/reproduce.py --source-root classic/sources/quant-for-beginners --output private/qfb_new_run
```

2026-11-19 已从公开文件在新建 Linux Python 3.11.15 和 Windows Python 3.12.14 环境完成六项入口检查：依赖一致性、总计划、真实 Notebook、失败停止、500 日归档重放和 7 个 HTML；两边的 64 份教材文件均未改变。详见 [portability_check.json](portability_check.json)。Windows 锁额外固定 `colorama` 与 `pywinpty`。

已有可用 Chrome / Edge 时，可省略浏览器下载，在 `reproduce.py` 后加 `--browser-executable "浏览器可执行文件完整路径"`。Linux 若提示 Chromium 缺少系统共享库，按 Playwright 的诊断安装操作系统依赖后再运行 HTML。

输出目录必须为新的私有路径。运行器会保留 API 调用参数、实际默认值、UTC 获取时间、响应表快照、SHA-256、环境版本、执行后的 Notebook、单元异常、挑战和浏览器记录。驱动的退出码不等于金融目标完成；按各 `summary.json`、`acquisitions.jsonl` 和 `challenges.json` 检查有效输入与结果。`--dry-run` 可查看调用计划，`--skip-html` 只用于已有单独 HTML 验证的场景。

`run_notebook.py` 可单本执行，`make_compat.py` 只生成运行副本；`run_html.py --browser-executable /path/to/chrome` 可显式选择浏览器。第 3 章是原文给出的两条独立输入分支，兼容 A 股复跑只跳过 Yahoo 输入单元 11，保留原 A 股单元 13 及后续目标。

## 实验室合同与终态

在两套经典上游原实验均取得终态后才开始实验室阶段。检查了原标的身份、动态日期范围、复权、ETF 基准和共同交易日。

美股原标的文件存在，但更新止于 2026-04-21；指定 SPY 和 XLE 在已查美股、ETF、指数目录无匹配。相关原动态日期合同不满足，21 项行情实验终止适配；包含构造路径和行情两部分的一项记部分完成。8 项纯模拟/构造输入无需替换。

第 3 章原文明确采用 600519 最近 500 个交易日，可以适配。基础日线缺 16 个交易日；完整的 `stock-trading-data-pro` 覆盖原 528 行区间。`lab_adapter.py` 从前收盘价重建腾讯口径的**加法前复权**，528 个日期及保留 2 位小数的收盘全部与阶段一冻结的腾讯原数据相同；腾讯数据仅用来校验，不参与填值。无影实跑后原 500 行分支零错误，MA5、MA20 与信号独立核算通过（15 次金叉、14 次死叉）。

该适配器只覆盖本次核验的 600519 区间，不是通用复权函数。普通乘法链式前复权在相同 pro 数据上与原腾讯归一价格仍有约 1.41% 最大差异，不能混用。私有输入由 `freeze_lab.py` 及补充源记录冻结；适配通过 `QFB_LAB_ROOT`、`QFB_LAB_REFERENCE` 明确提供私有数据与原公开源校验快照，配合 `run_notebook.py --prelude lab_prelude.py --skip-cells 11 --append lab_checks.py` 运行第 3 章兼容副本。输入日期不符或验证不等时适配器会拒绝继续；适配初始化失败会终止 Notebook，不会回落联网取数。

已有本次私有证据时，使用完整入口重放 **2026-11-19 的归档实验**（下例为 Linux；Windows 将解释器替换为 `./private/qfb_env/Scripts/python.exe`）：

```bash
private/qfb_env/bin/python experiments/classic/quant-for-beginners/replay_lab.py --source-root classic/sources/quant-for-beginners --lab-root private/qfb_20261119/lab_input --reference private/qfb_20261119/runs/chapter03_compatible/data_snapshots/0001_akshare_stock_zh_a_hist_tx.csv --output private/qfb_lab_replay --as-of 2026-11-19
```

归档入口在私有运行副本中显式绑定原单元的两处日期表达式，记录差异与哈希；原教材保持不变。这是对已验区间的重放，不代表重新获取当前行情。上述私有数据不在公开仓库，缺失时入口会明确报错；有授权数据的维护者可用 `freeze_lab.py --root "实验室数据根目录" --output "新的私有目录"` 冻结必要文件，并在逐日校验后使用。

## 教学时需要说明

- 第 8 章逐日等权收益对应每日再平衡，不能称固定股数买入持有。所有相关系数为 1 时，组合波动为各票波动的加权和，不必等于某一只股票。
- 第 4 章调仓日收盘价的交易回合收益与滞后信号乘当日收盘收益的净值口径不同，讲解胜率时应把成交时点说清楚。
- 第 1 章乘法随机游走的终值不能直接断言正态，最低价格也不等于最大回撤。
- HTML 通过的是实际运行、交互响应和视觉核验；教学公式及文字的全面审校仍是另一项工作。

原始行情、实验室数据、执行输出、日志和截图保留于 git 忽略的私有证据目录。公开结果只包含汇总状态、必要数值核验和证据包哈希。

来源：[上游仓库](https://github.com/datawhalechina/quant-for-beginners)、[AkShare 官方更名记录](https://akshare.akfamily.xyz/changelog.html#changelog-rename)。
