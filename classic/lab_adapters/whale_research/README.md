# WhaleQuant 研究实战的实验室数据适配

对应经典必修第 4、6、8 章。[逐项结果](results.json)给出可以公开的结论。8.2 的训练窗口为 2018-07 至 2021-12，验证窗口为 2022 年上半年，测试窗口为 2022-07 至 2023-06。将实验室的成交股数与人民币金额分别换算为手和千元，复用上游的股票编码顺序、日期连接、五日标签、训练参数和 0.6 测试阈值。无影使用 8 个线程。

## 用已有授权快照离线复算

原运行输入时点固定为 **2026-11-19**；8.1 读取原标的从 2015 年至前一日 **2026-11-18** 的日线，不随复跑日期自动延长。4.4 使用冻结 20 股面板，包含 2019-01-02 至 2024-12-31 的 29,086 行，字段为 `symbol,date,close,prev_close,total_market_cap`。该截面示例的原教材未指定固定市场窗口。8.2 使用上述固定训练、验证和测试窗口。

公开仓库不分发私有行情。下面需要课程私有保全目录中的文件，准备脚本会校验冻结哈希；缺少文件时先取得获准副本。已验证环境为 Linux / Python 3.10.12，依赖见 [WhaleQuant 复跑依赖](../../../experiments/classic/whale-quant/requirements-tested.txt)。Windows 可用相同 Python 参数与 UTF-8 输入；以下变量赋值采用 Bash 语法。

| 相对课程根的输入 | 用途 |
|---|---|
| `private/classic_lab/whale_research/admission.json` 及其中列明的 CSV | 输入准入与小波快照；本次包含 `wavelet.csv` |
| `private/classic_lab/whale_research/lgb_inputs.zip` | 8.2 的冻结面板及原股票池顺序 |
| `private/classic_lab/whale_research/lgb_input_receipt.json` | 8.2 的归档哈希、字段单位和覆盖回执 |
| `private/classic_lab/whale_research/terminal-lab-evidence.tar.gz` | 内含必需的 `results/wavelet_independent_quotes.csv`，用于核验全部活跃交易日与收盘价 |
| `private/archive/课程可行性验证_20261109/data/daily_panel.csv` | 4.4 实际使用的冻结 20 股面板，哈希与迁移保全清单一致 |

设置实际课程根。`WORK` 必须是课程 `private/` 下的新目录或空目录；准备脚本只提取所需参考文件，不载入归档中的历史结果或模型。

```bash
COURSE_ROOT="/path/to/course"
ADAPTER="$COURSE_ROOT/classic/lab_adapters/whale_research"
WORK="$COURSE_ROOT/private/classic_lab/whale_research_replay_20261119"

python "$ADAPTER/prepare_frozen_work.py" --repo "$COURSE_ROOT" --verify-only
python "$ADAPTER/prepare_frozen_work.py" --repo "$COURSE_ROOT" --work "$WORK"
python "$ADAPTER/lab_lgb_train.py" --archive "$WORK/lgb_inputs.zip" --receipt "$WORK/lgb_input_receipt.json" --out "$WORK/lgb" --threads 8
python "$ADAPTER/lab_research_run.py" --inputs "$WORK/inputs" --panel "$WORK/panel.csv" --out "$WORK/results"
```

LightGBM 命令会重新训练 2,000 轮。它成功生成 `WORK/lgb/summary.json` 后，研究执行器从同一父目录读取该摘要。小波执行器要求参考 CSV 位于它自己的输出目录 `WORK/results/wavelet_independent_quotes.csv`；改动 `--out` 时须一起调整该文件位置。全流程不需要再次访问行情 API。

## 从实验室共享源重新准备输入

已有共享源授权时可运行以下两条命令，位置参数分别是共享数据根、课程根和冻结日期。它们读取共享源并写入 `private/classic_lab/whale_research/`。固定版本上游的 `notebook/ch08*/stock/company_info.csv` 也必须存在。重新抽取会覆盖该目录的输入与回执，应使用独立课程副本保存新一次实验。

```text
python /path/to/adapter/lab_research_prepare.py SHARE_ROOT COURSE_ROOT 2026-11-19
python /path/to/adapter/lab_lgb_extract.py SHARE_ROOT COURSE_ROOT
```

`check_market_inputs.py WORK` 是可选的**联网核验**入口，会通过 BaoStock 下载独立日线并写入 `WORK/results/`；其日期固定为本次原运行窗口（小波至 2026-11-18，8.2 抽查至 2023-06-30）。它需要 `WORK/inputs/wavelet.csv` 和 `WORK/lgb_inputs.zip`，不属于上述离线复跑流程。若改变运行时点，应另行定义与核验新的合同，不能将新数据写作本次冻结结果。

市值中性化沿用上游“收益减去拟合斜率乘以市值”的定义，保留截距；用独立中心化公式复核。小波案例在训练前核对原股票全部活跃交易日与收盘价。LightGBM 逐标的读取原窗口的必要字段，过滤记录不足 100 条的股票，再按原日期编码生成滞后特征和未来五日标签；代码包含日期连接的抽样交叉核验。全部模型、逐样本预测、行情快照和拟合图只保存在私有运行目录。

本目录按 [WhaleQuant 固定版本](../../sources/whale-quant/README.md) 的实验定义编写适配，保留原作者署名及 CC BY-NC-SA 4.0。