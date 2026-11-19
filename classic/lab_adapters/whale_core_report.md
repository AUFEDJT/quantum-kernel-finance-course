# WhaleQuant 第 3、5、7 章实验室适配结果

2026-11-19｜16 项均有终态：6 项完成实验室适配；6 项仅完成字段/公式诊断且原完整接口合同终止；4 项在输入合同门禁处终止。

实验室行情替换后的教学计算；不是原在线供应商数值复现、投资建议或量子优势证据。逐行输入输出、订单和完整回执留在 private。

[合同](whale_core_contracts.json) · [执行器](whale_core.py) · [机器可读结果](whale_core_results.json)

无影冻结输入执行，计算阶段禁用网络。11 个输入文件哈希、两次回传包哈希全部通过。保留首次均线验证的舍入误报及仅重跑该项的记录；未改原 pandas 均线舍入逻辑。

运行环境：Python 3.10.12；NumPy 1.26.3；pandas 2.1.4；SciPy 1.13.1；TA-Lib 0.8.3；Backtrader 1.9.78.123。

## W03_daily

3.3.1 日线｜partial｜原合同：terminated

同标的、窗口的 125 行日线已读；缺少 turn、tradestatus、isST，原完整接口合同终止。

输入行数：125。

缺失字段：`turn`、`tradestatus`、`isST`。

## W03_minute5

3.3.1 分钟线｜partial｜原合同：terminated

原完整接口合同终止。1008 个时间戳及 OHLC 与原输出完全相同；volume×100 后仍有 304 行差异（最大 92 股），amount 有 324 行差异（最大 16 元）。仅完成显式聚合与字段诊断。

输入行数：1008。

## W03_index

3.3.1 指数｜completed｜原合同：completed_with_lab_data

同上证指数、同窗口日线完成；preclose 与 pctChg 由完整日序列前一观测计算，未声称供应商响应逐值等价。

输入行数：117。

## W03_sz50

3.3.1 上证50成分｜blocked｜原合同：terminated

终止：实验室候选表是历史持仓/净值产品，不能证明查询时点的权威上证 50 成分、名称与 updateDate。

## W03_profit

3.3.2 profit｜partial｜原合同：terminated

原接口合同终止：同标的 2022Q4 原始银行财报及公告版本可读，但未建立 BaoStock 盈利能力派生字段映射。

输入行数：5。

缺失字段：`roeAvg`、`npMargin`、`gpMargin`、`netProfit`、`epsTTM`、`MBRevenue`、`totalShare`、`liqaShare`。

## W03_operation

3.3.2 operation｜partial｜原合同：terminated

原接口合同终止：同标的 2022Q4 原始银行财报可读，但未建立营运能力派生字段映射。

输入行数：5。

缺失字段：`NRTurnRatio`、`NRTurnDays`、`INVTurnRatio`、`INVTurnDays`、`CATurnRatio`、`AssetTurnRatio`。

## W03_growth

3.3.2 growth｜partial｜原合同：terminated

原接口合同终止：同标的 2022Q4 原始银行财报可读，但未建立成长能力派生字段映射。

输入行数：5。

缺失字段：`YOYEquity`、`YOYAsset`、`YOYNI`、`YOYEPSBasic`、`YOYPNI`。

## W03_balance

3.3.2 balance｜partial｜原合同：terminated

原接口合同终止：同标的 2022Q4 原始银行财报可读，但未建立偿债能力派生字段映射。

输入行数：5。

缺失字段：`currentRatio`、`quickRatio`、`cashRatio`、`YOYLiability`、`liabilityToAsset`、`assetToEquity`。

## W03_603019

3.5 TuShare均线信号｜blocked｜原合同：terminated

终止：原 get_hist_data 未指定窗口与排序；实验室有该股票，但不能从最新截止日反推原 API 默认输入。未另换窗口运行。

## W05_ma_values

5.3.3｜completed｜原合同：completed_with_lab_data

按原窗口计算 M5/M10/M20/M30，逐窗口未舍入平均值经独立求和核验，展示仍用原 pandas 两位小数舍入。

输入行数：40。

## W05_ma_strategy

5.3.5｜completed｜原合同：completed_with_lab_data

保留原 signal.diff 仓位公式，分别记录升序与降序输入；另存仅做多持仓诊断。原 position 是信号变化而非持有仓位，输出不作为有效交易业绩。

输入行数：603。

原公式升序末值 1.123079169，降序末值 0.914096646；额外仅做多诊断末值 0.518417703。三个结果分别保留，未互相替代。

## W05_macd

5.4.5｜completed｜原合同：completed_with_lab_data

同沪深 300、同 2015–2018 窗口；TA-Lib 默认 12/26/9，信号滞后一期，无费用，核验 MACD=DIF−DEA。

输入行数：975。

策略归一化末值 1.133942703，价格基准末值 0.826752630。

## W07_metrics

7.1｜completed｜原合同：completed_with_lab_data

同三只股票与沪深 300、同 2018–2022 窗口；保留原 1826 分母及 sqrt(N) 比率，同时另列 252 交易日诊断。使用原价，未将其解释为含分红总回报。

输入行数：1215。

| 序列 | 原年化公式 | 交易日年化诊断 | 原夏普 | 252 缩放夏普 |
|---|---:|---:|---:|---:|
| sh600519 | 13.1870% | 20.4804% | 1.404549 | 0.639923 |
| sh601398 | -4.7607% | -7.0741% | -1.052569 | -0.479558 |
| sh601318 | -5.8225% | -8.6279% | -0.612243 | -0.278943 |
| sh000300_index | -0.7457% | -1.1194% | -0.212935 | -0.097015 |

## W07_backtrader

7.3｜completed｜原合同：completed_with_lab_data

严格按实际代码使用 000001.SZ、10000 初始现金、固定 100 股、1% 佣金、MA5/10 与下一根开盘成交；核验订单、佣金、现金及末值。

输入行数：1215。

完成订单 133 笔；末值 7847.890000。独立逐笔现金/仓位账本与 broker 末值一致，所有成交价匹配当日开盘，佣金匹配 1%。

## W07_T888_1d

7.5 StupidHead｜blocked｜原合同：terminated

终止：未找到可证明同一 T888 国债连续合约及换月/复权口径的实验室日线；教材附带原 CSV 保留，未用股指期货替代。

## W07_T888_15m

7.5 StupidHead｜blocked｜原合同：terminated

终止：未找到相同 T888 连续合约、换月/复权规则、15 分钟频率与窗口的实验室输入。

## 复跑

输入为经授权取得并冻结的私有 CSV 与 manifest.json。公开仓库不包含行情。

```bash
python classic/lab_adapters/whale_core.py --input-dir /path/to/private/inputs --output-dir /path/to/private/outputs
```

可用 `--case W05_ma_values` 只运行一个合同。输入和输出目录必须分离；执行器验证输入 SHA256，合同不满足时终止该项。
