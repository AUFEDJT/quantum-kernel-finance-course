# 来源与修改说明

上游：[Datawhale WhaleQuant](https://github.com/datawhalechina/whale-quant/tree/828009d910db5eb2c06fa14f4439421b235bef44)，固定提交 `828009d910db5eb2c06fa14f4439421b235bef44`。原作者和贡献者的署名继续保留在 `classic/sources/whale-quant`。上游 README 声明采用 [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)。

本目录中的上游派生代码、修改说明与实验说明沿用 CC BY-NC-SA 4.0。课程其他区域的许可不能替代这一上游许可。公开行情提供方的数据条款需独立遵守；本目录不分发新取得的行情快照。

`StupidHead.py` 来自上游第 7 章 Markdown 内的完整框架代码。修改包括显式 pandas `.loc` 赋值、Linux 绘图字体兼容。`recover_experiments.py` 从固定教学源抽取原算法与回调，并记录原来源；恢复调用中对 TA-Lib 参数采用整数周期，对 pandas 配置/模块导入和 Matplotlib 样式作兼容处理。原始源正文保留，教学副本的凭据占位和链接修复另由项目导入记录说明。

`inventory.py`、`run_workflows.py` 和 `recover_experiments.py` 是本课程在 2026-11-19 增补的执行与检查辅助工具。替代数据源、明确的局部目标验证和平台模拟均在结果中单独标识，未把它们视为原平台完整复现。
