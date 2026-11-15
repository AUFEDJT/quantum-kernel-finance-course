# 第三方来源与适用许可

原创代码采用MIT，原创教学文字与图表采用CC BY 4.0。该声明不改变第三方内容的许可，也不授予行情数据再分发权。

| 来源 | 固定依据与使用方式 | 许可处理 |
|---|---|---|
| QuKerNet | `iamconglei-ctrl/QuKerNet`，提交`c9948606145eb85c2bf53f27acebaf5a071a3fa7`；方法复现、结构对照与实现修补 | 原MIT文本保存在`experiments/qukernet_finance/UPSTREAM_LICENSE.txt`；原版权声明保留 |
| L001正式论文 | [ICLR 2024正式版](https://proceedings.iclr.cc/paper_files/paper/2024/file/4d9944ab3330fe6af8efb9260aa9f307-Paper-Conference.pdf) | 提供引用与定位，不随仓库分发论文PDF或复制原图 |
| quant-for-beginners | 借鉴短Notebook教学流程，定位见教学对标文档 | 原项目MIT；实际复制内容应保留其署名 |
| whale-quant、diy-llm、zero-to-sglang | 引用教学组织、数据接口和验收方式；定位见教学对标文档 | 项目中CC BY-NC-SA等内容不得纳入本课程原创MIT/CC BY授权范围；第三方改编须逐项注明 |
| Microsoft Qlib | 提交`54355232463878d2eebb91fe0ee5fa7fa1f5976c`；核读Rolling、Fillna、DropnaLabel、训练期归一化与Alpha158处理器配置，独立实现本课缺失处理 | MIT；见`references/licenses/QLIB_MIT.txt`及[定位与差异](docs/停牌与缺失处理_qlib_missing_v1.md)，不授予行情再分发权 |
| Cqlib、PennyLane、PyTorch等 | 作为独立安装依赖 | 遵循各依赖自身许可，不将依赖代码捆绑改署名 |

公开的实验汇总与图表由本课程生成。原始行情、逐样本金融数据、线路角度、核矩阵、模型参数和云账户信息不在公开仓库内。数据获取接口可用不等于取得数据公开分发许可。
