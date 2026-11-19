# 第三方来源与适用许可

原创代码采用MIT，原创教学文字与图表采用CC BY 4.0。该声明不改变第三方内容的许可，也不授予行情数据再分发权。

| 来源 | 固定依据与使用方式 | 许可处理 |
|---|---|---|
| QuKerNet | `iamconglei-ctrl/QuKerNet`，提交`c9948606145eb85c2bf53f27acebaf5a071a3fa7`；方法复现、结构对照与实现修补 | 原MIT文本保存在`experiments/qukernet_finance/UPSTREAM_LICENSE.txt`；原版权声明保留 |
| L001正式论文 | [ICLR 2024正式版](https://proceedings.iclr.cc/paper_files/paper/2024/file/4d9944ab3330fe6af8efb9260aa9f307-Paper-Conference.pdf) | 提供引用与定位，不随仓库分发论文PDF或复制原图 |
| quant-for-beginners | 完整收录固定版本教学材料，出处与许可见下文 | 原项目MIT；实际复制内容应保留其署名 |
| diy-llm、zero-to-sglang | 引用教学组织、数据接口和验收方式；定位见教学对标文档 | 项目中CC BY-NC-SA等内容不得纳入本课程原创MIT/CC BY授权范围；第三方改编须逐项注明 |
| Microsoft Qlib | 提交`54355232463878d2eebb91fe0ee5fa7fa1f5976c`；核读Rolling、Fillna、DropnaLabel、训练期归一化与Alpha158处理器配置，独立实现本课缺失处理 | MIT；见`references/licenses/QLIB_MIT.txt`及[定位与差异](docs/停牌与缺失处理_qlib_missing_v1.md)，不授予行情再分发权 |
| Cqlib、PennyLane、PyTorch等 | 作为独立安装依赖 | 遵循各依赖自身许可，不将依赖代码捆绑改署名 |

公开的实验汇总与图表由本课程生成。本课程自行获取的原始行情、私有逐样本数据、线路角度、核矩阵、模型参数和云账户信息均保留在本地。随上游教材分发的教学样例 CSV 见下表。数据获取接口可用不等于取得数据公开分发许可。

## 固定版本的经典必修教材

| 上游来源 | 本地教材与署名 | 适用许可及本地处理 |
|---|---|---|
| [Quant-for-Beginners 8fbf8942](https://github.com/datawhalechina/quant-for-beginners/tree/8fbf8942afc0e80d01538ee545e72174e099e81a) | [教材与作者说明](classic/sources/quant-for-beginners/README.md)，Yibo Cheng | [MIT 全文与版权声明](classic/sources/quant-for-beginners/LICENSE)。原章、挑战、交互演示和图像保留在源目录。 |
| [WhaleQuant 828009d9](https://github.com/datawhalechina/whale-quant/tree/828009d910db5eb2c06fa14f4439421b235bef44) | [Datawhale 及分章贡献者](classic/sources/whale-quant/README.md) | 依据原 README 的 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 声明保留署名与相同许可。保留 Notebook、Markdown、代码和随附 CSV；源示例中的凭据已改为占位符，并修复了章节目录、配套 CSV 与图片的相对链接。旧 Windows 安装器可从固定上游获取。 |

新章节重新安排教学次序，补写了概念衔接、练习与复现说明。引用或改编自 WhaleQuant 的材料沿用 CC BY-NC-SA 4.0；上游目录中的文件保留原有第三方权利标识。根目录的 MIT 与 CC BY 4.0 声明适用于本项目原创部分。随附行情与图像仍以其各自来源及使用条件为准。
