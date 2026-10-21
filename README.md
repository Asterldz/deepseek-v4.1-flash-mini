# DeepSeek-V4.1-Flash-Mini

[English](README.en.md) · [MIT License](LICENSE)

基于 DeepSeek-V4.1-Flash 架构设计的小参数文本模型 PyTorch 实现，默认主干约 **0.5B（501,539,072）参数**，用于学习、架构研究与训练实验。

非官方实现，与 DeepSeek-AI 无隶属关系。当前提供模型源码，不包含预训练权重。

## 架构亮点

核心机制参考 DeepSeek-V4.1-Flash，本项目着重于这些机制的小参数适配与组合实现。

```mermaid
flowchart LR
    A[Token Embedding] --> B[4 mHC Streams]
    B --> C[12 Causal Encoder Blocks]
    C --> D[12 Causal Decoder Blocks]
    D --> E[Collapse + RMSNorm]
    E --> F[LM Head]
```

### 1. CED：分段组织上下文计算

24 层采用 **12+12 因果编码器—解码器分段**，隐藏维度 640。前半段组织压缩记忆，后半段围绕共享的全局 KV 进行检索与复用；两段均保持因果约束。

标准路径让每个 token 经过全部 24 层。另实现可选 `bounded_replay`：全量输入经过编码器并构建边界 KV，仅将尾部窗口送入解码器。这是一条默认关闭的近似 prefill 路径，新增 token 的 decode 仍经过全部层。

### 2. CSA2：压缩、稀疏检索与跨层共享

注意力同时使用 **局部 SWA 窗口＋稀疏选择的全局 KV**，并在同一 softmax 中计算权重。全局记忆在编码器侧采用 2 倍 token 压缩，解码器侧倍率为 1；跨层职责拆分为：

| 层类型 | 工作 |
|---|---|
| Full | 构建全局 KV 与索引键，并选择可见条目 |
| Reindex | 共享已有 KV，重新计算条目选择 |
| Reuse | 共享已有 KV 和选择结果 |

这种设计让部分层省去全局 KV 构建或索引选择，各层仍保留独立的局部注意力。共享张量在训练时保留梯度。

### 3. MoE：将专家容量与每 token 激活量分开

每层包含 **32 个路由专家＋1 个共享专家**，每 token 激活 Top-4 路由专家，共享专家始终参与计算。路由采用 `sqrt(softplus)` 评分、选中权重归一化和负载校正偏置；每个专家使用带限幅的 SwiGLU。

该结构允许保留较多专家参数，同时控制每 token 使用的专家数量。当前专家分发使用 PyTorch 参考实现，实际吞吐取决于算子与硬件。

### 4. Single-Pass mHC：多流残差连接

将残差表示扩展为 **4 条流**，通过可学习映射完成流间读取、混合与写入。混合矩阵使用 **20 次 Sinkhorn 迭代**进行近似双随机归一化，为信息跨层传递提供多条可学习路径；注意力和 MoE 子层分别接入该机制。

### 5. Engram：增加离散模式的记忆通路

两个 Engram 模块对 **2/3/4-gram** 做确定性多头哈希查表，再通过上下文门控将检索值注入残差流。它为局部 token 模式提供可学习记忆，与注意力和专家网络互补。哈希寻址由本项目实现。

## 小参数适配

在 24 层中重新安排 KV 来源、索引刷新、Engram 位置和草稿特征来源，并缩放隐藏宽度、专家维度及记忆表容量。默认主干为 **501,539,072 参数**，另提供 172M 和约 1B 配置。

可选三层 **DSpark** 使用主干特征，并共享词嵌入与输出头，提供草稿预测、Markov 修正和置信度训练接口；完整推测解码尚未接入生成 CLI。

配置见 [`configs/`](configs/)，网络实现见 [`v41_mini/`](v41_mini/)。这些机制的实现不代表已验证官方级别的模型能力或性能。

## 快速开始

建议 Python 3.12，并安装适合运行平台的 [PyTorch](https://pytorch.org/get-started/locally/)。

```bash
git clone https://github.com/Asterldz/deepseek-v4.1-flash-mini.git
cd deepseek-v4.1-flash-mini
python -m pip install -e '.[dev,data]'

# 检查完整模型的结构和参数量，不分配真实权重
python -m v41_mini.inspect --config configs/v41_text_mini_500m.json

# 运行测试
python -m pytest -q
```

训练入口为 `python -m v41_mini.train --help`，数据处理与启动脚本位于 [`scripts/`](scripts/)。

## 参考与许可

参考 [DeepSeek-V4.1-Flash 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)。项目采用 [MIT License](LICENSE)，第三方材料与依赖遵循各自许可，见 [说明](THIRD_PARTY_NOTICES.md)。
