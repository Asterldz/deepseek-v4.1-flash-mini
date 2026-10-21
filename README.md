# DeepSeek-V4.1-Flash-Mini

[English](README.en.md) · [MIT License](LICENSE)

基于 DeepSeek-V4.1-Flash 架构设计的小参数文本模型 PyTorch 实现，默认主干约 **0.5B（501,539,072）参数**，用于学习、架构研究与训练实验。

非官方实现，与 DeepSeek-AI 无隶属关系。当前提供模型源码，不包含预训练权重。

## 架构

- 24 层，12+12 CED 分段，隐藏维度 640。
- CSA2 压缩稀疏注意力、SWA 与跨层 KV 共享。
- 每层 32 个路由专家＋1 个共享专家，Top-4 激活。
- Single-Pass mHC、Engram，以及可选 DSpark 草稿网络接口。
- 支持预训练、SFT、增量生成和断点恢复。

配置位于 [`configs/`](configs/)，网络实现位于 [`v41_mini/`](v41_mini/)。另提供 172M 和约 1B 的对照配置。

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
