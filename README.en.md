# DeepSeek-V4.1-Flash-Mini

[中文](README.md) · [MIT License](LICENSE)

A small-scale, text-only PyTorch implementation inspired by DeepSeek-V4.1-Flash, with a default **501,539,072-parameter** backbone for architecture study and training experiments.

Unofficial and unaffiliated with DeepSeek-AI. This repository provides source code; pretrained weights are not included.

## Architecture

- 24 causal blocks with a 12+12 CED split; hidden width 640.
- CSA2 compressed sparse attention, SWA, and cross-layer KV sharing.
- 32 routed experts plus one shared expert per block; top-4 routing.
- Single-Pass mHC, Engram, and optional DSpark module interfaces.
- Pretraining, SFT, incremental generation, and checkpoint resuming.

See [`configs/`](configs/) for the default, 172M, and approximately 1B configurations, and [`v41_mini/`](v41_mini/) for the model implementation.

## Quick start

Python 3.12 is recommended. Install a platform-appropriate [PyTorch build](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/Asterldz/deepseek-v4.1-flash-mini.git
cd deepseek-v4.1-flash-mini
python -m pip install -e '.[dev,data]'

# Inspect full-size dimensions without allocating model weights
python -m v41_mini.inspect --config configs/v41_text_mini_500m.json
python -m pytest -q
```

Use `python -m v41_mini.train --help` for training options. Data utilities and launchers are in [`scripts/`](scripts/).

## References and license

Inspired by the [DeepSeek-V4.1-Flash technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf). Project code is [MIT-licensed](LICENSE); external materials and dependencies retain their own terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
