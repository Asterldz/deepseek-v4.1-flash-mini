# DeepSeek-V4.1-Flash-Mini

[中文](README.md) · [MIT License](LICENSE)

A small-scale, text-only PyTorch implementation inspired by DeepSeek-V4.1-Flash, with a default **501,539,072-parameter** backbone for architecture study and training experiments.

Unofficial and unaffiliated with DeepSeek-AI. This repository provides source code; pretrained weights are not included.

## Architecture highlights

The core mechanisms follow DeepSeek-V4.1-Flash. This project's focus is their adaptation and integration at a smaller parameter scale.

```mermaid
flowchart LR
    A[Token Embedding] --> B[4 mHC Streams]
    B --> C[12 Causal Encoder Blocks]
    C --> D[12 Causal Decoder Blocks]
    D --> E[Collapse + RMSNorm]
    E --> F[LM Head]
```

### 1. CED: staged context computation

The backbone uses **24 blocks in a 12+12 causal encoder–decoder split**, with hidden width 640. The encoder organizes compressed memory; the decoder retrieves and reuses shared global KV. Both stages remain causal.

The standard path processes every token through all 24 blocks. An optional `bounded_replay` prefill path encodes the full prompt, builds boundary KV, and runs only the tail window through the decoder. This approximate path is disabled by default; subsequent decode tokens still traverse every block.

### 2. CSA2: compression, sparse retrieval, and cross-layer sharing

Attention combines a **local SWA window and sparsely selected global KV under a joint softmax**. Global memory uses token compression ratio 2 in the encoder and ratio 1 in the decoder. Layer responsibilities are separated:

| Layer mode | Responsibility |
|---|---|
| Full | Build global KV and index keys, then select visible entries |
| Reindex | Reuse global KV and recompute entry selection |
| Reuse | Reuse both global KV and selection |

Some layers can therefore omit global KV construction or index selection, while each layer retains its own local attention. Shared tensors preserve gradients during training.

### 3. MoE: separate expert capacity from per-token activation

Each block has **32 routed experts plus one shared expert**, with top-4 routing. The shared expert is always active. Routing uses `sqrt(softplus)` scores, normalized selected weights, and load-correcting biases; experts use clamped SwiGLU.

This retains expert capacity while limiting the number of experts used per token. Dispatch currently uses reference PyTorch operators, so realized throughput depends on kernels and hardware.

### 4. Single-Pass mHC: multiple residual streams

Residual representations expand into **four streams** with learned read, mixing, and write mappings. **Twenty Sinkhorn iterations** approximately normalize the mixing matrix to be doubly stochastic. Attention and MoE sublayers each use these connections, providing multiple learned paths for information across depth.

### 5. Engram: a discrete-pattern memory path

Two Engram modules use deterministic multi-head hashing of **2/3/4-grams** to retrieve trainable table entries. Context-dependent gates inject retrieved values into residual streams, complementing attention and expert computation. Hash addressing is project-defined.

## Miniature adaptation

The 24-block topology remaps KV sources, index refresh layers, Engram positions, and draft feature sources, while scaling hidden width, expert dimensions, and memory tables. The default backbone has **501,539,072 parameters**; 172M and approximately 1B configurations are also available.

An optional three-layer **DSpark** module consumes backbone features and shares embedding/output weights. It exposes draft prediction, Markov correction, and confidence training interfaces; full speculative decoding is not integrated into the generation CLI.

See [`configs/`](configs/) and [`v41_mini/`](v41_mini/). Implementing these mechanisms does not establish official-model capability or performance parity.

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
