from dataclasses import asdict, dataclass, field, fields, replace
import json
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = "V4.1-Text-Mini"
    layers: int = 40
    encoder_layers: int = 20
    dim: int = 1024
    vocab_size: int = 32768
    expert_dim: int = 384
    routed_experts: int = 16
    shared_experts: int = 1
    topk_experts: int = 2
    routed_scale: float = 1.5
    swiglu_clamp: float = 10.0
    heads: int = 16
    head_dim: int = 128
    rope_dim: int = 32
    q_rank: int = 256
    output_groups: int = 4
    output_rank: int = 128
    swa_window: int = 128
    compress_ratios: list[int] = field(default_factory=lambda: [0]*2+[2]*18+[1]*20)
    kv_sources: list[int] = field(default_factory=lambda: [2, 8, 14, 20])
    index_sources: list[int] = field(default_factory=lambda: [2, 8, 14, 20, 24, 28, 32, 36])
    index_heads: int = 8
    index_dim: int = 32
    index_topk: int = 128
    candidate_source_layer: int = 20
    candidate_blocks: int = 64
    candidate_block_size: int = 8
    hc_streams: int = 4
    hc_sinkhorn_iterations: int = 20
    norm_eps: float = 1e-6
    engram_layers: list[int] = field(default_factory=lambda: [1, 14])
    engram_orders: list[int] = field(default_factory=lambda: [2, 3, 4])
    engram_heads_per_order: int = 8
    engram_head_dim: int = 32
    engram_bucket_base: int = 32768
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    query_chunk: int = 32
    loss_chunk: int = 128
    index_loss_queries: int = 8
    index_loss_weight: float = 0.01
    balance_loss_weight: float = 1e-4
    gradient_checkpointing: bool = False
    hierarchical_index: bool = False
    dspark_layers: int = 3
    dspark_block_size: int = 5
    dspark_markov_rank: int = 128
    dspark_target_layers: list[int] = field(default_factory=lambda: [37, 38, 39])
    dspark_noise_token_id: int = 3

    def __post_init__(self):
        topologies = {
            (40, 20): ([2, 8, 14, 20], [2, 8, 14, 20, 24, 28, 32, 36], [1, 14]),
            (24, 12): ([2, 6, 10, 12], [2, 6, 10, 12, 14, 16, 18, 20], [1, 10]),
        }
        if (self.layers, self.encoder_layers) not in topologies:
            raise ValueError("Supported CED topologies are 40-layer / 20+20 and 24-layer / 12+12")
        kv_sources, index_sources, engram_layers = topologies[self.layers, self.encoder_layers]
        if self.compress_ratios != [0]*2+[2]*(self.encoder_layers-2)+[1]*(self.layers-self.encoder_layers):
            raise ValueError("Unexpected CED compression schedule")
        if self.kv_sources != kv_sources or self.index_sources != index_sources:
            raise ValueError("Unexpected CSA2 sharing schedule")
        if self.candidate_source_layer != self.encoder_layers:
            raise ValueError("Candidate source must be the decoder boundary")
        if self.shared_experts != 1 or not 1 <= self.topk_experts <= self.routed_experts:
            raise ValueError("Invalid expert configuration")
        if self.heads % self.output_groups or self.rope_dim % 2:
            raise ValueError("Invalid grouped output / RoPE dimensions")
        if self.rope_dim > min(self.head_dim, self.index_dim):
            raise ValueError("RoPE dimension exceeds attention/indexer dimension")
        if self.engram_orders != [2, 3, 4] or self.engram_layers != engram_layers:
            raise ValueError("Unexpected Engram topology")
        if (not self.dspark_target_layers or self.dspark_target_layers != sorted(set(self.dspark_target_layers))
                or min(self.dspark_target_layers) < 0 or max(self.dspark_target_layers) >= self.layers):
            raise ValueError("DSpark target layers must be ordered, unique backbone layer indices")
        for key in ("dim", "vocab_size", "expert_dim", "swa_window", "query_chunk", "loss_chunk", "index_topk", "hc_sinkhorn_iterations"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.candidate_blocks*self.candidate_block_size < self.index_topk:
            raise ValueError("Candidate pool must contain at least index_topk positions")

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        # Explicit adapter for the existing, human-readable proposal (not an HF config).
        if data.get("vision_enabled", False) or data.get("tie_embeddings", False):
            raise ValueError("Only text-only, untied embedding/head is supported")
        ds = data.get("dspark", {})
        aliases = {"layers": "dspark_layers", "block_size": "dspark_block_size",
                   "markov_rank": "dspark_markov_rank", "target_layers": "dspark_target_layers"}
        for key, target in aliases.items():
            if key in ds:
                data[target] = ds[key]
        if "target_context" in data:
            data["max_seq_len"] = data["target_context"]
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def to_dict(self):
        return asdict(self)

    @classmethod
    def compact(cls):
        """Budget model: preserve mechanisms with a remapped 12+12 CED topology."""
        return cls(name="V4.1-Text-Mini-0.5B", layers=24, encoder_layers=12,
                   dim=640, expert_dim=288, routed_experts=32, topk_experts=4, heads=10, head_dim=64, rope_dim=16,
                   q_rank=160, output_groups=2, output_rank=64, index_heads=4, index_dim=16,
                   compress_ratios=[0]*2+[2]*10+[1]*12,
                   kv_sources=[2, 6, 10, 12], index_sources=[2, 6, 10, 12, 14, 16, 18, 20],
                   candidate_source_layer=12, engram_layers=[1, 10],
                   engram_head_dim=16, engram_bucket_base=8192,
                   dspark_markov_rank=64, dspark_target_layers=[21, 22, 23])

    @classmethod
    def tiny(cls, layers=40):
        """CPU correctness model retaining the selected topology's sharing boundaries."""
        if layers not in (24, 40):
            raise ValueError("tiny supports 24 or 40 layers")
        base = cls.compact() if layers == 24 else cls()
        return replace(base, dim=32, vocab_size=128, expert_dim=16, routed_experts=8 if layers == 24 else 4,
                   heads=2, head_dim=16, rope_dim=8, q_rank=16,
                   output_groups=1, output_rank=16, index_heads=2, index_dim=8,
                   index_topk=4, swa_window=8, candidate_blocks=2, candidate_block_size=4,
                   engram_head_dim=4, engram_bucket_base=17, max_seq_len=256,
                   query_chunk=8, loss_chunk=16, index_loss_queries=2,
                   dspark_markov_rank=8)

    @classmethod
    def smoke(cls):
        """The proposal's ~93M same-topology training probe."""
        return replace(cls(), dim=256, expert_dim=128, heads=4, head_dim=64,
                       rope_dim=16, q_rank=64, output_groups=1, output_rank=64,
                       index_heads=4, engram_bucket_base=1024)
