import argparse
import json
import torch
from .model import TextMini
from .train import load_config


def main():
    p = argparse.ArgumentParser(description='Verify real module shapes using meta tensors (no weight allocation)')
    p.add_argument('--config', default='configs/v41_text_mini_500m.json')
    args = p.parse_args()
    c = load_config(args.config)
    with torch.device('meta'):
        model = TextMini(c)
    groups = {'embedding': model.embed, 'head': model.head, 'final_norm': model.norm}
    counts = {name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()}
    layers = []
    for i, b in enumerate(model.blocks):
        layers.append(dict(layer=i, stage='encoder' if i < c.encoder_layers else 'decoder',
                           mode='SWA' if i < 2 else 'Full' if i in c.kv_sources else 'Reindex' if i in c.index_sources else 'Reuse',
                           dim=c.dim, experts=f'{c.routed_experts}+1', topk=c.topk_experts,
                           parameters=sum(p.numel() for p in b.parameters()), engram=b.engram is not None))
    print(json.dumps(dict(total_parameters=sum(p.numel() for p in model.parameters()),
                          non_block_parameters=counts, layers=layers), indent=2))


if __name__ == '__main__':
    main()
