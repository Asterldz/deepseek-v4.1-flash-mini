"""Pretraining / masked SFT entry point for single-process or torchrun DDP/ZeRO-1."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from .config import ModelConfig
from .data import SyntheticDataset, TokenDataset, collate_records
from .model import TextMini
from .optim import HybridOptimizer, parameter_groups
from .tokenization import token_map_fingerprint
from .integrity import verify_dataset


def load_config(value):
    if value in ('tiny', 'tiny-24'):
        return ModelConfig.tiny(layers=24 if value == 'tiny-24' else 40)
    return ModelConfig.smoke() if value == 'smoke' else ModelConfig.from_json(value)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda={'device': device.index, 'state': torch.cuda.get_rng_state(device)} if device.type == 'cuda' else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        if isinstance(state['cuda'], dict):
            torch.cuda.set_rng_state(state['cuda']['state'], state['cuda']['device'])
        else:
            # Older checkpoints stored every GPU's RNG. Each DDP rank only needs
            # its assigned GPU; avoid creating CUDA contexts on the other cards.
            torch.cuda.set_rng_state(state['cuda'][torch.cuda.current_device()])


def schedule(tokens, warmup, decay_start, total):
    if tokens < warmup:
        return max(tokens, 1)/max(warmup, 1)
    if tokens < decay_start:
        return 1.0
    t = min(1.0, max(0.0, (tokens-decay_start)/max(total-decay_start, 1)))
    return 0.1+0.9*0.5*(1+math.cos(math.pi*t))


def save_checkpoint(path, model, optimizer, progress, args, identity, zero1, world, rank, wall_seconds=0.0):
    states = [None]*world
    local_rng = rng_state(model.embed.weight.device)
    if world > 1:
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    if zero1:
        optimizer.consolidate_state_dict(to=0)
    if rank == 0:
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     config=model.config.to_dict(), progress=progress, args=vars(args),
                     rng=states, world_size=world, data_identity=identity, format_version=1,
                     wall_seconds=wall_seconds)
        temporary = path.with_suffix('.tmp')
        torch.save(state, temporary)
        os.replace(temporary, path)
        if path.name.startswith('step-'):
            # Publish the latest complete checkpoint atomically; hard links avoid a second copy.
            latest = path.parent/'latest.pt'
            temporary_link = path.parent/'latest-link.tmp'
            temporary_link.unlink(missing_ok=True)
            try:
                os.link(path, temporary_link)
            except OSError:
                shutil.copyfile(path, temporary_link)
            os.replace(temporary_link, latest)
            keep = getattr(args, 'keep_checkpoints', 0)
            if keep:
                checkpoints = sorted((f for f in path.parent.iterdir()
                                      if re.fullmatch(r'step-\d+\.pt', f.name)),
                                     key=lambda f: int(f.stem.split('-')[1]))
                for old in checkpoints[:-keep]:
                    old.unlink()
    if world > 1:
        dist.barrier()


def main():
    run_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='tiny', help='tiny (40 layers), tiny-24, smoke (~93M), or config JSON')
    p.add_argument('--data', help='Prepared mmap directory')
    p.add_argument('--validation-data', help='Separate held-out mmap directory')
    p.add_argument('--token-map', help='Engram token map JSON made with the tokenizer')
    p.add_argument('--synthetic', action='store_true', help='Explicit pipeline test data only')
    p.add_argument('--output', required=True)
    p.add_argument('--resume', help='Trusted local training checkpoint')
    p.add_argument('--init-from', help='Trusted Base checkpoint for a NEW SFT/continued-training run')
    p.add_argument('--max-steps', type=int, default=10)
    p.add_argument('--max-wall-hours', type=float, default=0,
                   help='Cumulative wall-hour cap across resume, checked at update boundaries; 0 disables. Reserve time for one update and final save.')
    p.add_argument('--seq-len', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=1, help='Per rank microbatch')
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--optimizer', choices=['hybrid', 'adamw'], default='hybrid')
    p.add_argument('--zero1', action='store_true', help='Shard optimizer ownership by whole parameters')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--engram-lr-scale', type=float, default=1.0)
    p.add_argument('--warmup-tokens', type=int, default=1000)
    p.add_argument('--decay-start-tokens', type=int, default=100000)
    p.add_argument('--total-tokens', type=int, default=200000)
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--router-bias-rate', type=float, default=0.001)
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--keep-checkpoints', type=int, default=0,
                   help='Keep this many recent step checkpoints; 0 keeps all. latest.pt is atomic.')
    p.add_argument('--benchmark-no-final-checkpoint', action='store_true',
                   help='Skip the final checkpoint for disposable throughput benchmarks only')
    p.add_argument('--verify-data', action='store_true', help='Verify complete prepared-file hashes and token/label ranges before model allocation')
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--eval-batches', type=int, default=10)
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--hierarchical-index', action='store_true')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if bool(args.data) == args.synthetic:
        p.error('Specify exactly one of --data and --synthetic')
    if args.resume and args.init_from:
        p.error('--resume and --init-from are mutually exclusive')
    if min(args.max_steps, args.batch_size, args.accumulation, args.checkpoint_every, args.eval_every, args.eval_batches, args.threads) <= 0:
        p.error('Step, batch, sequence and interval values must be positive')
    if args.seq_len < 2:
        p.error('--seq-len must be at least 2 for next-token training')
    if args.keep_checkpoints < 0:
        p.error('--keep-checkpoints must be nonnegative')
    if not math.isfinite(args.max_wall_hours) or args.max_wall_hours < 0:
        p.error('--max-wall-hours must be finite and nonnegative')
    if any(not math.isfinite(x) or x <= 0 for x in (args.lr, args.engram_lr_scale, args.clip_grad)):
        p.error('LR, Engram LR scale and gradient clipping must be finite and positive')
    if not math.isfinite(args.router_bias_rate) or args.router_bias_rate < 0:
        p.error('Router bias rate must be finite and nonnegative')
    if not 0 <= args.warmup_tokens <= args.decay_start_tokens < args.total_tokens:
        p.error('Require 0 <= warmup <= decay-start < total-tokens')
    torch.set_num_threads(args.threads)
    world, rank = int(os.getenv('WORLD_SIZE', '1')), int(os.getenv('RANK', '0'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    use_cuda = args.device == 'cuda' or args.device == 'auto' and torch.cuda.is_available()
    device = torch.device('cuda', local_rank) if use_cuda else torch.device('cpu')
    if use_cuda:
        torch.cuda.set_device(device)
        if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('GPU does not support BF16')
    if world > 1:
        dist.init_process_group('nccl' if use_cuda else 'gloo')
    if args.zero1 and world == 1:
        raise ValueError('--zero1 requires torchrun with multiple ranks')
    seed_all(args.seed)
    config = load_config(args.config)
    config.gradient_checkpointing = args.gradient_checkpointing
    config.hierarchical_index = args.hierarchical_index
    if args.seq_len > config.max_seq_len:
        raise ValueError('Requested training length exceeds model context')
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise ValueError('Output directory is nonempty; choose a new directory or use --resume')
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = SyntheticDataset(args.seq_len, config.vocab_size) if args.synthetic else TokenDataset(args.data, args.seq_len, config.vocab_size)
    vd = TokenDataset(args.validation_data, args.seq_len, config.vocab_size) if args.validation_data else None
    if vd is not None:
        for key in ('tokenizer_sha256', 'engram_token_map_sha256'):
            if getattr(dataset, 'meta', {}).get(key) != vd.meta.get(key):
                raise ValueError(f'Validation {key} differs from training data')
        if not args.synthetic and Path(args.data).resolve() == Path(args.validation_data).resolve():
            raise ValueError('Validation must be separate from training data')
    if args.verify_data and not args.synthetic:
        # Only one rank scans disk, but any verification failure reaches every rank.
        verification_error = [None]
        if rank == 0:
            try:
                verify_dataset(dataset)
                if vd is not None:
                    verify_dataset(vd)
            except Exception as exc:
                verification_error[0] = str(exc)
        if world > 1:
            dist.broadcast_object_list(verification_error, src=0)
        if verification_error[0] is not None:
            raise ValueError(verification_error[0])
    identity = {'synthetic': args.synthetic, 'seq_len': args.seq_len,
                'metadata': getattr(dataset, 'meta', None)}
    if not args.synthetic:
        if not args.token_map:
            raise ValueError('Real-data training requires --token-map from the matching tokenizer')
        identity['token_map_sha256'] = hashlib.sha256(Path(args.token_map).read_bytes()).hexdigest()
        expected_map = dataset.meta.get('engram_token_map_sha256')
        if not dataset.meta.get('tokenizer_sha256') or not expected_map:
            raise ValueError('Re-prepare real data with --tokenizer to bind the tokenizer and Engram map, including pretokenized records')
        if token_map_fingerprint(json.loads(Path(args.token_map).read_text())) != expected_map:
            raise ValueError('Engram token map does not match the tokenizer used to prepare this dataset')
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_records,
                        num_workers=0, drop_last=True, generator=torch.Generator().manual_seed(args.seed))
    if not len(loader):
        raise ValueError('Not enough records for the requested world size / batch')
    model = TextMini(config).to(device=device, dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float32)
    if args.token_map:
        model.set_token_map(json.loads(Path(args.token_map).read_text()))
    ckpt = None
    if args.resume or args.init_from:
        ckpt = torch.load(args.resume or args.init_from, map_location='cpu', weights_only=False)
        saved_config = dict(ckpt['config'])
        for key in ('gradient_checkpointing', 'hierarchical_index'):
            saved_config[key] = config.to_dict()[key]
        if saved_config != config.to_dict():
            raise ValueError('Checkpoint architecture differs from requested config')
        if args.token_map and not torch.equal(ckpt['model']['token_map'], model.token_map.cpu()):
            raise ValueError('Checkpoint Engram map differs from supplied tokenizer map')
        previous_meta = ckpt.get('data_identity', {}).get('metadata') or {}
        current_meta = getattr(dataset, 'meta', {})
        if previous_meta.get('tokenizer_sha256') != current_meta.get('tokenizer_sha256'):
            raise ValueError('Checkpoint and dataset use different tokenizers')
        model.load_state_dict(ckpt['model'])
    wrapped = DDP(model, device_ids=[local_rank] if use_cuda else None, broadcast_buffers=False) if world > 1 else model
    groups = parameter_groups(model, args.lr, args.optimizer, args.engram_lr_scale)
    if args.zero1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(groups, optimizer_class=HybridOptimizer, lr=args.lr)
    else:
        optimizer = HybridOptimizer(groups, lr=args.lr)
    progress = dict(step=0, tokens=0, epoch=0, batch_cursor=0)
    if args.resume:
        if ckpt['world_size'] != world or ckpt['data_identity'] != identity:
            raise ValueError('Exact resume requires the same world size and prepared dataset')
        for key in ('batch_size', 'accumulation', 'precision', 'optimizer', 'zero1', 'seed', 'lr',
                    'warmup_tokens', 'decay_start_tokens', 'total_tokens', 'hierarchical_index',
                    'engram_lr_scale', 'router_bias_rate', 'clip_grad'):
            if ckpt['args'][key] != vars(args)[key]:
                raise ValueError(f'Exact resume changed {key}; use --init-from for a new run')
        optimizer.load_state_dict(ckpt['optimizer'])
        if args.zero1:
            # ZeRO-1 loads FP32 state into its owning local optimizer, and the
            # Optimizer base also makes an unused parameter-dtype shadow copy.
            # Consolidation reconstructs the global export from local states.
            optimizer.state.clear()
        progress = ckpt['progress']
    sampler.set_epoch(progress['epoch'])
    iterator = iter(loader)
    for _ in range(progress['batch_cursor']):
        next(iterator)
    if args.resume:
        restore_rng(ckpt['rng'][rank])
    previous_wall_seconds = ckpt.get('wall_seconds', 0.0) if args.resume else 0.0
    del ckpt
    validation = None
    if vd is not None:
        validation = DataLoader(vd, batch_size=args.batch_size, collate_fn=collate_records)
    if rank == 0:
        (output_dir/'resolved_config.json').write_text(json.dumps(config.to_dict(), indent=2)+'\n')
        print(json.dumps(dict(event='start', parameters=sum(p.numel() for p in model.parameters()),
                              world_size=world, device=str(device), precision=args.precision,
                              optimizer=args.optimizer, zero1=args.zero1, data=identity)), flush=True)
    model.train()
    while progress['step'] < args.max_steps and progress['tokens'] < args.total_tokens:
        if args.max_wall_hours:
            stop = torch.tensor(int(previous_wall_seconds+time.perf_counter()-run_started >= args.max_wall_hours*3600), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                if rank == 0:
                    print(json.dumps(dict(event='wall_budget_stop', step=progress['step'], tokens=progress['tokens'])), flush=True)
                break
        started = time.perf_counter()
        batches = []
        for _ in range(args.accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                progress['epoch'] += 1
                progress['batch_cursor'] = 0
                sampler.set_epoch(progress['epoch'])
                iterator = iter(loader)
                batch = next(iterator)
            progress['batch_cursor'] += 1
            batches.append(batch)
        counts = torch.tensor([sum((b['labels'][:, 1:] != -100).sum().item() for b in batches),
                               sum((b['segment_ids'] >= 0).sum().item() for b in batches)], device=device, dtype=torch.long)
        if world > 1:
            dist.all_reduce(counts)
        if not counts[0]:
            raise ValueError('No supervised token in update')
        factor = schedule(progress['tokens']+int(counts[1]), args.warmup_tokens, args.decay_start_tokens, args.total_tokens)
        for group in optimizer.param_groups:
            group['lr'] = args.lr*group['lr_scale']*factor
        optimizer.zero_grad(set_to_none=True)
        route_counts = torch.zeros(config.layers, config.routed_experts, device=device)
        metric = torch.zeros(3, device=device)
        for micro, cpu_batch in enumerate(batches):
            batch = {k: v.to(device) for k, v in cpu_batch.items()}
            supervised = (batch['labels'][:, 1:] != -100).sum()
            sync = wrapped.no_sync() if world > 1 and micro+1 < len(batches) else nullcontext()
            with sync:
                amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.precision == 'bf16' else nullcontext()
                with amp:
                    out = wrapped(**batch, return_logits=False)
                    loss = out.lm_loss*supervised*world/counts[0]
                    loss = loss+(config.index_loss_weight*out.index_loss+config.balance_loss_weight*out.balance_loss)/len(batches)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss')
                loss.backward()
            route_counts += out.routing_counts.detach()
            metric += torch.stack((out.lm_loss.detach()*supervised, out.index_loss.detach()/len(batches), out.balance_loss.detach()/len(batches)))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
        optimizer.step()
        if world > 1:
            dist.all_reduce(route_counts)
            dist.all_reduce(metric)
        model.update_router_bias(route_counts, args.router_bias_rate)
        progress['step'] += 1
        progress['tokens'] += int(counts[1])
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        log = dict(step=progress['step'], tokens=progress['tokens'], lm_loss=float(metric[0]/counts[0]),
                   index_loss=float(metric[1]/world), balance_loss=float(metric[2]/world),
                   lr=args.lr*factor, grad_norm=float(grad_norm), step_seconds=elapsed,
                   aggregate_tokens_per_second=int(counts[1])/elapsed,
                   peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30 if use_cuda else None,
                   peak_cuda_reserved_GiB=torch.cuda.max_memory_reserved()/2**30 if use_cuda else None,
                   dataset_epoch=progress['epoch'],
                   cumulative_wall_seconds=previous_wall_seconds+time.perf_counter()-run_started)
        route_share = route_counts/route_counts.sum(-1, keepdim=True).clamp_min(1)
        log['router_max_share_per_layer'] = route_share.max(-1).values.tolist()
        log['router_unused_experts_per_layer'] = (route_counts == 0).sum(-1).tolist()
        log['router_entropy_per_layer'] = (-(route_share*route_share.clamp_min(1e-20).log()).sum(-1)).tolist()
        if validation is not None and progress['step'] % args.eval_every == 0:
            model.eval()
            values = torch.zeros(2, device=device)
            with torch.no_grad():
                for i, batch in enumerate(validation):
                    if i >= args.eval_batches:
                        break
                    batch = {k: v.to(device) for k, v in batch.items()}
                    n = (batch['labels'][:, 1:] != -100).sum()
                    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.precision == 'bf16' else nullcontext()
                    with amp:
                        out = model(**batch, return_logits=False)
                    values += torch.stack((out.lm_loss*n, n))
            log['validation_lm_loss'] = float(values[0]/values[1].clamp_min(1))
            model.train()
        if rank == 0:
            print(json.dumps(log), flush=True)
            with (output_dir/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(log)+'\n')
        if progress['step'] % args.checkpoint_every == 0:
            save_checkpoint(output_dir/f'step-{progress["step"]:07d}.pt', model, optimizer, progress, args, identity, args.zero1, world, rank,
                            previous_wall_seconds+time.perf_counter()-run_started)
    if not args.benchmark_no_final_checkpoint:
        save_checkpoint(output_dir/'last.pt', model, optimizer, progress, args, identity, args.zero1, world, rank,
                        previous_wall_seconds+time.perf_counter()-run_started)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
