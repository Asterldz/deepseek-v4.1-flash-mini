"""Read-only CUDA inventory and optional small allocation / BF16 matmul probe."""
import argparse
import json
import platform
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', action='store_true', help='Run a small BF16 matmul on each GPU')
    args = p.parse_args()
    info = dict(python=platform.python_version(), torch=torch.__version__,
                cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(), gpus=[])
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        gpu = dict(index=i, name=prop.name, memory_GiB=prop.total_memory/2**30,
                   capability=list(torch.cuda.get_device_capability(i)))
        with torch.cuda.device(i):
            gpu['bf16_supported'] = torch.cuda.is_bf16_supported()
            if args.probe:
                a = torch.randn(128, 128, device=f'cuda:{i}', dtype=torch.bfloat16)
                gpu['bf16_matmul_finite'] = bool(torch.isfinite(a@a.T).all())
                del a
        info['gpus'].append(gpu)
    print(json.dumps(info, indent=2))
    if not info['cuda_available']:
        print('CUDA unavailable: CPU correctness tests remain usable; no GPU throughput claim.')


if __name__ == '__main__':
    main()
