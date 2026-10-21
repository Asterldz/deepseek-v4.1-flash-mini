"""Opt-in loopback test; starts only tiny CPU workers, never a production run."""
import os
import platform
import socket
import subprocess
import sys
import pytest
import torch


@pytest.mark.skipif(os.getenv('V41_RUN_DISTRIBUTED_TESTS') != '1',
                    reason='Set V41_RUN_DISTRIBUTED_TESTS=1 to permit local loopback workers')
@pytest.mark.parametrize('config', ['tiny', 'tiny-24'])
def test_bf16_ddp_zero1_resume_matches_uninterrupted_run(tmp_path, config):
    def run(name, steps, resume=False):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        directory = tmp_path/name
        command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=2',
                   '--master-addr=127.0.0.1', f'--master-port={port}', '-m', 'v41_mini.train',
                   '--config', config, '--synthetic', '--device', 'cpu', '--precision', 'bf16',
                   '--zero1', '--gradient-checkpointing', '--seq-len', '12', '--accumulation', '2',
                   '--checkpoint-every', '2', '--keep-checkpoints', '2', '--max-wall-hours', '18',
                   '--threads', '1', '--max-steps', str(steps), '--output', str(directory)]
        if resume:
            command += ['--resume', str(directory/'last.pt')]
        env = dict(os.environ)
        env.setdefault('GLOO_SOCKET_IFNAME', 'lo0' if platform.system() == 'Darwin' else 'lo')
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout+'\n'+result.stderr
        return torch.load(directory/'last.pt', map_location='cpu', weights_only=False)

    expected = run('full', 4)
    run('split', 2)
    actual = run('split', 4, resume=True)
    assert expected['progress'] == actual['progress']
    for key, value in expected['model'].items():
        torch.testing.assert_close(value, actual['model'][key], rtol=0, atol=0, msg=key)
        if key.endswith('routing_bias'):
            assert value.dtype == torch.float32
    for key, values in expected['optimizer']['state'].items():
        for field, value in values.items():
            other = actual['optimizer']['state'][key][field]
            if isinstance(value, torch.Tensor):
                assert value.dtype == other.dtype == torch.float32
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other
