"""Run TensorRT-LLM TP1 and matched vLLM TP1/TP2 on a prepared two-GPU host."""
import argparse
import os
import subprocess
from pathlib import Path

from harness.measurement import metadata, positive_int, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trtllm-python', default='.venv-trtllm-cu12/bin/python')
    parser.add_argument('--vllm-python', default='.venv-vllm/bin/python')
    parser.add_argument('--stages', default='trtllm,tp1,tp2')
    parser.add_argument('--requests', type=positive_int, default=32)
    parser.add_argument('--out-dir', default='results/h100-dual')
    args = parser.parse_args()
    if args.requests < 16:
        parser.error('At least 16 requests are required for concurrency 1/4/8/16')
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    import json
    previous = json.loads((root / 'status.json').read_text()).get('stages', {}) if (root / 'status.json').exists() else {}
    report = {'metadata': metadata(), 'config': vars(args), 'stages': previous}
    work = {'trtllm': (args.trtllm_python, 'tensorrt-llm', 1),
            'tp1': (args.vllm_python, 'vllm', 1), 'tp2': (args.vllm_python, 'vllm', 2)}
    for stage in args.stages.split(','):
        if stage not in work:
            parser.error(f'Unknown stage {stage}')
        interpreter, backend, tp = work[stage]
        command = [interpreter, '-u', '-m', 'serving.run', '--backend', backend,
                   '--tensor-parallel-size', str(tp), '--requests', str(args.requests),
                   '--startup-timeout', '1200', '--out-dir', str(root / 'serving')]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0,1' if tp == 2 else '0'
        env['OMP_NUM_THREADS'] = '4'
        env['CUDA_HOME'] = '/usr/local/cuda'
        env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '')
        if backend == 'tensorrt-llm':
            env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        log_path = root / f'{stage}.log'
        print(f'Running {stage}: {command}', flush=True)
        with log_path.open('w') as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        report['stages'][stage] = [{'command': command, 'returncode': status,
                                   'cuda_visible_devices': env['CUDA_VISIBLE_DEVICES'], 'log': str(log_path)}]
        write_json(root / 'status.json', report)
    raise SystemExit(1 if any(row['returncode'] for rows in report['stages'].values() for row in rows) else 0)


if __name__ == '__main__':
    main()
