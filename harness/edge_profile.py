"""Batch-1 baseline/capped inference, labeled edge-style, not Jetson.

Records refused power-cap requests and restores the original limit in finally.
Does not change host driver settings. GPU selection uses the physical UUID.
"""
import argparse
import csv
import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from harness.measurement import metadata, positive_int, write_json

LABEL = 'edge-style, not Jetson'


def smi(gpu, *arguments):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), *arguments], capture_output=True,
                            text=True, timeout=15)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}


def query_limit(gpu):
    result = smi(gpu, '--query-gpu=uuid,power.limit,power.min_limit,power.max_limit', '--format=csv,noheader,nounits')
    if result['returncode']:
        raise RuntimeError(str(result))
    rows = list(csv.reader(io.StringIO(result['stdout'])))
    if len(rows) != 1:
        raise RuntimeError('Select exactly one physical GPU')
    uuid, current, minimum, maximum = [v.strip() for v in rows[0]]
    if not uuid.startswith('GPU-'):
        raise RuntimeError('Power profiles require a physical GPU UUID')
    try:
        return {'uuid': uuid, 'current_watts': float(current), 'min_watts': float(minimum), 'max_watts': float(maximum)}
    except ValueError:
        return {'uuid': uuid, 'current_watts': None, 'min_watts': None, 'max_watts': None}


def workload(args, root, name, uuid):
    output = root / f'{name}.csv'
    command = [sys.executable, '-m', 'harness.benchmark', '--model', args.model,
               '--batch-sizes', '1', '--precisions', args.precisions, '--frameworks', args.frameworks,
               '--engine-dir', args.engine_dir, '--iterations', str(args.iterations),
               '--warmup', str(args.warmup), '--out', str(output)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = uuid
    samples, errors = [], []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            start = time.time()
            try:
                reading = smi(uuid, '--query-gpu=power.draw,utilization.gpu,clocks.sm,temperature.gpu', '--format=csv,noheader,nounits')
                if reading['returncode']:
                    errors.append(reading)
                else:
                    samples.append({'timestamp': start, 'power_util_clock_temperature': reading['stdout']})
            except (OSError, subprocess.SubprocessError) as error:
                errors.append(str(error))
            stop.wait(.2)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        with (root / f'{name}.log').open('w') as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    finally:
        stop.set()
        thread.join(timeout=20)
    return {'command': command, 'returncode': status, 'benchmark_csv': str(output),
            'power_samples': samples, 'sampling_errors': errors,
            'power_method': 'whole-board NVML power sampled during workload; not Jetson power/thermal behavior'}


def run_profile(args):
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    original = query_limit(args.gpu)
    result = {'label': LABEL, 'metadata': metadata(['torch', 'tensorrt']), 'config': vars(args),
              'original': original, 'requested_limit_watts': args.power_limit, 'runs': {},
              'cap_status': 'not_attempted', 'restore': None}
    changed_or_attempted = False
    try:
        result['runs']['baseline'] = workload(args, root, 'baseline', original['uuid'])
        write_json(root / 'profile.json', result)
        if result['runs']['baseline']['returncode']:
            raise RuntimeError('Batch-1 baseline failed; see baseline.log')
        if original['current_watts'] is None:
            result['cap_status'] = 'unavailable: original limit cannot be read; no cap attempted'
        elif not original['min_watts'] <= args.power_limit <= original['max_watts']:
            result['cap_status'] = 'refused: requested limit outside supported range'
        else:
            changed_or_attempted = True
            result['cap_request'] = smi(original['uuid'], '-pl', str(args.power_limit))
            actual = query_limit(original['uuid'])
            result['after_request'] = actual
            applied = result['cap_request']['returncode'] == 0 and actual['current_watts'] is not None and abs(actual['current_watts'] - args.power_limit) <= 1
            result['cap_status'] = 'applied' if applied else 'refused_or_unverified'
            name = 'capped' if applied else 'cap-refused-uncapped'
            result['runs'][name] = workload(args, root, name, original['uuid'])
    finally:
        if changed_or_attempted:
            result['restore'] = smi(original['uuid'], '-pl', str(original['current_watts']))
            try:
                result['restored_limit'] = query_limit(original['uuid'])
                result['restore_verified'] = result['restored_limit']['current_watts'] == original['current_watts']
            except Exception as error:
                result['restore_verified'] = False
                result['restore_error'] = str(error)
        write_json(root / 'profile.json', result)
    failed = any(r['returncode'] for r in result['runs'].values()) or result.get('restore_verified') is False
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--power-limit', type=positive_int, default=200)
    parser.add_argument('--model', choices=['resnet18', 'resnet50'], default='resnet18')
    parser.add_argument('--precisions', default='fp32,fp16,int8')
    parser.add_argument('--frameworks', default='tensorrt')
    parser.add_argument('--engine-dir', default='engines')
    parser.add_argument('--iterations', type=positive_int, default=100)
    parser.add_argument('--warmup', type=positive_int, default=10)
    parser.add_argument('--out-dir', default='results/edge')
    args = parser.parse_args()
    raise SystemExit(run_profile(args))


if __name__ == '__main__':
    main()
