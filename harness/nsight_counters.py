"""Probe permission, then collect Nsight Compute sections for all four kernels.

Exit 2 means counters are unavailable: choose a host/provider allowing them.
Logs and exact commands are preserved; no driver policy changes are attempted.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from harness.measurement import metadata, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='results/nsight/counters')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = {'metadata': metadata(['torch', 'triton']), 'runs': [], 'status': 'pending'}
    ncu = shutil.which('ncu')
    if not ncu:
        result.update(status='unavailable', reason='ncu not installed; install Nsight Compute on a counter-enabled host')
        write_json(root / 'status.json', result)
        raise SystemExit(2)
    combinations = [('triton', 'softmax')] if args.probe_only else [(b, op) for b in ('triton', 'cuda') for op in ('softmax', 'attention')]
    for backend, op in combinations:
        name = f'{backend}-{op}'
        warm_command = [sys.executable, '-m', 'harness.profile_kernels', '--backend', backend, '--op', op, '--n', '1024']
        warm = subprocess.run(warm_command, capture_output=True, text=True)
        (root / f'{name}-warmup.log').write_text(warm.stdout + warm.stderr)
        if warm.returncode:
            result.update(status='warmup_failed', reason=f'{name}: see warmup log')
            write_json(root / 'status.json', result)
            raise SystemExit(1)
        command = [ncu, '--profile-from-start', 'off', '--launch-count', '1',
                   '--section', 'SpeedOfLight', '--section', 'Occupancy', '--section', 'MemoryWorkloadAnalysis',
                   '--section', 'InstructionStats', '--csv', '--log-file', str(root / f'{name}.csv'),
                   '--export', str(root / name), '--force-overwrite',
                   sys.executable, '-m', 'harness.profile_kernels', '--backend', backend, '--op', op, '--n', '1024']
        completed = subprocess.run(command, capture_output=True, text=True)
        (root / f'{name}.log').write_text(completed.stdout + completed.stderr)
        csv_path = root / f'{name}.csv'
        contents = csv_path.read_text() if csv_path.exists() else ''
        log = completed.stdout + completed.stderr + contents
        denied = 'ERR_NVGPUCTRPERM' in log or 'permission to access' in log.lower()
        collected = 'Metric Name' in contents and 'Metric Value' in contents
        row = {'backend': backend, 'op': op, 'command': command, 'warmup_command': warm_command, 'returncode': completed.returncode,
               'status': 'permission_denied' if denied else 'pass' if not completed.returncode and collected else 'failed',
               'csv': str(csv_path), 'report': str(root / f'{name}.ncu-rep')}
        result['runs'].append(row)
        result['status'] = row['status']
        write_json(root / 'status.json', result)
        if denied:
            print('Host denies GPU counters. Use a different counter-enabled host or bare metal.')
            raise SystemExit(2)
        if row['status'] != 'pass':
            raise SystemExit(1)
    result['status'] = 'pass'
    write_json(root / 'status.json', result)


if __name__ == '__main__':
    main()
