"""Polls nvidia-smi at a fixed interval and logs power draw, utilization,
temperature, and clock speeds to CSV - meant to run alongside a benchmark
so power/thermal behavior can be correlated with throughput afterward.

    python power_monitor.py --interval 0.5 --out power_log.csv &
    MONITOR_PID=$!
    python benchmark.py ...
    kill $MONITOR_PID

On a machine without nvidia-smi (like the sandbox this was built in), this
prints a warning and exits cleanly rather than crashing a benchmark run
that happens to be piped through it - verified by actually running it here.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time

FIELDS = ["timestamp", "utilization_gpu_pct", "utilization_mem_pct",
          "power_draw_w", "temperature_c", "clocks_sm_mhz", "clocks_mem_mhz"]


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def poll_once() -> dict:
    query = ",".join([
        "utilization.gpu", "utilization.memory", "power.draw",
        "temperature.gpu", "clocks.sm", "clocks.mem",
    ])
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        stderr=subprocess.DEVNULL, timeout=2,
    ).decode().strip()
    util_gpu, util_mem, power, temp, clk_sm, clk_mem = [v.strip() for v in out.split(",")]
    return {
        "timestamp": time.time(),
        "utilization_gpu_pct": util_gpu,
        "utilization_mem_pct": util_mem,
        "power_draw_w": power,
        "temperature_c": temp,
        "clocks_sm_mhz": clk_sm,
        "clocks_mem_mhz": clk_mem,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between samples")
    parser.add_argument("--out", default="power_log.csv")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = run until Ctrl-C")
    args = parser.parse_args()

    if not nvidia_smi_available():
        print("power_monitor: nvidia-smi not found on this system - "
              "no NVIDIA GPU/driver present, nothing to monitor. Exiting cleanly.",
              file=sys.stderr)
        sys.exit(0)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        count = 0
        try:
            while args.max_samples == 0 or count < args.max_samples:
                try:
                    writer.writerow(poll_once())
                    f.flush()
                except subprocess.CalledProcessError as e:
                    print(f"power_monitor: nvidia-smi query failed: {e}", file=sys.stderr)
                count += 1
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass

    print(f"power_monitor: wrote {count} samples to {args.out}")


if __name__ == "__main__":
    main()
