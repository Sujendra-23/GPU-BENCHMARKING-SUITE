"""Orchestrates this repo's three separate CLI scripts as one dependency-aware
pipeline: benchmark -> validate -> export, each gated on the previous step's
real exit code (Airflow's default trigger_rule=all_success - no extra config
needed for "only run if the previous step succeeded").

Runs the actual scripts in place, against the actual repo mounted into the
container at /gpu_suite (see ../docker-compose.yaml), with their real CLI
arguments - not re-implementations of their logic in Python.

Manually triggered only (schedule=None): this is a benchmarking suite run on
demand against whatever hardware is attached, not a recurring production job.

Honest limitations of the underlying scripts, not of this DAG (see
../../README.md "Orchestration (Airflow)" for the full picture):

- harness/benchmark.py has no real GPU/TensorRT in this container, so it
  always falls through to its numpy CPU self-test path - that's the script's
  own documented fallback, genuinely executed, not faked by this DAG.
  Running this for real under Airflow's non-root container UID surfaced an
  actual bug on the way to that fallback: query_gpu_metrics() only caught
  FileNotFoundError, but a non-root user that can't even stat /root/bin
  (first on $PATH) gets PermissionError from the PATH search instead, even
  though nvidia-smi is simply absent either way. Fixed by catching OSError.
- harness/validate.py's real (non-self-test) path calls evaluate_real_model(),
  which raises NotImplementedError by design (its DataLoader wiring is left
  abstract - see the script's own docstring) - so --self-test is currently
  the only way this script can pass or fail at all. This DAG uses that mode.
- training/export_tensorrt.py needs a real PyTorch checkpoint, TensorRT's
  Python bindings, and an NVIDIA GPU - none present here, and no checkpoint
  file is committed to this repo. This task is expected to fail in this
  environment; that failure is real, not simulated.
"""
from __future__ import annotations

import pendulum
from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

REPO_ROOT = "/gpu_suite"
OUTPUT_DIR = f"{REPO_ROOT}/orchestration/output"

with DAG(
    dag_id="gpu_suite_pipeline",
    description="benchmark -> validate -> export, gated on real exit codes",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["gpu-benchmarking-suite"],
) as dag:

    run_benchmark = BashOperator(
        task_id="run_benchmark",
        bash_command=(
            f"cd {REPO_ROOT} && "
            f"python harness/benchmark.py "
            f"--model resnet50 --batch-sizes 1,8,32,64 --precisions fp32,fp16 "
            f"--frameworks torch,tensorrt --out {OUTPUT_DIR}/benchmark_report.csv"
        ),
    )

    run_validation = BashOperator(
        task_id="run_validation",
        bash_command=f"cd {REPO_ROOT} && python harness/validate.py --self-test",
    )

    run_tensorrt_export = BashOperator(
        task_id="run_tensorrt_export",
        bash_command=(
            f"cd {REPO_ROOT} && "
            f"python training/export_tensorrt.py "
            f"--checkpoint model.pt --precision fp16 --batch-size 32 "
            f"--out {OUTPUT_DIR}/engines/resnet50_fp16_bs32.trt"
        ),
    )

    run_benchmark >> run_validation >> run_tensorrt_export
