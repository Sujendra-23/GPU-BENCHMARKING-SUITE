"""Validation gate: checks a candidate model's accuracy and p95 latency
against thresholds and exits non-zero if either fails, so this can be
wired into Jenkins/CI as a deploy-blocking step (see ../docker/Jenkinsfile).

Real usage loads a trained checkpoint and a held-out validation set:

    python validate.py --checkpoint model.pt --val-data ./val \
        --min-accuracy 0.70 --max-p95-latency-ms 50

Running with --self-test (no arguments needed) exercises the same
threshold-checking and exit-code logic against a synthetic classifier, so
the gate's *decision logic* is verified without needing a trained model or
GPU - see README for what that does and doesn't prove.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass
class ValidationResult:
    accuracy: float
    p95_latency_ms: float
    passed: bool
    reasons: list[str]


def evaluate(accuracy: float, p95_latency_ms: float,
             min_accuracy: float, max_p95_latency_ms: float) -> ValidationResult:
    reasons = []
    if accuracy < min_accuracy:
        reasons.append(f"accuracy {accuracy:.4f} below threshold {min_accuracy:.4f}")
    if p95_latency_ms > max_p95_latency_ms:
        reasons.append(f"p95 latency {p95_latency_ms:.2f}ms exceeds threshold {max_p95_latency_ms:.2f}ms")
    return ValidationResult(accuracy, p95_latency_ms, passed=(len(reasons) == 0), reasons=reasons)


def evaluate_real_model(checkpoint_path: str, val_data_path: str) -> tuple[float, float]:
    """Loads a trained model and a validation set, returns (accuracy, p95_latency_ms).
    Requires torch + torchvision + a real checkpoint; not runnable in this sandbox."""
    import torch
    import time

    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.eval()

    # In a real run this would be a DataLoader over the held-out set; kept
    # abstract here since the point of this function is the shape of the
    # evaluation loop, not a specific dataset format.
    raise NotImplementedError(
        "wire this up to your actual validation DataLoader - "
        "left abstract since it's dataset-specific"
    )


def self_test() -> int:
    """Exercises evaluate() against a few synthetic cases covering all
    branches (pass, accuracy-fail, latency-fail, both-fail) and asserts
    the expected pass/fail outcome for each - a real, if small, test of
    the gate's decision logic."""
    cases = [
        # (accuracy, p95_latency_ms, min_accuracy, max_latency, expect_pass)
        (0.82, 30.0, 0.70, 50.0, True),
        (0.55, 30.0, 0.70, 50.0, False),   # accuracy too low
        (0.82, 90.0, 0.70, 50.0, False),   # latency too high
        (0.40, 120.0, 0.70, 50.0, False),  # both fail
    ]
    all_ok = True
    for accuracy, latency, min_acc, max_lat, expect_pass in cases:
        result = evaluate(accuracy, latency, min_acc, max_lat)
        ok = result.passed == expect_pass
        all_ok &= ok
        status = "OK" if ok else "MISMATCH"
        print(f"[{status}] accuracy={accuracy} latency={latency}ms -> "
              f"passed={result.passed} (expected {expect_pass}) reasons={result.reasons}")
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint")
    parser.add_argument("--val-data")
    parser.add_argument("--min-accuracy", type=float, default=0.70)
    parser.add_argument("--max-p95-latency-ms", type=float, default=50.0)
    parser.add_argument("--self-test", action="store_true",
                         help="run the built-in self-test instead of evaluating a real checkpoint")
    args = parser.parse_args()

    if args.self_test or not args.checkpoint:
        sys.exit(self_test())

    accuracy, p95_latency_ms = evaluate_real_model(args.checkpoint, args.val_data)
    result = evaluate(accuracy, p95_latency_ms, args.min_accuracy, args.max_p95_latency_ms)
    print(result)
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
