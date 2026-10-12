"""Performance-per-watt report from recorded benchmark rows.

Reads a benchmark CSV in the format written by benchmark.py
(framework,batch_size,precision,...,throughput_samples_per_sec,...,gpu_power_watts)
and reports, per framework/batch size/precision:

    images/s/W   = throughput_samples_per_sec / gpu_power_watts
    J/image      = gpu_power_watts / throughput_samples_per_sec

plus the fp16-vs-fp32 ratios at each batch size. For an inference benchmark one
sample is one image, so samples/s is images/s.

Only measured rows are used. Rows without a power reading (for example the
numpy CPU self-test fallback, which has no GPU power) are skipped and counted,
never imputed.

    python harness/perf_per_watt.py results/resnet50_gpu_benchmark.csv \
        --markdown results/perf_per_watt.md --csv results/perf_per_watt.csv
    python harness/perf_per_watt.py --self-test
    python harness/perf_per_watt.py --check-readme   # CSV matches README table
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Row:
    framework: str
    batch: int
    precision: str
    throughput: float  # images/s
    watts: float

    @property
    def images_per_s_per_w(self) -> float:
        return self.throughput / self.watts

    @property
    def joules_per_image(self) -> float:
        return self.watts / self.throughput


def load_rows(path: Path) -> tuple[list[Row], int]:
    """Returns (usable rows, number of rows skipped for a missing/zero reading)."""
    rows: list[Row] = []
    skipped = 0
    with open(path, newline="") as handle:
        for record in csv.DictReader(handle):
            try:
                throughput = float(record["throughput_samples_per_sec"])
                watts = float(record["gpu_power_watts"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if throughput <= 0 or watts <= 0:
                skipped += 1
                continue
            rows.append(Row(record["framework"], int(record["batch_size"]),
                            record["precision"], throughput, watts))
    return rows, skipped


def ratios(rows: list[Row]) -> list[dict]:
    """fp16 relative to fp32 for every (framework, batch) that has both."""
    by_key = {(r.framework, r.batch, r.precision): r for r in rows}
    out = []
    for (framework, batch, precision), fp16 in sorted(by_key.items()):
        if precision != "fp16":
            continue
        fp32 = by_key.get((framework, batch, "fp32"))
        if fp32 is None:
            continue
        out.append({
            "framework": framework,
            "batch": batch,
            "throughput_ratio": fp16.throughput / fp32.throughput,
            "power_ratio": fp16.watts / fp32.watts,
            "perf_per_watt_ratio": fp16.images_per_s_per_w / fp32.images_per_s_per_w,
        })
    return out


def render_markdown(rows: list[Row], skipped: int, source: str) -> str:
    lines = [
        "# Performance per watt (ResNet-50 inference)",
        "",
        f"Computed by `harness/perf_per_watt.py` from `{source}`. Every input value is",
        "a recorded measurement from the GPU run described in the README; nothing is",
        "estimated or imputed. `images/s/W = throughput / gpu_power_watts` and",
        "`J/image = gpu_power_watts / throughput`.",
        "",
        "| framework | batch | precision | images/s | power (W) | images/s/W | J/image |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda r: (r.framework, r.batch, r.precision)):
        lines.append(f"| {r.framework} | {r.batch} | {r.precision} | {r.throughput:.2f} | "
                     f"{r.watts:.2f} | {r.images_per_s_per_w:.3f} | {r.joules_per_image:.4f} |")
    lines += ["", "## fp16 vs fp32", "",
              "| framework | batch | throughput x | power x | images/s/W x |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for q in ratios(rows):
        lines.append(f"| {q['framework']} | {q['batch']} | {q['throughput_ratio']:.2f} | "
                     f"{q['power_ratio']:.2f} | {q['perf_per_watt_ratio']:.2f} |")
    lines += [
        "",
        "## Caveats",
        "",
        f"- Rows skipped for a missing or non-positive power/throughput value: {skipped}.",
        "- Power is `nvidia-smi power.draw` sampled during the sweep on a MIG 1g.24gb slice of an",
        "  RTX PRO 6000 Blackwell Server Edition. `power.draw` may describe the whole board rather",
        "  than the slice (not checked in the recorded run), while the workload used only a slice,",
        "  so absolute images/s/W is not a statement about the full GPU's efficiency. The",
        "  fp16-vs-fp32 ratios at the same batch size compare like with like and are the more",
        "  meaningful figures.",
        "- One run per configuration; no confidence intervals, so small differences are not",
        "  statistically supported. Batch 1 fp16 is slower than fp32 in the recorded run.",
        "- The README's TensorRT result (batch 8 fp32, 98.24 W) is a different framework and engine",
        "  and is not included in the table above.",
        "",
    ]
    return "\n".join(lines)


def write_csv(rows: list[Row], path: Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["framework", "batch_size", "precision", "images_per_s",
                         "power_w", "images_per_s_per_w", "joules_per_image"])
        for r in sorted(rows, key=lambda r: (r.framework, r.batch, r.precision)):
            writer.writerow([r.framework, r.batch, r.precision, f"{r.throughput:.2f}",
                             f"{r.watts:.2f}", f"{r.images_per_s_per_w:.4f}",
                             f"{r.joules_per_image:.5f}"])


def readme_rows() -> list[str]:
    text = (REPO / "README.md").read_text()
    # The table sits in an indented code fence under a bullet; take the header
    # line and the rows that follow it up to the closing fence.
    match = re.search(r"^[ \t]*(framework,batch_size,precision.*?)\n[ \t]*```", text, re.S | re.M)
    if not match:
        raise SystemExit("README benchmark table not found")
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


def check_readme(csv_path: Path) -> int:
    expected = readme_rows()
    actual = [line.strip() for line in csv_path.read_text().splitlines() if line.strip()]
    if expected != actual:
        print("MISMATCH between README table and", csv_path)
        for a, b in zip(expected, actual):
            if a != b:
                print("  README:", a, "\n  CSV:   ", b)
        return 1
    print(f"OK: {csv_path} is identical to the README benchmark table ({len(expected) - 1} rows)")
    return 0


def self_test() -> int:
    import tempfile
    failures = []

    def check(name: str, ok: bool) -> None:
        print(("PASS" if ok else "FAIL"), name)
        if not ok:
            failures.append(name)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "b.csv"
        path.write_text(
            "framework,batch_size,precision,mean_latency_ms,p95_latency_ms,throughput_samples_per_sec,gpu_util_percent,gpu_power_watts\n"
            "torch,8,fp32,7,7,1000,,200\n"
            "torch,8,fp16,4,4,2000,,250\n"
            "torch,1,fp32,2,2,500,,\n"                 # no power: skipped
            "numpy-selftest,1,fp32,0,0,256322.62,,\n"  # CPU fallback: skipped
            "torch,2,fp32,2,2,0,,100\n"                # zero throughput: skipped
        )
        rows, skipped = load_rows(path)
        check("skips rows without power or throughput", len(rows) == 2 and skipped == 3)
        fp32 = next(r for r in rows if r.precision == "fp32")
        check("images/s/W = throughput / watts", abs(fp32.images_per_s_per_w - 5.0) < 1e-12)
        check("J/image = watts / throughput", abs(fp32.joules_per_image - 0.2) < 1e-12)
        q = ratios(rows)[0]
        check("fp16/fp32 ratios", abs(q["throughput_ratio"] - 2.0) < 1e-12
              and abs(q["power_ratio"] - 1.25) < 1e-12
              and abs(q["perf_per_watt_ratio"] - 1.6) < 1e-12)
        check("markdown mentions skipped count", "skipped for a missing" in render_markdown(rows, skipped, "b.csv")
              and ": 3." in render_markdown(rows, skipped, "b.csv"))
    # The README's own example: batch 8 fp16, 1990.06 samples/s at 261.69 W.
    check("README batch 8 fp16 value", abs(1990.06 / 261.69 - 7.6047) < 1e-3)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", type=Path, help="benchmark CSV from benchmark.py")
    parser.add_argument("--markdown", type=Path, help="write the table here")
    parser.add_argument("--csv-out", "--csv", dest="csv_out", type=Path, help="write the computed table as CSV")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-readme", action="store_true",
                        help="verify results/resnet50_gpu_benchmark.csv equals the README table")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.check_readme:
        return check_readme(args.csv or REPO / "results" / "resnet50_gpu_benchmark.csv")
    if args.csv is None:
        parser.error("a benchmark CSV is required")
    rows, skipped = load_rows(args.csv)
    if not rows:
        print("no usable rows (every row lacked a power reading)", file=sys.stderr)
        return 1
    try:
        source = str(args.csv.resolve().relative_to(REPO))
    except ValueError:
        source = str(args.csv)
    markdown = render_markdown(rows, skipped, source)
    if args.markdown:
        args.markdown.write_text(markdown)
    else:
        print(markdown)
    if args.csv_out:
        write_csv(rows, args.csv_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
