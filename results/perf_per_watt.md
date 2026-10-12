# Performance per watt (ResNet-50 inference)

Computed by `harness/perf_per_watt.py` from `results/resnet50_gpu_benchmark.csv`. Every input value is
a recorded measurement from the GPU run described in the README; nothing is
estimated or imputed. `images/s/W = throughput / gpu_power_watts` and
`J/image = gpu_power_watts / throughput`.

| framework | batch | precision | images/s | power (W) | images/s/W | J/image |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| torch | 1 | fp16 | 373.87 | 236.55 | 1.581 | 0.6327 |
| torch | 1 | fp32 | 478.54 | 233.65 | 2.048 | 0.4883 |
| torch | 8 | fp16 | 1990.06 | 261.69 | 7.605 | 0.1315 |
| torch | 8 | fp32 | 1134.84 | 258.88 | 4.384 | 0.2281 |
| torch | 32 | fp16 | 1949.95 | 290.58 | 6.711 | 0.1490 |
| torch | 32 | fp32 | 929.77 | 303.15 | 3.067 | 0.3260 |
| torch | 64 | fp16 | 1759.58 | 306.48 | 5.741 | 0.1742 |
| torch | 64 | fp32 | 887.36 | 298.96 | 2.968 | 0.3369 |

## fp16 vs fp32

| framework | batch | throughput x | power x | images/s/W x |
| --- | ---: | ---: | ---: | ---: |
| torch | 1 | 0.78 | 1.01 | 0.77 |
| torch | 8 | 1.75 | 1.01 | 1.73 |
| torch | 32 | 2.10 | 0.96 | 2.19 |
| torch | 64 | 1.98 | 1.03 | 1.93 |

## Caveats

- Rows skipped for a missing or non-positive power/throughput value: 0.
- Power is `nvidia-smi power.draw` sampled during the sweep on a MIG 1g.24gb slice of an
  RTX PRO 6000 Blackwell Server Edition. `power.draw` may describe the whole board rather
  than the slice (not checked in the recorded run), while the workload used only a slice,
  so absolute images/s/W is not a statement about the full GPU's efficiency. The
  fp16-vs-fp32 ratios at the same batch size compare like with like and are the more
  meaningful figures.
- One run per configuration; no confidence intervals, so small differences are not
  statistically supported. Batch 1 fp16 is slower than fp32 in the recorded run.
- The README's TensorRT result (batch 8 fp32, 98.24 W) is a different framework and engine
  and is not included in the table above.
