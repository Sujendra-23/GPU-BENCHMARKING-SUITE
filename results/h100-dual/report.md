# Two-H100 continuation: setup and blockers

The supplied pod initially exposed **two H100 80 GB HBM3 GPUs**, driver
570.124.06, with `NV18` between GPU0 and GPU1. The 40 GB container disk has no
mounted persistent workspace volume. `topology.txt` retains the topology
observed while GPU access worked.

| Phase | Observed result |
| --- | --- |
| TensorRT-LLM environment | Installed 1.0.0 with Python 3.10, PyTorch 2.7.1+cu126 and TensorRT 10.11.0.33; `pip check` passed |
| TensorRT-LLM serving | Did not start: CUDA devices became inaccessible; runner exited 2 |
| vLLM TP1 / TP2 | Not measured on this host; pending restored GPU access and vLLM environment setup |
| Nsight counter probe | Ran before GPU access failed; denied with `ERR_NVGPUCTRPERM` |

NVIDIA's TensorRT-LLM 1.0.0 index did not provide a Python 3.11 wheel, so the
successful installation uses a separate Python 3.10 environment. The earlier
Python 3.11 installation failure is retained in `trtllm-setup.log`; the
successful installation is in `trtllm-cu12-setup.log`. Exact installed versions
are in `trtllm-environment.txt`.

The container subsequently returned `Failed to initialize NVML: Unknown Error`
from `nvidia-smi`. Both its original PyTorch and the isolated environment saw
zero usable GPUs. Opening `/dev/nvidiactl`, `/dev/nvidia5`, `/dev/nvidia7` and
`/dev/nvidia-uvm` returned `EPERM`. `gpu-access-failure.log` retains a repeat of
the NVML and device-open checks. File permissions on those nodes were 0666;
this is a device-access failure beyond ordinary file mode permissions.

These observations are consistent with NVIDIA's documented container GPU
access failure. The exact host-side cause is unverified; container recreation
or provider intervention may be needed. Results and environment manifests
were copied locally before a restart was requested.
[NVIDIA troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error).

No TensorRT-LLM latency, TP2 throughput or hardware counter measurements are
reported from this failed serving attempt. Once GPUs are usable, prepare the
isolated environments in `docs/runpod.md` and run:

```bash
python -m scripts.run_remaining --requests 32 --out-dir results/h100-dual
```
