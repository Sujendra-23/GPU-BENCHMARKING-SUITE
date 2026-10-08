# HIP port of the tiled matmul. Needs ROCm (hipcc); running needs an AMD GPU.
# Set GPU_ARCH to your target (gfx90a = MI200, gfx942 = MI300, gfx1100 = RX 7900,
# gfx1030 = RX 6800); `rocminfo | grep gfx` shows the local one.
GPU_ARCH ?= gfx90a
HIPCC ?= hipcc
REF = ../reference/matmul_ref.c

.PHONY: run compile-check clean

matmul_hip: matmul.hip.cpp $(REF)
	$(HIPCC) -O3 --offload-arch=$(GPU_ARCH) -o $@ matmul.hip.cpp $(REF)

run: matmul_hip
	./matmul_hip 512 512 512
	./matmul_hip 2048 2048 2048

# Compiles device code for several AMD architectures without needing a GPU
# (what CI/this repo can verify on a machine with no AMD hardware).
compile-check:
	@for arch in gfx90a gfx942 gfx1030 gfx1100; do \
	  echo "hipcc --offload-arch=$$arch"; \
	  $(HIPCC) -O3 --offload-arch=$$arch -c matmul.hip.cpp -o /dev/null || exit 1; \
	done

clean:
	rm -f matmul_hip
