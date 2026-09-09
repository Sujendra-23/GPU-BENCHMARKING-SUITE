.PHONY: run clean

run:
	gcc -O2 -o matmul_opencl matmul_host.c ../reference/matmul_ref.c -lOpenCL
	./matmul_opencl 512 512 512

clean:
	rm -f matmul_opencl
