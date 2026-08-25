.PHONY: compare clean

COMPILERS := gcc clang
OPT_LEVELS := O0 O2 O3

compare:
	@echo "=== Compiler / optimization-level comparison (5-point Jacobi stencil) ==="
	@for cc in $(COMPILERS); do \
		for opt in $(OPT_LEVELS); do \
			bin=jacobi_$${cc}_$${opt}; \
			$$cc -$$opt -o $$bin jacobi.c; \
			echo "--- $$cc -$$opt ---"; \
			./$$bin; \
		done; \
	done

clean:
	rm -f jacobi_*
